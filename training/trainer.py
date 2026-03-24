"""
Training loop for MWIR→LWIR conditional diffusion model.

Features:
  - Mixed precision (bfloat16 / float16)
  - Exponential Moving Average (EMA) model
  - Gradient clipping
  - Comprehensive metric logging (SSIM, PSNR, FID proxy, CFC loss breakdown)
  - Checkpoint management
  - Validation-time DDIM sampling with quality metrics
"""

import os
import math
import time
import json
import shutil
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import GradScaler, autocast
import numpy as np

from models.conditional_unet import ConditionalUNet
from models.diffusion_scheduler import DDIMScheduler, DiffusionLoss
from data.dataset import build_dataloaders, MWIRLWIRDataset
from training.visualizer import Visualizer


# ─────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────

class EMA:
    """
    Exponential Moving Average of model weights.
    Use EMA model for inference — it's significantly more stable.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup_steps: int = 100):
        self.model = model
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step = 0
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}

    def update(self):
        self.step += 1
        # Linear warmup for EMA decay
        decay = min(self.decay, (1 + self.step) / (10 + self.step))
        with torch.no_grad():
            for k, v in self.model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k] = decay * self.shadow[k] + (1 - decay) * v.float()

    def apply_shadow(self):
        """Apply EMA weights to model."""
        for k, v in self.shadow.items():
            if k in dict(self.model.named_parameters()):
                self.model.state_dict()[k].copy_(v.to(self.model.state_dict()[k].dtype))

    def state_dict(self):
        return {'shadow': self.shadow, 'step': self.step, 'decay': self.decay}

    def load_state_dict(self, d):
        self.shadow = d['shadow']
        self.step = d['step']
        self.decay = d['decay']


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 2.0) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val / math.sqrt(mse))


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01**2,
    C2: float = 0.03**2,
) -> float:
    """Simplified SSIM (single-scale, single-channel compatible)."""
    mu_p = F.avg_pool2d(pred, window_size, 1, window_size//2)
    mu_t = F.avg_pool2d(target, window_size, 1, window_size//2)
    mu_p2, mu_t2, mu_pt = mu_p**2, mu_t**2, mu_p * mu_t

    sig_p2 = F.avg_pool2d(pred**2, window_size, 1, window_size//2) - mu_p2
    sig_t2 = F.avg_pool2d(target**2, window_size, 1, window_size//2) - mu_t2
    sig_pt = F.avg_pool2d(pred * target, window_size, 1, window_size//2) - mu_pt

    num = (2 * mu_pt + C1) * (2 * sig_pt + C2)
    den = (mu_p2 + mu_t2 + C1) * (sig_p2 + sig_t2 + C2)
    return (num / den).mean().item()


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────

class Trainer:
    def __init__(self, config: dict):
        self.cfg = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.output_dir = Path(config.get('output_dir', 'runs/mwir2lwir'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.global_step = 0
        self.epoch = 0

        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # ── Model ──
        self.model = ConditionalUNet(
            in_channels=config.get('lwir_channels', 1),
            mwir_channels=config.get('mwir_channels', 1),
            base_channels=config.get('base_channels', 128),
            channel_mults=tuple(config.get('channel_mults', [1, 2, 4, 8])),
            attn_resolutions=tuple(config.get('attn_resolutions', [16, 8])),
            num_res_blocks=config.get('num_res_blocks', 2),
            dropout=config.get('dropout', 0.1),
            use_cross_attn=config.get('use_cross_attn', True),
            image_size=config.get('image_size', 256),
        ).to(self.device)

        print(f"[Model] Parameters: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M")

        # ── Diffusion ──
        self.scheduler = DDIMScheduler(
            num_train_timesteps=config.get('num_train_timesteps', 1000),
            schedule=config.get('noise_schedule', 'cosine'),
            clip_sample=config.get('clip_sample', True),
            prediction_type=config.get('prediction_type', 'epsilon'),
        ).to(self.device)

        # ── Loss ──
        self.criterion = DiffusionLoss(
            lambda_cfc=config.get('lambda_cfc', 0.1),
            lambda_spectral=config.get('lambda_spectral', 0.05),
            cfc_patch_size=config.get('cfc_patch_size', 16),
        ).to(self.device)

        # ── EMA ──
        self.ema = EMA(self.model, decay=config.get('ema_decay', 0.9999))

        # ── Optimizer ──
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get('lr', 1e-4),
            betas=(0.9, 0.999),
            weight_decay=config.get('weight_decay', 1e-4),
        )

        # ── LR Schedule: linear warmup + cosine decay ──
        warmup_steps = config.get('warmup_steps', 1000)
        total_steps = config.get('total_steps', 200_000)
        self.scheduler_lr = SequentialLR(
            self.optimizer,
            schedulers=[
                LinearLR(self.optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_steps),
                CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6),
            ],
            milestones=[warmup_steps],
        )

        # ── Precision ──────────────────────────────────────────────
        # "float32"  — numerically safe, recommended for this pipeline.
        #              fft2 (SpectralConsistencyLoss), GroupNorm on homogeneous IR,
        #              and CFC cos/sin all have bfloat16 instabilities.
        # "bfloat16" — ~1.5× faster on A100/H100, lower memory.
        #              Safe only after the float32 upcasts in loss classes.
        #              Use when throughput matters and precision is verified.
        precision = config.get('precision', 'float32')
        self.use_amp  = (precision == 'bfloat16') and (self.device.type == 'cuda')
        self.amp_dtype = torch.bfloat16 if self.use_amp else torch.float32

        # GradScaler for mixed-precision training.
        # NOTE: GradScaler is only useful with float16.  With bfloat16 it
        # detects false inf/nan at its default 65536× scale and skips every
        # optimizer step, so the model never trains.  We create it here
        # (enabled=False for bfloat16/float32) so checkpoint save/load works.
        self.scaler = GradScaler(self.device.type, enabled=(precision == 'float16'))

        if self.use_amp:
            print("[Trainer] Precision: bfloat16 (AMP enabled)")
        else:
            print("[Trainer] Precision: float32")

        # ── Data ──
        self.train_loader, self.val_loader = build_dataloaders(
            root=config['data_root'],
            image_size=config.get('image_size', 256),
            batch_size=config.get('batch_size', 8),
            num_workers=config.get('num_workers', 4),
            use_lcn=config.get('use_lcn', False),
            val_frac=config.get('val_frac', 0.1),
            file_ext=config.get('file_ext', 'npy'),
        )

        # Keep dataset references for Visualizer (no augmentation for vis samples)
        _ds_kwargs = dict(
            root=config['data_root'],
            image_size=config.get('image_size', 256),
            use_lcn=config.get('use_lcn', False),
            val_frac=config.get('val_frac', 0.1),
            file_ext=config.get('file_ext', 'npy'),
        )
        self._train_ds = MWIRLWIRDataset(augment=False, split='train', **_ds_kwargs)
        self._test_ds  = MWIRLWIRDataset(augment=False, split='val',   **_ds_kwargs)

        # ── Visualizer ──
        self.visualizer = Visualizer(
            train_dataset=self._train_ds,
            test_dataset=self._test_ds,
            n_samples=config.get('vis_n_samples', 8),
            seed=config.get('vis_seed', 42),
            device=str(self.device),
        )
        self._vis_every = config.get('vis_every', config.get('val_every', 2000))
        self._best_psnr = -float('inf')

        self.log_path = self.output_dir / 'train_log.jsonl'

    # ─── Training step ───────────────────────

    def train_step(self, batch):
        mwir = batch['mwir'].to(self.device)
        lwir = batch['lwir'].to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        noise_pred, noise_target, x0_pred, x0_target, t = \
            self.scheduler.training_losses(self.model, lwir, mwir)
        total_loss, loss_dict = self.criterion(
            noise_pred, noise_target, x0_pred, x0_target
        )

        if not torch.isfinite(total_loss):
            print(f"[Warning] NaN/Inf loss at step {self.global_step}: {loss_dict}")
            return {k: float('nan') for k in loss_dict}

        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler_lr.step()
        self.ema.update()

        return loss_dict

    # ─── Validation ──────────────────────────

    @torch.no_grad()
    def validate(self, num_inference_steps: int = 50):
        self.model.eval()
        psnr_vals, ssim_vals = [], []
        val_loss_total = 0.0
        n = 0

        for batch in self.val_loader:
            mwir = batch['mwir'].to(self.device)
            lwir = batch['lwir'].to(self.device)
            B = mwir.shape[0]

            # Compute validation loss (single forward, not full sampling)
            t = torch.randint(0, self.scheduler.num_train_timesteps, (B,), device=self.device)
            noise = torch.randn_like(lwir)
            xt, _ = self.scheduler.q_sample(lwir, t, noise)
            with autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                noise_pred = self.model(xt, t, mwir)
            loss, _ = self.criterion(noise_pred, noise)
            val_loss_total += loss.item() * B

            # Full DDIM synthesis for first batch only (expensive)
            if n == 0:
                generated = self.scheduler.ddim_sample(
                    self.model, mwir[:4],
                    shape=(min(4, B), lwir.shape[1], lwir.shape[2], lwir.shape[3]),
                    num_inference_steps=num_inference_steps,
                    device=str(self.device),
                    verbose=False,
                )
                for i in range(generated.shape[0]):
                    psnr_vals.append(psnr(generated[i:i+1], lwir[i:i+1]))
                    ssim_vals.append(ssim(generated[i:i+1], lwir[i:i+1]))

            n += B
            if n >= 128:  # limit validation for speed
                break

        self.model.train()
        return {
            'val_loss': val_loss_total / n,
            'psnr': float(np.mean(psnr_vals)) if psnr_vals else 0.0,
            'ssim': float(np.mean(ssim_vals)) if ssim_vals else 0.0,
        }

    # ─── Main train loop ─────────────────────

    def train(self):
        cfg = self.cfg
        total_steps = cfg.get('total_steps', 200_000)
        log_every = cfg.get('log_every', 50)
        val_every = cfg.get('val_every', 2000)
        save_every = cfg.get('save_every', 5000)
        val_ddim_steps = cfg.get('val_ddim_steps', 20)

        print(f"\n{'='*60}")
        print(f"  MWIR→LWIR Diffusion Training")
        print(f"  Device: {self.device}")
        print(f"  Total steps: {total_steps:,}")
        print(f"{'='*60}\n")

        self.model.train()
        running_losses = {}
        t0 = time.time()

        while self.global_step < total_steps:
            for batch in self.train_loader:
                if self.global_step >= total_steps:
                    break

                loss_dict = self.train_step(batch)

                # Accumulate running averages
                for k, v in loss_dict.items():
                    running_losses[k] = running_losses.get(k, 0.0) + v

                # ── Logging ──
                if (self.global_step + 1) % log_every == 0:
                    elapsed = time.time() - t0
                    steps_per_sec = log_every / elapsed
                    lr = self.optimizer.param_groups[0]['lr']
                    avg = {k: v / log_every for k, v in running_losses.items()}
                    log_row = {
                        'step': self.global_step + 1,
                        'lr': lr,
                        'steps_per_sec': round(steps_per_sec, 2),
                        **{f'train/{k}': round(v, 6) for k, v in avg.items()},
                    }
                    with open(self.log_path, 'a') as f:
                        f.write(json.dumps(log_row) + '\n')
                    print(
                        f"Step {self.global_step+1:>7d} | "
                        f"Loss: {avg.get('total', 0):.4f} | "
                        f"MSE: {avg.get('mse', 0):.4f} | "
                        f"CFC: {avg.get('cfc', 0):.4f} | "
                        f"Spec: {avg.get('spectral', 0):.4f} | "
                        f"LR: {lr:.2e} | "
                        f"{steps_per_sec:.1f} it/s"
                    )
                    running_losses = {}
                    t0 = time.time()

                # ── Validation ──
                if (self.global_step + 1) % val_every == 0:
                    print(f"\n[Validation] step {self.global_step+1}...")
                    val_metrics = self.validate(val_ddim_steps)
                    log_row = {'step': self.global_step + 1, **{f'val/{k}': v for k, v in val_metrics.items()}}
                    with open(self.log_path, 'a') as f:
                        f.write(json.dumps(log_row) + '\n')
                    print(
                        f"  Val Loss: {val_metrics['val_loss']:.4f} | "
                        f"PSNR: {val_metrics['psnr']:.2f} dB | "
                        f"SSIM: {val_metrics['ssim']:.4f}\n"
                    )

                # ── Visualisation ──
                if (self.global_step + 1) % self._vis_every == 0:
                    self.model.eval()
                    vis_psnr = self.visualizer.save_both(
                        step=self.global_step + 1,
                        generate_fn=self._generate_fn,
                        output_dir=self.output_dir,
                    )
                    self.model.train()
                    # Save best checkpoint based on test-split PSNR
                    test_psnr = vis_psnr.get('test')
                    if self.visualizer.is_best(test_psnr, 'test'):
                        self.save_checkpoint(tag='best')

                # ── Checkpoint ──
                if (self.global_step + 1) % save_every == 0:
                    self.save_checkpoint()

                self.global_step += 1
            self.epoch += 1

        # Final save
        self.save_checkpoint(final=True)
        # Final visualisation pass
        self.model.eval()
        self.visualizer.save_both(
            step=self.global_step + 1,
            generate_fn=self._generate_fn,
            output_dir=self.output_dir,
        )
        self.model.train()
        print("\nTraining complete.")

    # ─── Generate function for Visualizer ───────────────────────

    def _generate_fn(self, mwir: torch.Tensor) -> torch.Tensor:
        """DDIM sampling over the fixed visualiser batch."""
        B, _, H, W = mwir.shape
        lwir_channels = self.cfg.get('lwir_channels', 1)
        return self.scheduler.ddim_sample(
            self.model, mwir,
            shape=(B, lwir_channels, H, W),   # use lwir_channels, not mwir shape
            num_inference_steps=self.cfg.get('val_ddim_steps', 20),
            device=str(self.device),
            verbose=False,
        )

    # ─── Checkpointing ───────────────────────

    def save_checkpoint(self, final: bool = False, tag: Optional[str] = None):
        if tag is None:
            tag = 'final' if final else f'step_{self.global_step+1:07d}'
        ckpt_dir = self.output_dir / 'checkpoints'
        ckpt_dir.mkdir(exist_ok=True)
        path = ckpt_dir / f'ckpt_{tag}.pt'
        torch.save({
            'step': self.global_step,
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'config': self.cfg,
        }, path)
        print(f"[Checkpoint] Saved → {path}")

        # Keep only last 3 periodic checkpoints (best and final are never pruned)
        if tag.startswith('step_'):
            ckpts = sorted(ckpt_dir.glob('ckpt_step_*.pt'))
            for old in ckpts[:-3]:
                old.unlink()

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.ema.load_state_dict(ckpt['ema'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scaler.load_state_dict(ckpt['scaler'])
        self.global_step = ckpt['step']
        self.epoch = ckpt['epoch']
        print(f"[Checkpoint] Loaded from step {self.global_step}")
