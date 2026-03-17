"""
Two-stage LDM Trainer for MWIR→LWIR synthesis.

Stage 1 — VAE Fine-tuning  (~50k steps, LWIR only, no pairs needed)
  Goal:  Learn a high-quality LWIR latent space.
  Input: Any LWIR images (paired + unpaired if available)
  Loss:  Reconstruction (L1) + Gabor perceptual + KL (free-bits annealed)

Stage 2 — Conditional DiT Training  (~200k steps, paired data)
  Goal:  Learn p(z_lwir | z_mwir_features) in latent space.
  Input: Paired (MWIR, LWIR) — LWIR encoded by frozen VAE
  Loss:  ε-prediction MSE + CFC + Spectral (all in latent space)

Key design choices for 500-2000 pairs:
  - VAE trained on all LWIR (including unpaired if any) → maximise reconstruction quality
  - DiT trained only on paired data (MWIR conditioning requires pairs)
  - Heavy EMA on DiT (decay=0.9999) — critical at this data scale
  - Classifier-free guidance: 10% of training uses null MWIR conditioning
    → enables guidance at inference for quality boost
"""

import json
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
# from torch.amp import GradScaler, autocast

from pathlib import Path
from typing import Optional

from models.ldm.vae import IRVAE
from models.ldm.dit import ConditionalDiT, DiT_B_4
from models.diffusion_scheduler import DDIMScheduler, CharacteristicFunctionConsistencyLoss, SpectralConsistencyLoss
from training.trainer import EMA, psnr, ssim
from data.dataset import build_dataloaders, MWIRLWIRDataset
from training.visualizer import Visualizer


# ─────────────────────────────────────────────
# KL weight annealing (free bits schedule)
# ─────────────────────────────────────────────

def kl_annealing_weight(step: int, warmup: int = 10_000, max_weight: float = 1e-4) -> float:
    """
    Linear warmup of KL weight from 0 → max_weight over `warmup` steps.
    Prevents posterior collapse in early training.
    """
    return min(1.0, step / warmup) * max_weight


# ─────────────────────────────────────────────
# Stage 1: VAE Trainer
# ─────────────────────────────────────────────

class VAETrainer:
    """
    Fine-tunes the KL-VAE on LWIR imagery.
    Can use unpaired LWIR if available (pass lwir_only_root).

    For 500-2000 pairs: train for 50k steps.
    Convergence criterion: val reconstruction loss < 0.02.
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.device = torch.device(config.get('device', 'cuda'))
        self.output_dir = Path(config['output_dir']) / 'stage1_vae'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0

        # Build VAE
        sd_vae_path = config.get('sd_vae_path', None)
        vae_kwargs = dict(
            in_channels=config.get('lwir_channels', 1),
            ch=config.get('vae_ch', 128),
            ch_mult=tuple(config.get('vae_ch_mult', [1, 2, 4])),
            num_res_blocks=config.get('vae_num_res_blocks', 2),
            z_channels=config.get('z_channels', 4),
            kl_weight=config.get('kl_weight_start', 1e-6),
            kl_weight_max=config.get('kl_weight_max', 1e-4),
        )
        if sd_vae_path:
            self.vae = IRVAE.from_pretrained_sd(sd_vae_path, **vae_kwargs)
        else:
            self.vae = IRVAE(**vae_kwargs)
        self.vae = self.vae.to(self.device)

        n_params = sum(p.numel() for p in self.vae.parameters())
        print(f"[VAE] Parameters: {n_params/1e6:.1f}M")

        # Separate LR for transferred vs new weights (if using SD init)
        if sd_vae_path:
            new_params = list(self.vae.encoder.conv_in.parameters()) + \
                         list(self.vae.decoder.conv_out.parameters())
            base_params = [p for p in self.vae.parameters()
                           if not any(p is q for q in new_params)]
            param_groups = [
                {'params': base_params, 'lr': config.get('vae_lr', 4.5e-6)},
                {'params': new_params, 'lr': config.get('vae_lr', 4.5e-6) * 10},
            ]
        else:
            param_groups = [{'params': self.vae.parameters(), 'lr': config.get('vae_lr', 1e-4)}]

        self.optimizer = AdamW(param_groups, betas=(0.9, 0.999), weight_decay=1e-4)

        total_steps = config.get('vae_total_steps', 50_000)
        self.lr_sched = SequentialLR(
            self.optimizer,
            schedulers=[
                LinearLR(self.optimizer, 1e-3, 1.0, total_iters=1000),
                CosineAnnealingLR(self.optimizer, T_max=total_steps - 1000, eta_min=1e-6),
            ],
            milestones=[1000],
        )
        self.scaler = GradScaler(enabled=True)

        self.train_loader, self.val_loader = build_dataloaders(
            root=config['data_root'],
            image_size=config.get('image_size', 256),
            batch_size=config.get('vae_batch_size', 16),
            num_workers=config.get('num_workers', 4),
            file_ext=config.get('file_ext', 'npy'),
        )
        self.log_path = self.output_dir / 'vae_log.jsonl'

    def train(self):
        cfg = self.cfg
        total = cfg.get('vae_total_steps', 50_000)
        log_every = cfg.get('log_every', 50)
        val_every = cfg.get('val_every', 2000)
        kl_warmup = cfg.get('kl_warmup_steps', 10_000)
        kl_max = cfg.get('kl_weight_max', 1e-4)

        print(f"\n{'='*55}")
        print(f"  Stage 1: VAE Fine-tuning ({total:,} steps)")
        print(f"{'='*55}\n")

        self.vae.train()
        running = {}
        t0 = time.time()

        while self.step < total:
            for batch in self.train_loader:
                if self.step >= total:
                    break

                lwir = batch['lwir'].to(self.device)
                kl_w = kl_annealing_weight(self.step, kl_warmup, kl_max)

                self.optimizer.zero_grad(set_to_none=True)
                with autocast(dtype=torch.bfloat16):
                    loss, loss_dict = self.vae.training_step(lwir, kl_weight=kl_w)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.lr_sched.step()

                for k, v in loss_dict.items():
                    running[k] = running.get(k, 0.0) + v

                if (self.step + 1) % log_every == 0:
                    avg = {k: v / log_every for k, v in running.items()}
                    lr = self.optimizer.param_groups[0]['lr']
                    elapsed = time.time() - t0
                    print(
                        f"Step {self.step+1:>6d} | "
                        f"Recon: {avg.get('recon', 0):.4f} | "
                        f"Perc: {avg.get('perceptual', 0):.4f} | "
                        f"KL: {avg.get('kl', 0):.4f} | "
                        f"KL_w: {kl_w:.2e} | "
                        f"LR: {lr:.2e} | "
                        f"{log_every/elapsed:.1f} it/s"
                    )
                    with open(self.log_path, 'a') as f:
                        f.write(json.dumps({'step': self.step+1, **avg}) + '\n')
                    running = {}
                    t0 = time.time()

                if (self.step + 1) % val_every == 0:
                    self._validate()

                self.step += 1

        # Compute scale factor on training data
        self.vae.compute_scale_factor(self.train_loader, str(self.device))
        self._save()
        print(f"\n[Stage 1] Done. VAE saved to {self.output_dir}")
        return self.vae

    @torch.no_grad()
    def _validate(self):
        self.vae.eval()
        losses = []
        psnr_vals = []
        for i, batch in enumerate(self.val_loader):
            if i >= 20:
                break
            lwir = batch['lwir'].to(self.device)
            recon, _ = self.vae(lwir, sample_posterior=False)
            losses.append(F.l1_loss(recon, lwir).item())
            psnr_vals.append(psnr(recon, lwir))
        self.vae.train()
        print(f"  [Val] Recon L1: {sum(losses)/len(losses):.4f} | PSNR: {sum(psnr_vals)/len(psnr_vals):.2f} dB")

    def _save(self):
        torch.save({
            'vae': self.vae.state_dict(),
            'scale_factor': self.vae.scale_factor,
            'config': self.cfg,
        }, self.output_dir / 'vae_final.pt')


# ─────────────────────────────────────────────
# Stage 2: Conditional DiT Trainer
# ─────────────────────────────────────────────

class DiTTrainer:
    """
    Trains the Conditional DiT in latent space using a frozen VAE.

    Key features:
      - Classifier-free guidance training (10% null conditioning)
      - EMA with warmup
      - CFC + spectral losses in latent space
      - Gradient checkpointing available for large models
    """

    def __init__(self, config: dict, vae: IRVAE):
        self.cfg = config
        self.device = torch.device(config.get('device', 'cuda'))
        self.output_dir = Path(config['output_dir']) / 'stage2_dit'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.global_step = 0

        # Frozen VAE
        self.vae = vae.to(self.device).eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        print(f"[DiT Trainer] VAE frozen. scale_factor = {self.vae.scale_factor:.4f}")

        # DiT
        dit_size = config.get('dit_size', 'B')  # 'S', 'B', 'L'
        dit_builders = {'S': DiT_B_4, 'B': DiT_B_4, 'L': DiT_B_4}  # import all from dit.py
        from models.ldm.dit import DiT_S_4, DiT_B_4, DiT_L_4
        builder = {'S': DiT_S_4, 'B': DiT_B_4, 'L': DiT_L_4}[dit_size]

        self.dit = builder(
            in_channels=config.get('z_channels', 4),
            mwir_channels=config.get('mwir_channels', 1),
            context_dim=config.get('dit_context_dim', 512),
            cond_dim=config.get('dit_cond_dim', 1024),
            vae_f=config.get('vae_f', 4),
            num_registers=config.get('num_registers', 4),
        ).to(self.device)

        n = sum(p.numel() for p in self.dit.parameters())
        print(f"[DiT] Parameters: {n/1e6:.1f}M")

        # Null MWIR token for classifier-free guidance
        self.cfg_prob = config.get('cfg_prob', 0.1)
        null_mwir = torch.zeros(1, config.get('mwir_channels', 1),
                                config.get('image_size', 256),
                                config.get('image_size', 256))
        self.register_buffer_null = null_mwir

        # Diffusion scheduler
        self.scheduler = DDIMScheduler(
            num_train_timesteps=config.get('num_train_timesteps', 1000),
            schedule=config.get('noise_schedule', 'cosine'),
            clip_sample=False,   # clip in latent space can hurt — use in pixel space
            prediction_type='epsilon',
        ).to(self.device)

        # Losses
        self.cfc_loss = CharacteristicFunctionConsistencyLoss(num_freqs=32, patch_size=8).to(self.device)
        self.spectral_loss = SpectralConsistencyLoss().to(self.device)
        self.lambda_cfc = config.get('lambda_cfc', 0.1)
        self.lambda_spectral = config.get('lambda_spectral', 0.05)

        # EMA
        self.ema = EMA(self.dit, decay=config.get('ema_decay', 0.9999))

        # Optimizer
        self.optimizer = AdamW(
            self.dit.parameters(),
            lr=config.get('dit_lr', 1e-4),
            betas=(0.9, 0.999),
            weight_decay=config.get('weight_decay', 1e-4),
        )
        total_steps = config.get('dit_total_steps', 200_000)
        warmup = config.get('warmup_steps', 2_000)
        self.lr_sched = SequentialLR(
            self.optimizer,
            schedulers=[
                LinearLR(self.optimizer, 1e-4, 1.0, warmup),
                CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup, eta_min=1e-6),
            ],
            milestones=[warmup],
        )

        self.scaler = GradScaler(enabled=True)
        self.train_loader, self.val_loader = build_dataloaders(
            root=config['data_root'],
            image_size=config.get('image_size', 256),
            batch_size=config.get('dit_batch_size', 8),
            num_workers=config.get('num_workers', 4),
            file_ext=config.get('file_ext', 'npy'),
        )

        # Non-augmented datasets for deterministic visualisation
        _ds_kwargs = dict(
            root=config['data_root'],
            image_size=config.get('image_size', 256),
            val_frac=config.get('val_frac', 0.1),
            file_ext=config.get('file_ext', 'npy'),
        )
        self._train_ds = MWIRLWIRDataset(augment=False, split='train', **_ds_kwargs)
        self._test_ds  = MWIRLWIRDataset(augment=False, split='val',   **_ds_kwargs)

        self.visualizer = Visualizer(
            train_dataset=self._train_ds,
            test_dataset=self._test_ds,
            n_samples=config.get('vis_n_samples', 8),
            seed=config.get('vis_seed', 42),
            device=str(self.device),
        )
        self._vis_every = config.get('vis_every', config.get('val_every', 2000))
        self._guidance_scale = config.get('inference_defaults', {}).get('guidance_scale', 5.0)
        self._vis_ddim_steps = config.get('val_ddim_steps', 20)

        self.log_path = self.output_dir / 'dit_log.jsonl'

    @torch.no_grad()
    def _encode(self, lwir: torch.Tensor) -> torch.Tensor:
        """Encode LWIR to latent and apply scale factor."""
        posterior = self.vae.encode(lwir)
        z = posterior.sample() * self.vae.scale_factor
        return z

    def train_step(self, batch):
        mwir = batch['mwir'].to(self.device)
        lwir = batch['lwir'].to(self.device)
        B = mwir.shape[0]

        # Encode LWIR to latent (frozen VAE, no grad)
        with torch.no_grad():
            z0 = self._encode(lwir)

        # Classifier-free guidance: randomly drop conditioning
        cfg_mask = torch.rand(B) < self.cfg_prob
        mwir_cond = mwir.clone()
        mwir_cond[cfg_mask] = 0.0   # null conditioning = zero MWIR

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(dtype=torch.bfloat16):
            # Sample timestep and add noise
            t = torch.randint(0, self.scheduler.num_train_timesteps, (B,), device=self.device)
            noise = torch.randn_like(z0)
            zt, _ = self.scheduler.q_sample(z0, t, noise)

            # DiT predicts noise
            noise_pred = self.dit(zt, t, mwir_cond)

            # Primary MSE loss
            mse = F.mse_loss(noise_pred, noise)

            # Predict z0 for data-space losses
            sqrt_a = self.scheduler._extract(self.scheduler.sqrt_alphas_cumprod, t, z0.shape)
            sqrt_1ma = self.scheduler._extract(self.scheduler.sqrt_one_minus_alphas_cumprod, t, z0.shape)
            z0_pred = (zt - sqrt_1ma * noise_pred) / (sqrt_a + 1e-8)

            # CFC + spectral on latent z0 estimates
            cfc = self.cfc_loss(z0_pred, z0)
            spec = self.spectral_loss(z0_pred, z0)

            total = mse + self.lambda_cfc * cfc + self.lambda_spectral * spec

        self.scaler.scale(total).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.dit.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.lr_sched.step()
        self.ema.update()

        return {
            'total': total.item(),
            'mse': mse.item(),
            'cfc': cfc.item(),
            'spectral': spec.item(),
        }

    def train(self):
        cfg = self.cfg
        total = cfg.get('dit_total_steps', 200_000)
        log_every = cfg.get('log_every', 50)
        val_every = cfg.get('val_every', 2000)
        save_every = cfg.get('save_every', 5000)

        print(f"\n{'='*55}")
        print(f"  Stage 2: Conditional DiT ({total:,} steps)")
        print(f"  CFG probability: {self.cfg_prob:.0%}")
        print(f"{'='*55}\n")

        self.dit.train()
        running = {}
        t0 = time.time()

        while self.global_step < total:
            for batch in self.train_loader:
                if self.global_step >= total:
                    break

                loss_dict = self.train_step(batch)
                for k, v in loss_dict.items():
                    running[k] = running.get(k, 0.0) + v

                if (self.global_step + 1) % log_every == 0:
                    avg = {k: v / log_every for k, v in running.items()}
                    lr = self.optimizer.param_groups[0]['lr']
                    elapsed = time.time() - t0
                    print(
                        f"Step {self.global_step+1:>7d} | "
                        f"Total: {avg.get('total',0):.4f} | "
                        f"MSE: {avg.get('mse',0):.4f} | "
                        f"CFC: {avg.get('cfc',0):.4f} | "
                        f"LR: {lr:.2e} | "
                        f"{log_every/elapsed:.1f} it/s"
                    )
                    with open(self.log_path, 'a') as f:
                        f.write(json.dumps({'step': self.global_step+1, **avg}) + '\n')
                    running = {}
                    t0 = time.time()

                if (self.global_step + 1) % val_every == 0:
                    self._validate()

                # ── Visualisation ──
                if (self.global_step + 1) % self._vis_every == 0:
                    self.dit.eval()
                    vis_psnr = self.visualizer.save_both(
                        step=self.global_step + 1,
                        generate_fn=self._generate_fn,
                        output_dir=self.output_dir,
                    )
                    self.dit.train()
                    if self.visualizer.is_best(vis_psnr.get('test'), 'test'):
                        self._save('best')

                if (self.global_step + 1) % save_every == 0:
                    self._save(f'step_{self.global_step+1:07d}')

                self.global_step += 1

        self._save('final')
        # Final visualisation pass
        self.dit.eval()
        self.visualizer.save_both(
            step=self.global_step + 1,
            generate_fn=self._generate_fn,
            output_dir=self.output_dir,
        )
        self.dit.train()

    # ─── Generate function for Visualizer ────────────────────────

    @torch.no_grad()
    def _generate_fn(self, mwir: torch.Tensor) -> torch.Tensor:
        """
        Full DDIM + CFG sampling over a fixed MWIR batch.
        Decodes the resulting latent via the frozen VAE decoder.
        mwir: (B, C, H, W) on self.device
        Returns: (B, 1, H, W) pixel-space LWIR, clamped [-1,1], on CPU.
        """
        B, _, H, W = mwir.shape
        vae_f      = self.cfg.get('vae_f', 4)
        z_channels = self.cfg.get('z_channels', 4)
        H_lat, W_lat = H // vae_f, W // vae_f

        z = torch.randn(B, z_channels, H_lat, W_lat, device=self.device)
        null_mwir = torch.zeros_like(mwir)

        timesteps = torch.linspace(
            self.scheduler.num_train_timesteps - 1, 0,
            self._vis_ddim_steps, dtype=torch.long, device=self.device,
        )

        for i, t_cur in enumerate(timesteps):
            t_b = t_cur.expand(B)
            cond_eps   = self.dit(z, t_b, mwir)
            uncond_eps = self.dit(z, t_b, null_mwir)
            eps = uncond_eps + self._guidance_scale * (cond_eps - uncond_eps)
            z = self.scheduler._ddim_step(z, t_cur, eps, timesteps, i, eta=0.0)

        return self.vae.decode(z).clamp(-1, 1).cpu()

    @torch.no_grad()
    def _validate(self, num_steps: int = 20, guidance_scale: float = 3.0):
        """Full DDIM sampling + PSNR/SSIM on a few validation batches."""
        self.dit.eval()
        psnr_vals, ssim_vals = [], []

        for i, batch in enumerate(self.val_loader):
            if i >= 4:
                break
            mwir = batch['mwir'][:4].to(self.device)
            lwir = batch['lwir'][:4].to(self.device)

            z0_real = self._encode(lwir)
            B, C, H, W = z0_real.shape
            z = torch.randn_like(z0_real)

            timesteps = torch.linspace(
                self.scheduler.num_train_timesteps - 1, 0,
                num_steps, dtype=torch.long, device=self.device
            )
            for step_i, t_cur in enumerate(timesteps):
                t_b = t_cur.expand(B)
                cond_out   = self.dit(z, t_b, mwir)
                uncond_out = self.dit(z, t_b, torch.zeros_like(mwir))
                eps = uncond_out + guidance_scale * (cond_out - uncond_out)
                z = self.scheduler._ddim_step(z, t_cur, eps, timesteps, step_i, eta=0.0)

            gen_lwir = self.vae.decode(z)
            for j in range(gen_lwir.shape[0]):
                psnr_vals.append(psnr(gen_lwir[j:j+1], lwir[j:j+1]))
                ssim_vals.append(ssim(gen_lwir[j:j+1], lwir[j:j+1]))

        self.dit.train()
        import numpy as np
        print(
            f"\n  [Val] PSNR: {np.mean(psnr_vals):.2f} dB | "
            f"SSIM: {np.mean(ssim_vals):.4f} "
            f"(cfg={guidance_scale})\n"
        )

    def _save(self, tag: str):
        ckpt_dir = self.output_dir / 'checkpoints'
        ckpt_dir.mkdir(exist_ok=True)
        torch.save({
            'step': self.global_step,
            'dit': self.dit.state_dict(),
            'ema': self.ema.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.cfg,
        }, ckpt_dir / f'dit_{tag}.pt')
        print(f"  [Checkpoint] → checkpoints/dit_{tag}.pt")
        # Prune old periodic checkpoints; keep best and final untouched
        if tag.startswith('step_'):
            for old in sorted(ckpt_dir.glob('dit_step_*.pt'))[:-3]:
                old.unlink()


# ─────────────────────────────────────────────
# Unified entrypoint
# ─────────────────────────────────────────────

def train_ldm(config: dict, skip_vae: bool = False, vae_ckpt: Optional[str] = None):
    """
    Run both training stages.

    Args:
        skip_vae:  If True, load VAE from vae_ckpt and go straight to Stage 2.
        vae_ckpt:  Path to saved VAE checkpoint (for skip_vae=True).
    """
    device = torch.device(config.get('device', 'cuda'))

    if skip_vae and vae_ckpt:
        print(f"[LDM] Loading pre-trained VAE from {vae_ckpt}")
        ckpt = torch.load(vae_ckpt, map_location=device)
        vae = IRVAE(
            in_channels=config.get('lwir_channels', 1),
            ch=config.get('vae_ch', 128),
            ch_mult=tuple(config.get('vae_ch_mult', [1, 2, 4])),
            z_channels=config.get('z_channels', 4),
        )
        vae.load_state_dict(ckpt['vae'])
        vae.scale_factor = ckpt['scale_factor']
    else:
        vae_trainer = VAETrainer(config)
        vae = vae_trainer.train()

    dit_trainer = DiTTrainer(config, vae)
    dit_trainer.train()
