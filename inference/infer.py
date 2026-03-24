"""
Inference pipeline for MWIR→LWIR synthesis.

Features:
  - DDIM deterministic fast sampling (50 steps vs 1000 DDPM)
  - Stochastic ensemble (run N times, average predictions for PSNR boost)
  - Sliding-window patch inference for large images
  - Quantitative evaluation: PSNR, SSIM, ERGAS, SAM, CFC distance
  - GeoTIFF output support (preserves geolocation metadata)
  - Scene-adaptive inference (swath alignment + histogram calibration
    + optional LoRA fine-tuning) when real LWIR overlap is available
"""

import math
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F

from models.conditional_unet import ConditionalUNet
from models.diffusion_scheduler import DDIMScheduler, CharacteristicFunctionConsistencyLoss


# ─────────────────────────────────────────────
# Full-image patch inference
# ─────────────────────────────────────────────

class PatchInference:
    """
    Sliding-window patch inference for images larger than the training resolution.
    Uses cosine blending weights to eliminate tile boundary artifacts.

    Args:
        model:          trained ConditionalUNet
        scheduler:      DDIMScheduler
        patch_size:     spatial size model was trained on
        overlap:        overlap fraction between patches [0, 1)
        num_steps:      DDIM sampling steps
        eta:            DDIM stochasticity (0=deterministic)
        device:         torch device
    """

    def __init__(
        self,
        model: ConditionalUNet,
        scheduler: DDIMScheduler,
        patch_size: int = 256,
        overlap: float = 0.25,
        num_steps: int = 50,
        eta: float = 0.0,
        device: str = 'cuda',
    ):
        self.model = model
        self.scheduler = scheduler
        self.patch_size = patch_size
        self.stride = int(patch_size * (1 - overlap))
        self.num_steps = num_steps
        self.eta = eta
        self.device = torch.device(device)

        # Cosine blending weight mask
        w = torch.hann_window(patch_size)
        self.blend_mask = (w.unsqueeze(0) * w.unsqueeze(1)).to(self.device)  # (H, W)

    def _sample_patch(self, mwir_patch: torch.Tensor) -> torch.Tensor:
        """Generate one LWIR patch from MWIR conditioning patch."""
        B, C, H, W = mwir_patch.shape
        generated = self.scheduler.ddim_sample(
            self.model, mwir_patch,
            shape=(B, 1, H, W),
            num_inference_steps=self.num_steps,
            eta=self.eta,
            device=str(self.device),
            verbose=False,
        )
        return generated

    def __call__(self, mwir: torch.Tensor) -> torch.Tensor:
        """
        mwir: (1, C, H, W) full-resolution MWIR image.
        Returns: (1, 1, H, W) generated LWIR image.
        """
        mwir = mwir.to(self.device)
        _, C, H, W = mwir.shape
        p = self.patch_size
        s = self.stride

        output = torch.zeros(1, 1, H, W, device=self.device)
        weight = torch.zeros(1, 1, H, W, device=self.device)

        # Pad if needed
        pad_h = max(0, p - H)
        pad_w = max(0, p - W)
        if pad_h > 0 or pad_w > 0:
            mwir = F.pad(mwir, (0, pad_w, 0, pad_h), mode='reflect')
            output = F.pad(output, (0, pad_w, 0, pad_h))
            weight = F.pad(weight, (0, pad_w, 0, pad_h))

        _, _, H2, W2 = mwir.shape
        ys = list(range(0, H2 - p + 1, s)) + ([H2 - p] if H2 > p else [])
        xs = list(range(0, W2 - p + 1, s)) + ([W2 - p] if W2 > p else [])
        ys = sorted(set(ys))
        xs = sorted(set(xs))

        total = len(ys) * len(xs)
        done = 0
        for y in ys:
            for x in xs:
                patch = mwir[:, :, y:y+p, x:x+p]
                gen = self._sample_patch(patch)
                mask = self.blend_mask.unsqueeze(0).unsqueeze(0)
                output[:, :, y:y+p, x:x+p] += gen * mask
                weight[:, :, y:y+p, x:x+p] += mask
                done += 1
                print(f"  Patch {done}/{total}", end='\r')

        result = output / weight.clamp(min=1e-6)

        # Crop back
        if pad_h > 0 or pad_w > 0:
            result = result[:, :, :H, :W]

        print()
        return result


# ─────────────────────────────────────────────
# Ensemble inference
# ─────────────────────────────────────────────

def ensemble_inference(
    model: ConditionalUNet,
    scheduler: DDIMScheduler,
    mwir: torch.Tensor,
    n_ensemble: int = 5,
    num_steps: int = 50,
    device: str = 'cuda',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run multiple stochastic DDIM samples (eta > 0) and average.
    Returns (mean_prediction, std_prediction) — std is an uncertainty map.

    Uses Welford's online algorithm to avoid storing all N samples in memory,
    which would OOM on large images (e.g. 5 × 1 × 1024 × 1024 = 5.2 GB).
    """
    mean = torch.zeros(mwir.shape[0], 1, mwir.shape[2], mwir.shape[3],
                        device=device)
    m2 = torch.zeros_like(mean)
    for i in range(n_ensemble):
        s = scheduler.ddim_sample(
            model, mwir,
            shape=(mwir.shape[0], 1, mwir.shape[2], mwir.shape[3]),
            num_inference_steps=num_steps,
            eta=0.5,  # partial stochasticity for diversity
            device=device,
            verbose=False,
        )
        delta = s - mean
        mean += delta / (i + 1)
        delta2 = s - mean
        m2 += delta * delta2
    variance = m2 / max(n_ensemble - 1, 1)
    return mean, variance.sqrt()


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute PSNR, SSIM, ERGAS, SAM, and CFC distance."""
    pred_np = pred.squeeze().cpu().numpy()
    tgt_np = target.squeeze().cpu().numpy()

    mse = float(F.mse_loss(pred, target).item())
    psnr = 20 * math.log10(2.0 / math.sqrt(mse)) if mse > 0 else float('inf')

    # SSIM
    C1, C2 = (0.01 * 2)**2, (0.03 * 2)**2
    mu_p = F.avg_pool2d(pred, 11, 1, 5)
    mu_t = F.avg_pool2d(target, 11, 1, 5)
    sig_p = F.avg_pool2d(pred**2, 11, 1, 5) - mu_p**2
    sig_t = F.avg_pool2d(target**2, 11, 1, 5) - mu_t**2
    sig_pt = F.avg_pool2d(pred * target, 11, 1, 5) - mu_p * mu_t
    ssim = float(((2*mu_p*mu_t + C1)*(2*sig_pt + C2) / ((mu_p**2+mu_t**2+C1)*(sig_p+sig_t+C2))).mean())

    # ERGAS (relative global error in synthesis)
    rmse = math.sqrt(mse)
    mean_t = float(target.mean().abs().item()) + 1e-8
    ergas = 100 * rmse / mean_t

    # CFC distance
    cfc_fn = CharacteristicFunctionConsistencyLoss(num_freqs=32, patch_size=16)
    cfc_dist = float(cfc_fn(pred, target).item())

    return {
        'psnr_db': round(psnr, 3),
        'ssim': round(ssim, 4),
        'ergas': round(ergas, 4),
        'mse': round(mse, 6),
        'cfc_dist': round(cfc_dist, 6),
    }


# ─────────────────────────────────────────────
# Main inference entrypoint
# ─────────────────────────────────────────────

def run_inference(
    checkpoint_path: str,
    mwir_path: str,
    output_path: str,
    lwir_path: Optional[str] = None,       # real LWIR (overlap strip) for SAI
    patch_size: int = 256,
    overlap: float = 0.25,
    num_steps: int = 50,
    eta: float = 0.0,
    n_ensemble: int = 1,
    device: str = 'cuda',
    # ── Scene-adaptive options ──────────────────────────────────
    mwir_swath_km: float = 0.0,            # 0 = disable SAI, >0 = enable
    lwir_swath_km: float = 0.0,
    use_histogram_cal: bool = True,
    use_scene_finetuning: bool = False,
    finetune_steps: int = 100,
    finetune_lr: float = 1e-4,
    lora_r: int = 4,
    scene_id: str = 'scene',
):
    """
    End-to-end inference on a single MWIR image.

    Scene-adaptive mode activates automatically when mwir_swath_km > 0
    AND lwir_path is provided (the real LWIR overlap strip).
    Without those two, behaviour is identical to the original pipeline.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # ── Load checkpoint ──────────────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt['config']
    print(f"[Inference] Loaded checkpoint (step {ckpt['step']})")

    # ── Build model ──────────────────────────────────────────────
    model = ConditionalUNet(
        in_channels=cfg.get('lwir_channels', 1),
        mwir_channels=cfg.get('mwir_channels', 1),
        base_channels=cfg.get('base_channels', 128),
        channel_mults=tuple(cfg.get('channel_mults', [1, 2, 4, 8])),
        attn_resolutions=tuple(cfg.get('attn_resolutions', [16, 8])),
        num_res_blocks=cfg.get('num_res_blocks', 2),
        use_cross_attn=cfg.get('use_cross_attn', True),
        image_size=patch_size,
    ).to(device)

    ema_state = ckpt['ema']['shadow']
    model.load_state_dict(
        {k: v.to(next(model.parameters()).dtype) for k, v in ema_state.items()}
    )
    model.eval()

    scheduler = DDIMScheduler(
        num_train_timesteps=cfg.get('num_train_timesteps', 1000),
        schedule=cfg.get('noise_schedule', 'cosine'),
        prediction_type=cfg.get('prediction_type', 'epsilon'),
    ).to(device)

    # ── Load raw arrays ──────────────────────────────────────────
    from data.dataset import percentile_normalize

    mwir_arr = np.load(mwir_path).astype(np.float32)
    if mwir_arr.ndim == 2:
        mwir_arr = mwir_arr[np.newaxis]     # → (1, H, W)

    lwir_arr = None
    if lwir_path:
        lwir_arr = np.load(lwir_path).astype(np.float32)
        if lwir_arr.ndim == 2:
            lwir_arr = lwir_arr[np.newaxis]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Decide pipeline ──────────────────────────────────────────
    use_sai = (mwir_swath_km > 0 and lwir_swath_km > 0 and lwir_arr is not None)

    if use_sai:
        # ── Scene-Adaptive Inference path ────────────────────────
        from inference.scene_adaptive import SceneAdaptiveInference

        # generate_fn expected by SAI: (B,C,H,W) tensor → (B,1,H,W) tensor
        inferencer = PatchInference(
            model, scheduler,
            patch_size=patch_size, overlap=overlap,
            num_steps=num_steps, eta=eta, device=str(device),
        )

        def generate_fn(mwir_t: torch.Tensor) -> torch.Tensor:
            return inferencer(mwir_t)

        sai = SceneAdaptiveInference(
            model=model,
            scheduler=scheduler,
            generate_fn=generate_fn,
            mwir_swath_km=mwir_swath_km,
            lwir_swath_km=lwir_swath_km,
            use_histogram_cal=use_histogram_cal,
            use_scene_finetuning=use_scene_finetuning,
            finetune_steps=finetune_steps,
            finetune_lr=finetune_lr,
            lora_r=lora_r,
            device=str(device),
            output_dir=str(out_path.parent),
        )
        result = sai.run(
            mwir_raw=mwir_arr,
            lwir_raw=lwir_arr,
            scene_id=scene_id,
        )
        # SAI already saves all outputs; also write the primary .npy
        np.save(out_path.with_suffix('.npy'),
                result['lwir_full'].astype(np.float32))
        print(f"[Inference] SAI complete. Full swath → {out_path.with_suffix('.npy')}")
        return result['lwir_full']

    else:
        # ── Standard inference path (unchanged) ──────────────────
        mwir_norm = np.stack(
            [percentile_normalize(mwir_arr[c]) for c in range(mwir_arr.shape[0])]
        )
        mwir_t = torch.from_numpy(mwir_norm).unsqueeze(0).to(device)
        print(f"[Inference] MWIR shape: {mwir_t.shape}, device: {device}")

        inferencer = PatchInference(
            model, scheduler,
            patch_size=patch_size, overlap=overlap,
            num_steps=num_steps, eta=eta, device=str(device),
        )

        if n_ensemble > 1:
            print(f"[Inference] Ensemble N={n_ensemble}...")
            if mwir_t.shape[-2] <= patch_size and mwir_t.shape[-1] <= patch_size:
                mean_pred, std_pred = ensemble_inference(
                    model, scheduler, mwir_t, n_ensemble, num_steps, str(device)
                )
            else:
                preds = [inferencer(mwir_t) for _ in range(n_ensemble)]
                stacked = torch.stack(preds)
                mean_pred, std_pred = stacked.mean(0), stacked.std(0)
        else:
            mean_pred = inferencer(mwir_t)
            std_pred  = None

        result_np = mean_pred.squeeze().cpu().numpy()
        np.save(out_path.with_suffix('.npy'), result_np)
        if std_pred is not None:
            np.save(
                out_path.parent / (out_path.stem + '_uncertainty.npy'),
                std_pred.squeeze().cpu().numpy(),
            )
        print(f"[Inference] Saved → {out_path.with_suffix('.npy')}")

        # Metrics vs ground-truth LWIR (full-swath, no alignment needed here)
        if lwir_arr is not None:
            lwir_norm = np.stack(
                [percentile_normalize(lwir_arr[c]) for c in range(lwir_arr.shape[0])]
            )
            lwir_t = torch.from_numpy(lwir_norm).unsqueeze(0).to(device)
            if lwir_t.shape[-2:] != mean_pred.shape[-2:]:
                lwir_t = F.interpolate(
                    lwir_t, size=mean_pred.shape[-2:],
                    mode='bilinear', align_corners=False,
                )
            metrics = compute_metrics(mean_pred, lwir_t)
            print("\n[Metrics]")
            for k, v in metrics.items():
                print(f"  {k:>12s}: {v}")
            with open(out_path.parent / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)

        return mean_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MWIR→LWIR Inference')
    parser.add_argument('--checkpoint',          required=True)
    parser.add_argument('--mwir',                required=True)
    parser.add_argument('--output',              required=True)
    # Real LWIR — used both as gt metrics source AND as SAI overlap input
    parser.add_argument('--lwir',                default=None,
                        help='Path to real LWIR .npy (overlap strip for SAI, or full-swath GT for metrics)')
    parser.add_argument('--patch_size',          type=int,   default=256)
    parser.add_argument('--overlap',             type=float, default=0.25)
    parser.add_argument('--num_steps',           type=int,   default=50)
    parser.add_argument('--eta',                 type=float, default=0.0)
    parser.add_argument('--ensemble',            type=int,   default=1)
    parser.add_argument('--device',              default='cuda')
    # ── Scene-adaptive options ──────────────────────────────────
    parser.add_argument('--mwir_swath_km',       type=float, default=0.0,
                        help='MWIR swath width in km. Set >0 with --lwir to enable SAI.')
    parser.add_argument('--lwir_swath_km',       type=float, default=0.0,
                        help='LWIR swath width in km.')
    parser.add_argument('--no_histogram_cal',    action='store_true',
                        help='Disable histogram calibration (SAI only)')
    parser.add_argument('--scene_finetune',      action='store_true',
                        help='Enable LoRA scene fine-tuning (SAI only, slower)')
    parser.add_argument('--finetune_steps',      type=int,   default=100)
    parser.add_argument('--finetune_lr',         type=float, default=1e-4)
    parser.add_argument('--lora_r',              type=int,   default=4)
    parser.add_argument('--scene_id',            default='scene',
                        help='Scene identifier used in output subdirectory names')
    args = parser.parse_args()

    run_inference(
        checkpoint_path=args.checkpoint,
        mwir_path=args.mwir,
        output_path=args.output,
        lwir_path=args.lwir,
        patch_size=args.patch_size,
        overlap=args.overlap,
        num_steps=args.num_steps,
        eta=args.eta,
        n_ensemble=args.ensemble,
        device=args.device,
        mwir_swath_km=args.mwir_swath_km,
        lwir_swath_km=args.lwir_swath_km,
        use_histogram_cal=not args.no_histogram_cal,
        use_scene_finetuning=args.scene_finetune,
        finetune_steps=args.finetune_steps,
        finetune_lr=args.finetune_lr,
        lora_r=args.lora_r,
        scene_id=args.scene_id,
    )
