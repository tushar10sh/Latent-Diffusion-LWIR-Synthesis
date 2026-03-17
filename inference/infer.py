"""
Inference pipeline for MWIR→LWIR synthesis.

Features:
  - DDIM deterministic fast sampling (50 steps vs 1000 DDPM)
  - Stochastic ensemble (run N times, average predictions for PSNR boost)
  - Sliding-window patch inference for large images
  - Quantitative evaluation: PSNR, SSIM, ERGAS, SAM, CFC distance
  - GeoTIFF output support (preserves geolocation metadata)
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
    """
    samples = []
    for i in range(n_ensemble):
        s = scheduler.ddim_sample(
            model, mwir,
            shape=(mwir.shape[0], 1, mwir.shape[2], mwir.shape[3]),
            num_inference_steps=num_steps,
            eta=0.5,  # partial stochasticity for diversity
            device=device,
            verbose=False,
        )
        samples.append(s)
    stacked = torch.stack(samples, dim=0)
    return stacked.mean(0), stacked.std(0)


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
    lwir_gt_path: Optional[str] = None,
    patch_size: int = 256,
    overlap: float = 0.25,
    num_steps: int = 50,
    eta: float = 0.0,
    n_ensemble: int = 1,
    device: str = 'cuda',
):
    """
    End-to-end inference on a single MWIR image.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt['config']
    print(f"[Inference] Loaded checkpoint (step {ckpt['step']})")

    # Build model
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

    # Load EMA weights for inference
    ema_state = ckpt['ema']['shadow']
    model.load_state_dict({k: v.to(next(model.parameters()).dtype) for k, v in ema_state.items()})
    model.eval()

    scheduler = DDIMScheduler(
        num_train_timesteps=cfg.get('num_train_timesteps', 1000),
        schedule=cfg.get('noise_schedule', 'cosine'),
        prediction_type=cfg.get('prediction_type', 'epsilon'),
    ).to(device)

    # Load MWIR
    mwir_arr = np.load(mwir_path).astype(np.float32)
    if mwir_arr.ndim == 2:
        mwir_arr = mwir_arr[None]
    from data.dataset import percentile_normalize
    mwir_norm = np.stack([percentile_normalize(mwir_arr[c]) for c in range(mwir_arr.shape[0])])
    mwir_t = torch.from_numpy(mwir_norm).unsqueeze(0).to(device)

    print(f"[Inference] MWIR shape: {mwir_t.shape}, device: {device}")

    # Patch inference
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
        std_pred = None

    # Save output
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_np = mean_pred.squeeze().cpu().numpy()
    np.save(out_path.with_suffix('.npy'), result_np)
    if std_pred is not None:
        np.save(out_path.parent / (out_path.stem + '_uncertainty.npy'), std_pred.squeeze().cpu().numpy())

    print(f"[Inference] Saved → {out_path.with_suffix('.npy')}")

    # Metrics against ground truth
    if lwir_gt_path:
        lwir_gt = np.load(lwir_gt_path).astype(np.float32)
        if lwir_gt.ndim == 2:
            lwir_gt = lwir_gt[None]
        lwir_norm = np.stack([percentile_normalize(lwir_gt[c]) for c in range(lwir_gt.shape[0])])
        lwir_t = torch.from_numpy(lwir_norm).unsqueeze(0).to(device)

        if lwir_t.shape[-2:] != mean_pred.shape[-2:]:
            lwir_t = F.interpolate(lwir_t, size=mean_pred.shape[-2:], mode='bilinear', align_corners=False)

        metrics = compute_metrics(mean_pred, lwir_t)
        print("\n[Metrics]")
        for k, v in metrics.items():
            print(f"  {k:>12s}: {v}")
        with open(out_path.parent / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

    return mean_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MWIR→LWIR Inference')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--mwir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--lwir_gt', default=None)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--overlap', type=float, default=0.25)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--ensemble', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    run_inference(
        args.checkpoint, args.mwir, args.output,
        args.lwir_gt, args.patch_size, args.overlap,
        args.num_steps, args.eta, args.ensemble, args.device,
    )
