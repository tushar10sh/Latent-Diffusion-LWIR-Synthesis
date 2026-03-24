"""
LDM Inference: MWIR → Synthetic LWIR via latent diffusion.

Features:
  - Classifier-Free Guidance (CFG) — critical for quality at this data scale
  - DDIM fast sampling (50 steps)
  - Patch-based inference for large images (operates in latent space)
  - Uncertainty estimation via stochastic ensemble
  - Full metric evaluation
"""

import math
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple

from models.ldm.vae import IRVAE
from models.ldm.dit import DiT_S_4, DiT_B_4, DiT_L_4
from models.diffusion_scheduler import DDIMScheduler
from data.dataset import percentile_normalize
from inference.infer import compute_metrics


class LDMInference:
    """
    Full LDM inference pipeline.

    Args:
        vae_ckpt:        Path to Stage 1 VAE checkpoint
        dit_ckpt:        Path to Stage 2 DiT checkpoint
        guidance_scale:  CFG scale (1.0 = no guidance, 3.0-7.0 recommended)
        num_steps:       DDIM sampling steps (50 is good, 20 acceptable)
        eta:             DDIM stochasticity (0=deterministic)
        use_ema:         Whether to use EMA weights (always True for inference)
    """

    def __init__(
        self,
        vae_ckpt: str,
        dit_ckpt: str,
        guidance_scale: float = 5.0,
        num_steps: int = 50,
        eta: float = 0.0,
        use_ema: bool = True,
        device: str = 'cuda',
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.guidance_scale = guidance_scale
        self.num_steps = num_steps
        self.eta = eta

        # Load VAE
        vae_state = torch.load(vae_ckpt, map_location=self.device)
        cfg = vae_state.get('config', {})
        self.vae = IRVAE(
            in_channels=cfg.get('lwir_channels', 1),
            ch=cfg.get('vae_ch', 128),
            ch_mult=tuple(cfg.get('vae_ch_mult', [1, 2, 4])),
            z_channels=cfg.get('z_channels', 4),
        ).to(self.device)
        self.vae.load_state_dict(vae_state['vae'])
        self.vae.scale_factor = vae_state['scale_factor']
        self.vae.eval()
        print(f"[LDM] VAE loaded. scale_factor={self.vae.scale_factor:.4f}")

        # Load DiT
        dit_state = torch.load(dit_ckpt, map_location=self.device)
        dit_cfg = dit_state.get('config', {})
        dit_size = dit_cfg.get('dit_size', 'B')
        builder = {'S': DiT_S_4, 'B': DiT_B_4, 'L': DiT_L_4}[dit_size]
        self.dit = builder(
            in_channels=dit_cfg.get('z_channels', 4),
            mwir_channels=dit_cfg.get('mwir_channels', 1),
            context_dim=dit_cfg.get('dit_context_dim', 512),
            cond_dim=dit_cfg.get('dit_cond_dim', 1024),
            vae_f=dit_cfg.get('vae_f', 4),
        ).to(self.device)

        if use_ema and 'ema' in dit_state:
            weights = dit_state['ema']['shadow']
            self.dit.load_state_dict({k: v.to(next(self.dit.parameters()).dtype) for k, v in weights.items()})
            print("[LDM] DiT loaded with EMA weights.")
        else:
            self.dit.load_state_dict(dit_state['dit'])
            print("[LDM] DiT loaded (no EMA).")
        self.dit.eval()

        # Diffusion scheduler
        self.scheduler = DDIMScheduler(
            num_train_timesteps=dit_cfg.get('num_train_timesteps', 1000),
            schedule=dit_cfg.get('noise_schedule', 'cosine'),
            clip_sample=False,
            prediction_type='epsilon',
        ).to(self.device)

        self.image_size = dit_cfg.get('image_size', 256)
        self.vae_f = dit_cfg.get('vae_f', 4)
        self.z_channels = dit_cfg.get('z_channels', 4)

    @torch.no_grad()
    def sample(
        self,
        mwir: torch.Tensor,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Generate LWIR from a single MWIR tile (same size as training resolution).
        mwir: (B, C, H, W) normalised to [-1, 1]
        Returns: (B, 1, H, W) generated LWIR
        """
        B, _, H, W = mwir.shape
        H_lat, W_lat = H // self.vae_f, W // self.vae_f

        z = torch.randn(B, self.z_channels, H_lat, W_lat, device=self.device)

        timesteps = torch.linspace(
            self.scheduler.num_train_timesteps - 1, 0,
            self.num_steps, dtype=torch.long, device=self.device
        )

        null_mwir = torch.zeros_like(mwir)  # null conditioning for CFG

        for i, t_cur in enumerate(timesteps):
            t_b = t_cur.expand(B)

            if self.guidance_scale > 1.0:
                # Classifier-Free Guidance
                cond_eps = self.dit(z, t_b, mwir)
                uncond_eps = self.dit(z, t_b, null_mwir)
                eps = uncond_eps + self.guidance_scale * (cond_eps - uncond_eps)
            else:
                eps = self.dit(z, t_b, mwir)

            z = self.scheduler._ddim_step(z, t_cur, eps, timesteps, i, self.eta)

            if verbose and (i + 1) % 10 == 0:
                print(f"  DDIM {i+1}/{self.num_steps}", end='\r')

        if verbose:
            print()

        # Decode latent → pixel space via VAE decoder
        lwir_gen = self.vae.decode(z)
        return lwir_gen.clamp(-1, 1)

    @torch.no_grad()
    def sample_large(
        self,
        mwir: torch.Tensor,
        patch_size: int = 256,
        overlap: float = 0.25,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Sliding-window inference for images larger than training resolution.
        Blending is done in LATENT SPACE before decoding — avoids seams.
        """
        _, _, H, W = mwir.shape
        stride = int(patch_size * (1 - overlap))
        H_lat = H // self.vae_f
        W_lat = W // self.vae_f
        p_lat = patch_size // self.vae_f

        # Build Hann blending mask in latent space
        hann = torch.hann_window(p_lat, device=self.device)
        mask = (hann.unsqueeze(0) * hann.unsqueeze(1)).unsqueeze(0).unsqueeze(0)

        out_z = torch.zeros(1, self.z_channels, H_lat, W_lat, device=self.device)
        weight = torch.zeros(1, 1, H_lat, W_lat, device=self.device)

        s_lat = stride // self.vae_f
        ys = list(range(0, H_lat - p_lat + 1, s_lat)) + ([H_lat - p_lat] if H_lat > p_lat else [])
        xs = list(range(0, W_lat - p_lat + 1, s_lat)) + ([W_lat - p_lat] if W_lat > p_lat else [])
        ys = sorted(set(ys))
        xs = sorted(set(xs))

        total = len(ys) * len(xs)
        done = 0

        for y in ys:
            for x in xs:
                # MWIR patch (pixel space)
                y_px, x_px = y * self.vae_f, x * self.vae_f
                mwir_patch = mwir[:, :, y_px:y_px+patch_size, x_px:x_px+patch_size]

                # Sample latent patch
                z_patch = self._sample_latent(mwir_patch)  # (1, C, p_lat, p_lat)

                out_z[:, :, y:y+p_lat, x:x+p_lat] += z_patch * mask
                weight[:, :, y:y+p_lat, x:x+p_lat] += mask
                done += 1
                if verbose:
                    print(f"  Patch {done}/{total}", end='\r')

        if verbose:
            print()

        # Blend and decode once
        z_blended = out_z / weight.clamp(min=1e-6)
        return self.vae.decode(z_blended).clamp(-1, 1)

    def _sample_latent(self, mwir_patch: torch.Tensor) -> torch.Tensor:
        """Sample a latent z from a MWIR patch without decoding."""
        B, _, H, W = mwir_patch.shape
        H_lat, W_lat = H // self.vae_f, W // self.vae_f

        z = torch.randn(B, self.z_channels, H_lat, W_lat, device=self.device)
        null_mwir = torch.zeros_like(mwir_patch)

        timesteps = torch.linspace(
            self.scheduler.num_train_timesteps - 1, 0,
            self.num_steps, dtype=torch.long, device=self.device
        )
        for i, t_cur in enumerate(timesteps):
            t_b = t_cur.expand(B)
            cond_eps = self.dit(z, t_b, mwir_patch)
            uncond_eps = self.dit(z, t_b, null_mwir)
            eps = uncond_eps + self.guidance_scale * (cond_eps - uncond_eps)
            z = self.scheduler._ddim_step(z, t_cur, eps, timesteps, i, self.eta)
        return z


def run_ldm_inference(
    vae_ckpt: str,
    dit_ckpt: str,
    mwir_path: str,
    output_path: str,
    lwir_path: Optional[str] = None,       # real LWIR overlap strip for SAI, or GT for metrics
    image_size: int = 256,
    guidance_scale: float = 5.0,
    num_steps: int = 50,
    device: str = 'cuda',
    # ── Scene-adaptive options ──────────────────────────────────
    mwir_swath_km: float = 0.0,
    lwir_swath_km: float = 0.0,
    use_histogram_cal: bool = True,
    use_scene_finetuning: bool = False,
    finetune_steps: int = 100,
    finetune_lr: float = 1e-4,
    lora_r: int = 4,
    scene_id: str = 'scene',
):
    pipeline = LDMInference(vae_ckpt, dit_ckpt, guidance_scale, num_steps, device=device)

    # Load raw arrays
    mwir_arr = np.load(mwir_path).astype(np.float32)
    if mwir_arr.ndim == 2:
        mwir_arr = mwir_arr[np.newaxis]

    lwir_arr = None
    if lwir_path:
        lwir_arr = np.load(lwir_path).astype(np.float32)
        if lwir_arr.ndim == 2:
            lwir_arr = lwir_arr[np.newaxis]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    use_sai = (mwir_swath_km > 0 and lwir_swath_km > 0 and lwir_arr is not None)

    if use_sai:
        # ── Scene-Adaptive Inference path ────────────────────────
        from inference.scene_adaptive import SceneAdaptiveInference

        def generate_fn(mwir_t: torch.Tensor) -> torch.Tensor:
            """Wraps LDMInference.sample / sample_large as a simple callable."""
            if mwir_t.shape[-2] <= image_size and mwir_t.shape[-1] <= image_size:
                mwir_resized = F.interpolate(
                    mwir_t, size=(image_size, image_size),
                    mode='bilinear', align_corners=False,
                )
                return pipeline.sample(mwir_resized, verbose=False)
            else:
                return pipeline.sample_large(mwir_t, patch_size=image_size, verbose=False)

        sai = SceneAdaptiveInference(
            model=pipeline.dit,            # SAI's LoRA injection targets the DiT
            scheduler=pipeline.scheduler,
            generate_fn=generate_fn,
            mwir_swath_km=mwir_swath_km,
            lwir_swath_km=lwir_swath_km,
            use_histogram_cal=use_histogram_cal,
            use_scene_finetuning=use_scene_finetuning,
            finetune_steps=finetune_steps,
            finetune_lr=finetune_lr,
            lora_r=lora_r,
            device=device,
            output_dir=str(out.parent),
        )
        result = sai.run(
            mwir_raw=mwir_arr,
            lwir_raw=lwir_arr,
            scene_id=scene_id,
        )
        np.save(out.with_suffix('.npy'), result['lwir_full'].astype(np.float32))
        print(f"[LDM+SAI] Full swath saved → {out.with_suffix('.npy')}")
        return

    # ── Standard inference path (unchanged) ──────────────────────
    mwir_norm = np.stack(
        [percentile_normalize(mwir_arr[c]) for c in range(mwir_arr.shape[0])]
    )
    mwir_t = torch.from_numpy(mwir_norm).unsqueeze(0).to(pipeline.device)
    print(f"[Inference] MWIR: {mwir_t.shape}, guidance={guidance_scale}, steps={num_steps}")

    if mwir_t.shape[-2] <= image_size and mwir_t.shape[-1] <= image_size:
        mwir_t = F.interpolate(
            mwir_t, size=(image_size, image_size), mode='bilinear', align_corners=False
        )
        result_t = pipeline.sample(mwir_t)
    else:
        result_t = pipeline.sample_large(mwir_t, patch_size=image_size)

    np.save(out.with_suffix('.npy'), result_t.squeeze().cpu().numpy())
    print(f"[Inference] Saved → {out.with_suffix('.npy')}")

    if lwir_arr is not None:
        lwir_norm = np.stack(
            [percentile_normalize(lwir_arr[c]) for c in range(lwir_arr.shape[0])]
        )
        lwir_t = torch.from_numpy(lwir_norm).unsqueeze(0).to(pipeline.device)
        if lwir_t.shape[-2:] != result_t.shape[-2:]:
            lwir_t = F.interpolate(
                lwir_t, size=result_t.shape[-2:], mode='bilinear', align_corners=False
            )
        metrics = compute_metrics(result_t, lwir_t)
        print("\n[Metrics]")
        for k, v in metrics.items():
            print(f"  {k:>14s}: {v}")
        with open(out.parent / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LDM MWIR→LWIR Inference')
    parser.add_argument('--vae_ckpt',          required=True)
    parser.add_argument('--dit_ckpt',          required=True)
    parser.add_argument('--mwir',              required=True)
    parser.add_argument('--output',            required=True)
    parser.add_argument('--lwir',              default=None,
                        help='Real LWIR .npy — overlap strip for SAI, or full-swath GT for metrics')
    parser.add_argument('--image_size',        type=int,   default=256)
    parser.add_argument('--guidance_scale',    type=float, default=5.0)
    parser.add_argument('--num_steps',         type=int,   default=50)
    parser.add_argument('--device',            default='cuda')
    # ── Scene-adaptive options ──────────────────────────────────
    parser.add_argument('--mwir_swath_km',     type=float, default=0.0,
                        help='MWIR swath width in km. Set >0 with --lwir to enable SAI.')
    parser.add_argument('--lwir_swath_km',     type=float, default=0.0,
                        help='LWIR swath width in km.')
    parser.add_argument('--no_histogram_cal',  action='store_true')
    parser.add_argument('--scene_finetune',    action='store_true')
    parser.add_argument('--finetune_steps',    type=int,   default=100)
    parser.add_argument('--finetune_lr',       type=float, default=1e-4)
    parser.add_argument('--lora_r',            type=int,   default=4)
    parser.add_argument('--scene_id',          default='scene')
    args = parser.parse_args()

    run_ldm_inference(
        vae_ckpt=args.vae_ckpt,
        dit_ckpt=args.dit_ckpt,
        mwir_path=args.mwir,
        output_path=args.output,
        lwir_path=args.lwir,
        image_size=args.image_size,
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps,
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
