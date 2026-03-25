"""
Diffusion scheduler combining:
  1. DDPM (Ho et al. 2020)           — stochastic training
  2. DDIM (Song et al. 2020)         — deterministic fast sampling
  3. Characteristic Function Consistency (CFC) loss
       — penalizes mismatch in empirical characteristic functions between
         generated and real LWIR patches, addressing the heterogeneous/low-contrast
         problem in thermal IR imagery where pixel-space losses are insufficient.

Reference physics motivation:
  LWIR (8–12 µm) radiance ∝ emissivity × Planck function.
  MWIR (3–5 µm) has stronger solar-reflected component → different spectral
  statistics and spatial heterogeneity. CFC loss aligns distributions in
  frequency domain rather than pixel domain.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# ─────────────────────────────────────────────
# Noise Schedule
# ─────────────────────────────────────────────

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine beta schedule (Nichol & Dhariwal 2021).
    Better than linear for sensor imagery — avoids over-noising low-contrast IR.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def sqrt_beta_schedule(timesteps: int, beta_start=1e-4, beta_end=0.02):
    """Square-root schedule — emphasises mid-noise levels (good for heterogeneous data)."""
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


# ─────────────────────────────────────────────
# Characteristic Function Consistency (CFC) Loss
# ─────────────────────────────────────────────

class CharacteristicFunctionConsistencyLoss(nn.Module):
    """
    Computes the L2 distance between empirical characteristic functions (ECF)
    of generated and real LWIR patches.

    The ECF of a random variable X is:  φ(ω) = E[exp(i·ω·X)]
    Matching ECFs is equivalent to matching all moments of the distribution,
    making it robust to the heavy-tailed, heterogeneous statistics of thermal IR.

    Args:
        num_freqs:   number of frequency test points
        max_freq:    maximum frequency (tune to image dynamic range)
        patch_size:  spatial patch for local distribution matching
    """

    def __init__(self, num_freqs: int = 32, max_freq: float = 1.0, patch_size: int = 16,
                 max_patches: int = 64):
        super().__init__()
        self.patch_size = patch_size
        self.max_patches = max_patches
        # Fixed test frequencies ω ∈ [-max_freq, max_freq]
        freqs = torch.linspace(-max_freq, max_freq, num_freqs)
        self.register_buffer('freqs', freqs)

    def empirical_cf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute empirical CF: φ̂(ω) = mean_n[ exp(i·ω·x_n) ]
        Returns real and imaginary parts concatenated.
        x: (B, N) — B batches, N samples per batch
        """
        # x: (B, N), freqs: (F,)
        phase = torch.einsum('bn,f->bnf', x, self.freqs)  # (B, N, F)
        cos_part = phase.cos().mean(dim=1)                  # (B, F)
        sin_part = phase.sin().mean(dim=1)                  # (B, F)
        return torch.cat([cos_part, sin_part], dim=-1)      # (B, 2F)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: (B, C, H, W)
        Returns scalar CFC loss.
        """
        B, C, H, W = pred.shape
        p = self.patch_size

        # Extract non-overlapping patches and flatten
        pred_patches = pred.unfold(2, p, p).unfold(3, p, p)   # (B, C, nH, nW, p, p)
        tgt_patches = target.unfold(2, p, p).unfold(3, p, p)

        nH, nW = pred_patches.shape[2], pred_patches.shape[3]
        pred_flat = pred_patches.reshape(B * nH * nW, -1)      # (B*nH*nW, C*p*p)
        tgt_flat = tgt_patches.reshape(B * nH * nW, -1)

        # Subsample patches to bound compute — random sample changes each step
        total_patches = pred_flat.shape[0]
        if total_patches > self.max_patches:
            idx = torch.randperm(total_patches, device=pred_flat.device)[:self.max_patches]
            pred_flat = pred_flat[idx]
            tgt_flat = tgt_flat[idx]

        # Compute ECF for each patch
        pred_ecf = self.empirical_cf(pred_flat)
        tgt_ecf = self.empirical_cf(tgt_flat)

        return F.mse_loss(pred_ecf, tgt_ecf)


# ─────────────────────────────────────────────
# Spectral Consistency Loss
# ─────────────────────────────────────────────

class SpectralConsistencyLoss(nn.Module):
    """
    Matches the 2D power spectral density of generated vs real LWIR.
    Addresses spatial frequency artifacts common in IR synthesis
    (blurring / ringing at edges of man-made structures).
    """
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.fft2(pred, norm='ortho')
        tgt_fft = torch.fft.fft2(target, norm='ortho')
        pred_psd = pred_fft.abs() ** 2
        tgt_psd = tgt_fft.abs() ** 2
        return F.l1_loss(torch.log1p(pred_psd), torch.log1p(tgt_psd))


# ─────────────────────────────────────────────
# Composite Diffusion Loss
# ─────────────────────────────────────────────

class DiffusionLoss(nn.Module):
    """
    Combines:
      - Standard ε-prediction MSE loss (DDPM objective)
      - Characteristic Function Consistency loss (CFC)
      - Spectral Consistency loss
      - Optional perceptual loss (if feature extractor provided)
    """
    def __init__(
        self,
        lambda_cfc: float = 0.1,
        lambda_spectral: float = 0.05,
        cfc_num_freqs: int = 32,
        cfc_patch_size: int = 16,
    ):
        super().__init__()
        self.lambda_cfc = lambda_cfc
        self.lambda_spectral = lambda_spectral
        self.cfc_loss = CharacteristicFunctionConsistencyLoss(
            num_freqs=cfc_num_freqs, patch_size=cfc_patch_size
        )
        self.spectral_loss = SpectralConsistencyLoss()

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        x0_pred: Optional[torch.Tensor] = None,
        x0_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        # Primary DDPM ε-prediction loss
        mse = F.mse_loss(noise_pred, noise_target)
        losses = {'mse': mse.item()}
        total = mse

        # CFC and spectral losses on x0 estimates (more meaningful in data space)
        if x0_pred is not None and x0_target is not None:
            cfc = self.cfc_loss(x0_pred, x0_target)
            spec = self.spectral_loss(x0_pred, x0_target)
            losses['cfc'] = cfc.item()
            losses['spectral'] = spec.item()
            total = total + self.lambda_cfc * cfc + self.lambda_spectral * spec

        losses['total'] = total.item()
        return total, losses


# ─────────────────────────────────────────────
# DDPM / DDIM Scheduler
# ─────────────────────────────────────────────

class DDIMScheduler(nn.Module):
    """
    Unified DDPM training + DDIM sampling scheduler.

    Key improvements for MWIR→LWIR:
      - Cosine noise schedule (better for low-contrast thermal)
      - x0-clipping with learned dynamic range
      - DDIM deterministic sampling (η=0) or stochastic (η>0)
      - Supports classifier-free guidance at inference
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        schedule: str = 'cosine',   # 'cosine' | 'linear' | 'sqrt'
        clip_sample: bool = True,
        clip_range: float = 1.0,
        prediction_type: str = 'epsilon',  # 'epsilon' | 'x0' | 'v'
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.clip_range = clip_range
        self.prediction_type = prediction_type

        # Build schedule
        if schedule == 'cosine':
            betas = cosine_beta_schedule(num_train_timesteps)
        elif schedule == 'sqrt':
            betas = sqrt_beta_schedule(num_train_timesteps)
        else:
            betas = linear_beta_schedule(num_train_timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', alphas_cumprod.sqrt())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', (1 - alphas_cumprod).sqrt())
        self.register_buffer('sqrt_recip_alphas', (1.0 / alphas).sqrt())
        self.register_buffer('posterior_variance',
            betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1 - alphas_cumprod))

    # ── Training: add noise ──

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: q(xₜ | x₀) = N(√ᾱₜ x₀, (1-ᾱₜ)I)"""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_a = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_1ma = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_a * x0 + sqrt_1ma * noise, noise

    def training_losses(
        self,
        model,
        x0: torch.Tensor,
        mwir: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """Sample a random t, add noise, predict, compute loss components."""
        if t is None:
            t = torch.randint(0, self.num_train_timesteps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        xt, _ = self.q_sample(x0, t, noise)
        model_out = model(xt, t, mwir)

        if self.prediction_type == 'epsilon':
            noise_pred = model_out
            x0_pred = self._predict_x0_from_eps(xt, t, noise_pred)
        elif self.prediction_type == 'x0':
            x0_pred = model_out
            noise_pred = self._predict_eps_from_x0(xt, t, x0_pred)
        else:  # 'v'
            noise_pred = self._v_to_eps(xt, t, model_out)
            x0_pred = self._predict_x0_from_eps(xt, t, noise_pred)

        # Hard clamp x0_pred regardless of clip_sample config.
        # At high timesteps (t → T), sqrt(ᾱ_t) → 0 and x0_pred = (x_t - sqrt(1-ᾱ)·ε̂)/sqrt(ᾱ)
        # diverges to O(100–600) for a unit-norm prediction. Passing these extreme
        # values to CFC loss (cos/sin of large phases) and spectral loss produces
        # chaotic gradients that trigger GradScaler inf-detection even in bfloat16.
        # Clamping here only affects the auxiliary losses — the primary MSE loss
        # on noise_pred is unaffected and is still trained correctly at high t.
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        return noise_pred, noise, x0_pred, x0, t

    # ── Inference: DDIM sampling ──

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        mwir: torch.Tensor,
        shape: Tuple,
        num_inference_steps: int = 50,
        eta: float = 0.0,            # 0=deterministic DDIM, 1=DDPM
        guidance_scale: float = 1.0,
        device: str = 'cuda',
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        DDIM sampling loop.
        eta=0: fully deterministic (faster, good for qualitative)
        eta=1: stochastic (matches DDPM, sometimes better diversity)
        """
        B = mwir.shape[0]
        x = torch.randn(shape, device=device)

        # Build timestep sequence
        timesteps = torch.linspace(
            self.num_train_timesteps - 1, 0,
            num_inference_steps, dtype=torch.long, device=device
        )

        for i, t_cur in enumerate(timesteps):
            t_batch = t_cur.expand(B)

            # Classifier-free guidance (if scale > 1, pass both cond and uncond)
            if guidance_scale > 1.0:
                uncond_mwir = torch.zeros_like(mwir)
                x_in = torch.cat([x, x])
                t_in = torch.cat([t_batch, t_batch])
                mwir_in = torch.cat([mwir, uncond_mwir])
                out = model(x_in, t_in, mwir_in)
                cond_out, uncond_out = out.chunk(2)
                model_out = uncond_out + guidance_scale * (cond_out - uncond_out)
            else:
                model_out = model(x, t_batch, mwir)

            x = self._ddim_step(x, t_cur, model_out, timesteps, i, eta)

            if verbose and i % 10 == 0:
                print(f"  DDIM step {i+1}/{num_inference_steps}")

        return x.clamp(-1, 1)

    def _ddim_step(self, x, t, model_out, timesteps, step_idx, eta):
        t_prev = timesteps[step_idx + 1] if step_idx + 1 < len(timesteps) else torch.tensor(-1)

        a_t = self.alphas_cumprod[t]
        a_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x.device)

        if self.prediction_type == 'epsilon':
            x0_pred = (x - (1 - a_t).sqrt() * model_out) / a_t.sqrt()
        elif self.prediction_type == 'x0':
            x0_pred = model_out
        else:
            x0_pred = a_t.sqrt() * x - (1 - a_t).sqrt() * model_out

        if self.clip_sample:
            x0_pred = x0_pred.clamp(-self.clip_range, self.clip_range)

        eps = (x - a_t.sqrt() * x0_pred) / (1 - a_t).sqrt()

        # DDIM variance
        sigma = eta * ((1 - a_prev) / (1 - a_t)).sqrt() * (1 - a_t / a_prev).sqrt()
        noise = torch.randn_like(x) if eta > 0 else 0.0

        x_prev = a_prev.sqrt() * x0_pred + (1 - a_prev - sigma**2).sqrt() * eps + sigma * noise
        return x_prev

    # ── Helpers ──

    @staticmethod
    def _extract(arr, t, shape):
        out = arr[t]
        while out.ndim < len(shape):
            out = out.unsqueeze(-1)
        return out.expand(shape)

    def _predict_x0_from_eps(self, xt, t, eps):
        sqrt_a = self._extract(self.sqrt_alphas_cumprod, t, xt.shape)
        sqrt_1ma = self._extract(self.sqrt_one_minus_alphas_cumprod, t, xt.shape)
        return (xt - sqrt_1ma * eps) / sqrt_a

    def _predict_eps_from_x0(self, xt, t, x0):
        sqrt_a = self._extract(self.sqrt_alphas_cumprod, t, xt.shape)
        sqrt_1ma = self._extract(self.sqrt_one_minus_alphas_cumprod, t, xt.shape)
        return (xt - sqrt_a * x0) / sqrt_1ma

    def _v_to_eps(self, xt, t, v):
        sqrt_a = self._extract(self.sqrt_alphas_cumprod, t, xt.shape)
        sqrt_1ma = self._extract(self.sqrt_one_minus_alphas_cumprod, t, xt.shape)
        return sqrt_a * v + sqrt_1ma * xt
