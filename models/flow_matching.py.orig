"""
Flow Matching for MWIR→LWIR synthesis.

Replaces DDPM/DDIM with Rectified Flow (Liu et al. 2022) as implemented
in Stable Diffusion 3 and Flux. Key advantages over DDPM for this task:

  1. Straight trajectories: x_t = (1-t)·x₀ + t·ε
     The model learns a constant velocity field u = ε - x₀.
     Straighter paths → fewer ODE steps at inference (8–20 vs 50).

  2. Uniform time training: t ~ Uniform[0,1] instead of a discrete schedule.
     No noise schedule to tune. The cosine schedule becomes irrelevant.

  3. Better conditioning: the velocity target is fixed (ε - x₀),
     not timestep-dependent. The model can focus on learning the
     MWIR→LWIR cross-modal relationship rather than denoising dynamics.

  4. Logit-Normal time sampling (optionally): concentrates training steps
     near t=0.5 where the trajectory curvature is highest, improving
     sample quality at a given NFE budget. Used in SD3.

  5. CFG is identical — drop conditioning for uncond, same formula.

API is designed to be a drop-in replacement for DDIMScheduler in all
three trainers (Trainer, ImprovedTrainer, DiTTrainer). The model
architecture (UNet, DiT) is UNCHANGED — only the training objective
and sampler change.

Timestep convention:
  t ∈ [0, 1]   where t=0 is clean data, t=1 is pure noise.
  Models still receive t scaled to [0, 1000] for embedding compatibility.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable


# ─────────────────────────────────────────────
# Time sampling distributions
# ─────────────────────────────────────────────

def sample_t_uniform(B: int, device: torch.device) -> torch.Tensor:
    """Uniform t ~ U[0, 1]. Standard rectified flow."""
    return torch.rand(B, device=device)


def sample_t_logit_normal(B: int, device: torch.device,
                           loc: float = 0.0, scale: float = 1.0) -> torch.Tensor:
    """
    Logit-Normal time sampling (Esser et al. 2024, SD3).
    Concentrates samples near t=0.5 where the trajectory is hardest.
    Empirically improves sample quality for the same training budget.

    loc=0.0, scale=1.0 → symmetric bell centred at t=0.5
    loc>0              → shift toward t=1 (more noise-heavy training)
    """
    u = torch.randn(B, device=device) * scale + loc
    return torch.sigmoid(u)


def sample_t_cosmap(B: int, device: torch.device) -> torch.Tensor:
    """
    CosMap sampling (Gao et al. 2024) — maps uniform samples through
    an arccosine warp that up-weights the mid-trajectory region.
    Alternative to logit-normal with similar properties.
    """
    u = torch.rand(B, device=device)
    return 1 - (1 / (math.pi / 2 * (torch.tan(math.pi / 2 * u) + 1)))


# ─────────────────────────────────────────────
# Flow Matching Scheduler
# ─────────────────────────────────────────────

class FlowMatchingScheduler(nn.Module):
    """
    Rectified Flow / Flow Matching scheduler.

    Training objective (velocity prediction):
        x_t = (1 - t) · x₀ + t · ε,   ε ~ N(0, I),   t ~ p(t)
        u_target = ε - x₀               (constant velocity along path)
        L = E_t E_ε [ w(t) · ||model(x_t, t, cond) - u_target||² ]

    Inference (ODE integration from t=1 to t=0):
        dx/dt = -u_θ(x_t, t, cond)
        Euler: x_{t-Δt} = x_t - u_θ · Δt
        Heun:  2nd-order correction (better quality, same NFE)

    Args:
        t_scale:       scale t ∈ [0,1] to [0, t_scale] for the timestep
                       embedding. Use 1000 to match existing embeddings
                       trained with T=1000 steps.
        time_sampling: 'uniform' | 'logit_normal' | 'cosmap'
        logit_loc:     mean of logit-normal distribution (default 0.0)
        logit_scale:   std of logit-normal distribution (default 1.0)
        loss_weighting:'constant' | 'snr' (SNR-weighted, as in DDPM Min-SNR)
    """

    def __init__(
        self,
        t_scale:       int   = 1000,
        time_sampling: str   = 'logit_normal',
        logit_loc:     float = 0.0,
        logit_scale:   float = 1.0,
        loss_weighting: str  = 'constant',
    ):
        super().__init__()
        self.t_scale       = t_scale
        self.time_sampling = time_sampling
        self.logit_loc     = logit_loc
        self.logit_scale   = logit_scale
        self.loss_weighting = loss_weighting

    # ── Helpers ────────────────────────────────────────────────────

    def _sample_t(self, B: int, device: torch.device) -> torch.Tensor:
        """Sample continuous t ∈ [0, 1] according to the chosen distribution."""
        if self.time_sampling == 'logit_normal':
            return sample_t_logit_normal(B, device, self.logit_loc, self.logit_scale)
        elif self.time_sampling == 'cosmap':
            return sample_t_cosmap(B, device)
        else:
            return sample_t_uniform(B, device)

    def _t_to_embed(self, t_cont: torch.Tensor) -> torch.Tensor:
        """
        Scale continuous t ∈ [0, 1] to integer-range for timestep embedding.
        The UNet/DiT timestep embedding was designed for t ∈ [0, T] where T=1000.
        We multiply by t_scale and round to the nearest integer.
        """
        return (t_cont * self.t_scale).long().clamp(0, self.t_scale - 1)

    def _loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """
        Optional per-timestep loss weighting.
        'constant' = 1.0 (standard rectified flow)
        'snr'      = min(SNR(t), 5) / SNR(t)  (analogous to Min-SNR for flow)
        """
        if self.loss_weighting == 'constant':
            return torch.ones_like(t)
        # Flow SNR: at time t, signal = (1-t)·x₀, noise = t·ε
        # SNR(t) = (1-t)² / t²
        snr = ((1 - t) / t.clamp(min=1e-5)) ** 2
        return (snr.clamp(max=5.0) / snr.clamp(min=1e-5)).detach()

    # ── Forward process ────────────────────────────────────────────

    def q_sample(
        self,
        x0:    torch.Tensor,
        t:     torch.Tensor,   # continuous t ∈ [0, 1], shape (B,)
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Linear interpolation: x_t = (1-t)·x₀ + t·ε
        Returns (x_t, noise).
        """
        if noise is None:
            noise = torch.randn_like(x0)
        t_b = t.view(-1, 1, 1, 1)          # broadcast over spatial dims
        x_t = (1 - t_b) * x0 + t_b * noise
        return x_t, noise

    def get_velocity_target(
        self,
        x0:    torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Velocity target: u = ε - x₀.
        This is the time derivative of x_t along the straight path.
        """
        return noise - x0

    # ── Training ───────────────────────────────────────────────────

    def training_losses(
        self,
        model:  Callable,
        x0:     torch.Tensor,
        mwir:   torch.Tensor,
        t_cont: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One training step for flow matching.

        Returns: (u_pred, u_target, x0_pred, x0, t_embed)
        — same return signature as DDIMScheduler.training_losses for compatibility.

        x0_pred is estimated from the velocity prediction:
            x₀_pred = x_t - t · u_pred   (rearranging x_t = (1-t)x₀ + t·ε)
        """
        B = x0.shape[0]
        device = x0.device

        if t_cont is None:
            t_cont = self._sample_t(B, device)          # (B,) in [0, 1]

        noise = torch.randn_like(x0)
        x_t, _ = self.q_sample(x0, t_cont, noise)
        u_target = self.get_velocity_target(x0, noise)  # ε - x₀

        # Scale t for embedding (UNet/DiT expect long integers in [0, T])
        t_embed = self._t_to_embed(t_cont)
        u_pred = model(x_t, t_embed, mwir)

        # Recover x₀ estimate from velocity:
        # x_t = (1-t)·x₀ + t·ε  and  u = ε - x₀
        # → x₀ = x_t - t·u  (solve for x₀)
        t_b = t_cont.view(-1, 1, 1, 1)
        x0_pred = x_t - t_b * u_pred
        x0_pred = x0_pred.clamp(-1, 1)

        return u_pred, u_target, x0_pred, x0, t_embed

    def training_loss_weighted(
        self,
        model:  Callable,
        x0:     torch.Tensor,
        mwir:   torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the weighted flow matching loss.
        Returns (total_loss, loss_dict) — compatible with existing trainer pattern.
        """
        B = x0.shape[0]
        t_cont = self._sample_t(B, x0.device)
        u_pred, u_target, x0_pred, _, _ = self.training_losses(
            model, x0, mwir, t_cont
        )

        # Per-sample MSE over spatial dims
        mse_spatial = F.mse_loss(u_pred, u_target, reduction='none')
        mse_per_sample = mse_spatial.mean(dim=(1, 2, 3))   # (B,)

        # Optional per-timestep weighting
        weights = self._loss_weight(t_cont)
        loss = (weights * mse_per_sample).mean()

        return loss, {
            'fm_mse': loss.item(),
            't_mean': t_cont.mean().item(),
        }, x0_pred

    # ── Inference: ODE samplers ────────────────────────────────────

    @torch.no_grad()
    def sample_euler(
        self,
        model:          Callable,
        mwir:           torch.Tensor,
        shape:          Tuple,
        num_steps:      int   = 20,
        guidance_scale: float = 1.0,
        device:         str   = 'cuda',
        verbose:        bool  = False,
    ) -> torch.Tensor:
        """
        Euler ODE integration from t=1 (noise) to t=0 (data).

        dx/dt = -u_θ(x_t, t, cond)
        x_{t-Δt} = x_t + u_θ · Δt   (Δt > 0 since we go t=1→0, so dt = -1/N)

        Fast and simple. For most use cases 20 steps gives quality
        equivalent to DDIM 50 steps.
        """
        B    = mwir.shape[0]
        x    = torch.randn(shape, device=device)
        dt   = 1.0 / num_steps
        null = mwir.mean(dim=(-2, -1), keepdim=True).expand_as(mwir)

        for i in range(num_steps):
            t_val  = 1.0 - i * dt                       # t goes 1 → dt
            t_cont = torch.full((B,), t_val, device=device)
            t_emb  = self._t_to_embed(t_cont)

            if guidance_scale > 1.0:
                u_cond   = model(x, t_emb, mwir)
                u_uncond = model(x, t_emb, null)
                u = u_uncond + guidance_scale * (u_cond - u_uncond)
            else:
                u = model(x, t_emb, mwir)

            x = x - u * dt                              # Euler step toward t=0

            if verbose and (i + 1) % 5 == 0:
                print(f"  FM Euler {i+1}/{num_steps}", end='\r')

        if verbose:
            print()
        return x.clamp(-1, 1)

    @torch.no_grad()
    def sample_heun(
        self,
        model:          Callable,
        mwir:           torch.Tensor,
        shape:          Tuple,
        num_steps:      int   = 20,
        guidance_scale: float = 1.0,
        device:         str   = 'cuda',
        verbose:        bool  = False,
    ) -> torch.Tensor:
        """
        Heun's method (2nd-order Runge-Kutta) for ODE integration.

        Same NFE cost per step as Euler but 2nd-order accurate → better
        quality for the same number of steps. Recommended for inference.

        Two evaluations per step:
            u₁ = u_θ(x_t,   t,      cond)   — predictor
            u₂ = u_θ(x̃_{t-Δt}, t-Δt, cond)  — corrector
            x_{t-Δt} = x_t - 0.5·(u₁ + u₂)·Δt
        """
        B    = mwir.shape[0]
        x    = torch.randn(shape, device=device)
        dt   = 1.0 / num_steps
        null = mwir.mean(dim=(-2, -1), keepdim=True).expand_as(mwir)

        def eval_u(x_in, t_val):
            t_cont = torch.full((B,), t_val, device=device)
            t_emb  = self._t_to_embed(t_cont)
            if guidance_scale > 1.0:
                u_c = model(x_in, t_emb, mwir)
                u_u = model(x_in, t_emb, null)
                return u_u + guidance_scale * (u_c - u_u)
            return model(x_in, t_emb, mwir)

        for i in range(num_steps):
            t_cur  = 1.0 - i * dt
            t_next = t_cur - dt

            u1 = eval_u(x, t_cur)
            x_pred = x - u1 * dt                        # Euler predictor

            if t_next > 0:
                u2 = eval_u(x_pred, t_next)
                x = x - 0.5 * (u1 + u2) * dt           # Heun corrector
            else:
                x = x_pred                              # last step: Euler

            if verbose and (i + 1) % 5 == 0:
                print(f"  FM Heun {i+1}/{num_steps}", end='\r')

        if verbose:
            print()
        return x.clamp(-1, 1)

    # Alias so DiTTrainer._generate_fn can call the same interface
    # as the old DDIM sampler
    def sample(self, model, mwir, shape, num_steps=20, guidance_scale=1.0,
               device='cuda', verbose=False, method='heun'):
        if method == 'euler':
            return self.sample_euler(model, mwir, shape, num_steps,
                                     guidance_scale, device, verbose)
        return self.sample_heun(model, mwir, shape, num_steps,
                                guidance_scale, device, verbose)
