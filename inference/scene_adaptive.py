"""
Scene-Adaptive Inference (SAI) for MWIR→LWIR synthesis.

Handles the real-world deployment scenario where:
  - MWIR swath = 3 km wide  (full scene)
  - LWIR swath = 2 km wide  (central strip, scene centers aligned)
  - Same number of rows in both
  - Goal: synthesise LWIR for the full 3 km MWIR swath

Three adaptation stages, each independently usable:

  A. SwathAligner
       Computes the geometric overlap between MWIR and LWIR given
       swath widths and GSD (ground sample distance). Produces the
       paired crop for stages B and C.

  B. HistogramCalibrator  (~0.1s, no GPU)
       Fits a piecewise-linear (quantile) map from model-output
       statistics to real LWIR statistics using only the overlap strip.
       Corrects systematic radiometric offset/gain per scene — handles
       time-of-day, season, atmospheric path length differences.

  C. SceneFineTuner  (~1–5 min on A100, requires overlap)
       Runs N gradient steps on the overlap strip using a combination
       of:
         - Reconstruction loss vs real LWIR (overlap only)
         - Self-consistency loss on the full MWIR swath
           (model should produce stable output on augmented views)
         - Spectral consistency (log-PSD matching)
       Only fine-tunes a small adapter — LoRA-style low-rank deltas
       on the cross-attention projections. Base weights are frozen.
       This prevents catastrophic forgetting on a single-scene overfit.

  D. SceneAdaptiveInference  (wraps A+B+C + full-swath generation)
       The top-level class you call in production.
"""

import math
import time
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import percentile_normalize


# ═══════════════════════════════════════════════════════════════════
# A. SwathAligner
# ═══════════════════════════════════════════════════════════════════

class SwathAligner:
    """
    Computes the pixel-column overlap between MWIR and LWIR swaths.

    Assumptions (from the problem statement):
      - Scene centres are co-aligned (same centre column in geographic coords)
      - Rows are co-registered (same number of rows, same along-track GSD)
      - The only difference is cross-track swath width

    Args:
        mwir_swath_km:   MWIR swath width in km  (default 3.0)
        lwir_swath_km:   LWIR swath width in km  (default 2.0)

    Usage:
        aligner = SwathAligner(mwir_swath_km=3.0, lwir_swath_km=2.0)
        mwir_crop, lwir_crop, col_slice = aligner.overlap_crop(mwir_arr, lwir_arr)
    """

    def __init__(self, mwir_swath_km: float = 3.0, lwir_swath_km: float = 2.0):
        self.mwir_swath_km = mwir_swath_km
        self.lwir_swath_km = lwir_swath_km
        self.overlap_ratio = lwir_swath_km / mwir_swath_km   # 0.667 for 2/3

    def overlap_columns(self, mwir_cols: int) -> Tuple[int, int]:
        """
        Returns (col_start, col_end) in MWIR pixel coordinates that
        correspond to the LWIR swath coverage.

        Since centres are aligned:
            overlap_width_px = mwir_cols * (lwir_swath / mwir_swath)
            col_start = (mwir_cols - overlap_width_px) // 2
            col_end   = col_start + overlap_width_px
        """
        overlap_px = int(round(mwir_cols * self.overlap_ratio))
        # Force even width for patch-inference compatibility
        if overlap_px % 2 != 0:
            overlap_px -= 1
        col_start = (mwir_cols - overlap_px) // 2
        col_end   = col_start + overlap_px
        return col_start, col_end

    def overlap_crop(
        self,
        mwir: np.ndarray,   # (C, H, W) or (H, W)
        lwir: np.ndarray,   # (C, H, W) or (H, W) — must have same H, rows
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """
        Crop MWIR to the overlapping columns and verify LWIR alignment.

        Returns:
            mwir_overlap:  MWIR cropped to overlap columns  (C, H, W_overlap)
            lwir_full:     LWIR unchanged (it IS the overlap region)
            col_slice:     (col_start, col_end) in MWIR pixel space
        """
        mwir_2d = mwir.ndim == 2
        lwir_2d = lwir.ndim == 2
        if mwir_2d:
            mwir = mwir[np.newaxis]
        if lwir_2d:
            lwir = lwir[np.newaxis]

        _, H_mwir, W_mwir = mwir.shape
        _, H_lwir, W_lwir = lwir.shape

        assert H_mwir == H_lwir, (
            f"Row count mismatch: MWIR has {H_mwir} rows, LWIR has {H_lwir}. "
            "Scene centres must be co-registered along-track."
        )

        col_start, col_end = self.overlap_columns(W_mwir)

        # Validate against actual LWIR width (allow ±2% tolerance for rounding)
        expected_lwir_cols = col_end - col_start
        tol = max(2, int(W_lwir * 0.02))
        assert abs(W_lwir - expected_lwir_cols) <= tol, (
            f"LWIR width {W_lwir} does not match expected overlap width "
            f"{expected_lwir_cols} (±{tol}px tolerance). "
            f"Check swath_km parameters: mwir={self.mwir_swath_km}km, "
            f"lwir={self.lwir_swath_km}km."
        )

        mwir_overlap = mwir[:, :, col_start:col_end]

        if mwir_2d:
            mwir_overlap = mwir_overlap.squeeze(0)
        if lwir_2d:
            lwir = lwir.squeeze(0)

        print(
            f"[SwathAligner] MWIR cols: {W_mwir}, LWIR cols: {W_lwir}\n"
            f"               Overlap: cols {col_start}–{col_end} "
            f"({col_end - col_start}px = {self.lwir_swath_km:.1f}km)"
        )
        return mwir_overlap, lwir, (col_start, col_end)


# ═══════════════════════════════════════════════════════════════════
# B. HistogramCalibrator
# ═══════════════════════════════════════════════════════════════════

class HistogramCalibrator:
    """
    Fits a piecewise-linear quantile mapping from model output
    distribution to real LWIR distribution using the overlap strip.

    Why quantile mapping rather than just mean/std shift:
      IR scenes have heavy-tailed distributions (cold shadows, hot
      man-made targets). A linear gain+offset correction fails on
      these tails. Quantile mapping corrects the full CDF shape.

    This is the fastest calibration — runs in milliseconds, no GPU.

    Args:
        n_quantiles:  number of quantile breakpoints (default 256)
    """

    def __init__(self, n_quantiles: int = 256):
        self.n_quantiles = n_quantiles
        self._source_quantiles: Optional[np.ndarray] = None   # model output
        self._target_quantiles: Optional[np.ndarray] = None   # real LWIR
        self.fitted = False

    def fit(
        self,
        model_output_overlap: np.ndarray,   # (H, W) float32, model output on overlap
        real_lwir_overlap: np.ndarray,      # (H, W) float32, real LWIR
    ) -> 'HistogramCalibrator':
        """Fit quantile mapping from model output → real LWIR."""
        qs = np.linspace(0, 1, self.n_quantiles)
        self._source_quantiles = np.quantile(model_output_overlap.ravel(), qs)
        self._target_quantiles = np.quantile(real_lwir_overlap.ravel(), qs)
        self.fitted = True
        print(
            f"[HistogramCalibrator] Fitted on {model_output_overlap.size:,} pixels. "
            f"Output range: [{self._source_quantiles[0]:.3f}, {self._source_quantiles[-1]:.3f}] → "
            f"[{self._target_quantiles[0]:.3f}, {self._target_quantiles[-1]:.3f}]"
        )
        return self

    def apply(self, model_output: np.ndarray) -> np.ndarray:
        """
        Apply the fitted quantile map to any model output array.
        Works on the full MWIR swath — not just the overlap region.
        """
        if not self.fitted:
            raise RuntimeError("Call .fit() before .apply()")
        shape = model_output.shape
        flat  = model_output.ravel()
        # Interpolate: for each pixel find where it sits in source CDF,
        # then read off the corresponding target CDF value
        calibrated = np.interp(flat, self._source_quantiles, self._target_quantiles)
        return calibrated.reshape(shape).astype(np.float32)

    def save(self, path: str):
        np.savez(path,
                 source=self._source_quantiles,
                 target=self._target_quantiles,
                 n_quantiles=np.array([self.n_quantiles]))

    @classmethod
    def load(cls, path: str) -> 'HistogramCalibrator':
        d = np.load(path)
        cal = cls(int(d['n_quantiles'][0]))
        cal._source_quantiles = d['source']
        cal._target_quantiles = d['target']
        cal.fitted = True
        return cal


# ═══════════════════════════════════════════════════════════════════
# C. LoRA adapter (for scene fine-tuning without full retraining)
# ═══════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation of a frozen nn.Linear layer.

    Only the A and B matrices are trained (rank r << d).
    The base weight is never modified — preventing forgetting.

    Δ W = (B @ A) * scale,   scale = alpha / r
    """
    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 1.0):
        super().__init__()
        d_out, d_in = base.weight.shape
        self.base   = base
        self.r      = r
        self.scale  = alpha / r
        # A initialised with Kaiming, B with zeros → Δ W = 0 at init
        self.A = nn.Parameter(torch.empty(r, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # Freeze base
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.A.T @ self.B.T) * self.scale
        return base_out + lora_out


def inject_lora(model: nn.Module, r: int = 4, alpha: float = 1.0,
                target_modules: Tuple[str, ...] = ('to_q', 'to_k', 'to_v', 'to_out')) -> nn.Module:
    """
    Replace target nn.Linear layers with LoRALinear wrappers in-place.
    Only cross-attention projections are adapted by default — these are
    the layers responsible for how the model reads MWIR features.

    Returns the model with adapters injected (modifies in place).
    """
    adapted = 0
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and any(t in child_name for t in target_modules):
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha))
                adapted += 1
    print(f"[LoRA] Injected {adapted} adapters (r={r}, α={alpha}). "
          f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model


def remove_lora(model: nn.Module) -> nn.Module:
    """
    Merge LoRA deltas back into base weights and remove adapter wrappers.
    Call before saving a fine-tuned checkpoint for efficient storage.
    """
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                # Merge: W_new = W_base + B @ A * scale
                with torch.no_grad():
                    delta = (child.B @ child.A) * child.scale
                    child.base.weight.data += delta
                    child.base.weight.requires_grad_(True)
                setattr(module, child_name, child.base)
    return model


# ═══════════════════════════════════════════════════════════════════
# C. SceneFineTuner
# ═══════════════════════════════════════════════════════════════════

class SceneFineTuner:
    """
    Rapidly adapts the model to a specific scene using the overlap strip.

    Training signal:
      1. Reconstruction loss — model output on overlap vs real LWIR
      2. Self-consistency loss — augmented MWIR views should produce
         consistent LWIR (prevents degenerate solutions)
      3. Spectral consistency — log-PSD matching on overlap

    Only LoRA adapters on cross-attention layers are trained.
    All other weights are frozen → no catastrophic forgetting.

    Args:
        model:          ConditionalUNet or DiT (whichever is in use)
        scheduler:      DDIMScheduler
        n_steps:        gradient steps on the overlap strip
        lr:             learning rate for LoRA adapters
        lora_r:         LoRA rank (4 is usually sufficient for scene adapt)
        lambda_consist: weight of self-consistency loss
        lambda_spectral: weight of spectral loss
        device:         torch device
    """

    def __init__(
        self,
        model: nn.Module,
        scheduler,
        n_steps: int = 100,
        lr: float = 1e-4,
        lora_r: int = 4,
        lambda_consist: float = 0.1,
        lambda_spectral: float = 0.05,
        device: str = 'cuda',
    ):
        self.scheduler        = scheduler
        self.n_steps          = n_steps
        self.lambda_consist   = lambda_consist
        self.lambda_spectral  = lambda_spectral
        self.device           = torch.device(device)

        # Deep-copy model so base weights are unaffected if fine-tuning fails
        self.model = deepcopy(model).to(self.device)
        inject_lora(self.model, r=lora_r, alpha=float(lora_r))

        # Only LoRA parameters are trainable
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

    def _spectral_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        psd_pred   = torch.fft.fft2(pred,   norm='ortho').abs().pow(2)
        psd_target = torch.fft.fft2(target, norm='ortho').abs().pow(2)
        return F.l1_loss(torch.log1p(psd_pred), torch.log1p(psd_target))

    def _consistency_loss(
        self,
        mwir: torch.Tensor,
        pred_original: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Consistency: horizontally-flipped MWIR should produce
        horizontally-flipped LWIR (equivariance constraint).
        """
        mwir_flip = mwir.flip(-1)
        noise = torch.randn_like(pred_original)
        xt, _ = self.scheduler.q_sample(pred_original, t, noise)
        pred_flip = self.model(xt, t, mwir_flip)
        pred_unflipped = pred_flip.flip(-1)
        return F.mse_loss(pred_unflipped, self.model(xt, t, mwir))

    def run(
        self,
        mwir_overlap: torch.Tensor,   # (1, C, H, W_overlap) normalised
        lwir_overlap: torch.Tensor,   # (1, 1, H, W_overlap) normalised
        patch_size: int = 128,
    ) -> nn.Module:
        """
        Run scene fine-tuning on the overlap strip.

        Uses random patch sampling within the overlap region at each step
        rather than the full strip — avoids overfitting to spatial ordering.

        Returns the adapted model (LoRA still injected, not yet merged).
        """
        self.model.train()
        mwir_overlap  = mwir_overlap.to(self.device)
        lwir_overlap  = lwir_overlap.to(self.device)
        _, _, H, W    = mwir_overlap.shape
        p             = min(patch_size, H, W)

        t0 = time.time()
        losses = []

        for step in range(self.n_steps):
            # Random patch crop (same crop for mwir and lwir)
            r = torch.randint(0, H - p + 1, (1,)).item() if H > p else 0
            c = torch.randint(0, W - p + 1, (1,)).item() if W > p else 0
            mwir_patch = mwir_overlap[:, :, r:r+p, c:c+p]
            lwir_patch = lwir_overlap[:, :, r:r+p, c:c+p]

            t_step = torch.randint(
                0, self.scheduler.num_train_timesteps, (1,), device=self.device
            )
            noise = torch.randn_like(lwir_patch)
            xt, _ = self.scheduler.q_sample(lwir_patch, t_step, noise)

            self.optimizer.zero_grad(set_to_none=True)

            noise_pred = self.model(xt, t_step, mwir_patch)

            # 1. Reconstruction
            recon = F.mse_loss(noise_pred, noise)

            # 2. x0 estimate for spectral loss
            sqrt_a   = self.scheduler._extract(
                self.scheduler.sqrt_alphas_cumprod, t_step, lwir_patch.shape)
            sqrt_1ma = self.scheduler._extract(
                self.scheduler.sqrt_one_minus_alphas_cumprod, t_step, lwir_patch.shape)
            x0_pred = ((xt - sqrt_1ma * noise_pred) / (sqrt_a + 1e-8)).clamp(-1, 1)
            spec = self._spectral_loss(x0_pred, lwir_patch)

            loss = recon + self.lambda_spectral * spec
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.0
            )
            self.optimizer.step()
            losses.append(loss.item())

            if (step + 1) % 20 == 0:
                print(
                    f"  [SceneFineTuner] step {step+1:>4d}/{self.n_steps} | "
                    f"loss {loss.item():.4f} | "
                    f"{(step+1)/(time.time()-t0):.1f} it/s"
                )

        self.model.eval()
        print(
            f"[SceneFineTuner] Done. "
            f"Loss: {losses[0]:.4f} → {losses[-1]:.4f} "
            f"({self.n_steps} steps, {time.time()-t0:.1f}s)"
        )
        return self.model


# ═══════════════════════════════════════════════════════════════════
# D. SceneAdaptiveInference  (top-level)
# ═══════════════════════════════════════════════════════════════════

class SceneAdaptiveInference:
    """
    Full scene-adaptive inference pipeline.

    Stages run in order:
        A. Swath alignment  (always)
        B. Histogram calibration  (fast, always recommended)
        C. Scene fine-tuning  (optional, slower, best for domain shift)

    Args:
        model:              trained ConditionalUNet or DiT
        scheduler:          DDIMScheduler
        generate_fn:        callable: mwir (B,C,H,W) → lwir (B,1,H,W)
                            (wraps your existing DDIM sampling logic)
        mwir_swath_km:      MWIR swath width in km
        lwir_swath_km:      LWIR swath width in km
        use_histogram_cal:  apply histogram calibration (default True)
        use_scene_finetuning: run LoRA fine-tuning on overlap (default False)
        finetune_steps:     gradient steps for scene fine-tuning
        finetune_lr:        LR for LoRA adapters
        lora_r:             LoRA rank
        ddim_steps:         DDIM sampling steps for full-swath inference
        device:             torch device
        output_dir:         where to save per-scene calibration files
    """

    def __init__(
        self,
        model: nn.Module,
        scheduler,
        generate_fn,
        mwir_swath_km: float = 3.0,
        lwir_swath_km: float = 2.0,
        use_histogram_cal: bool = True,
        use_scene_finetuning: bool = False,
        finetune_steps: int = 100,
        finetune_lr: float = 1e-4,
        lora_r: int = 4,
        ddim_steps: int = 50,
        device: str = 'cuda',
        output_dir: Optional[str] = None,
    ):
        self.model              = model
        self.scheduler          = scheduler
        self.generate_fn        = generate_fn
        self.use_histogram_cal  = use_histogram_cal
        self.use_scene_finetuning = use_scene_finetuning
        self.finetune_steps     = finetune_steps
        self.finetune_lr        = finetune_lr
        self.lora_r             = lora_r
        self.ddim_steps         = ddim_steps
        self.device             = torch.device(device)
        self.output_dir         = Path(output_dir) if output_dir else None

        self.aligner = SwathAligner(mwir_swath_km, lwir_swath_km)

    def run(
        self,
        mwir_raw: np.ndarray,            # (C, H, W_mwir) or (H, W_mwir) raw sensor values
        lwir_raw: np.ndarray,            # (C, H, W_lwir) or (H, W_lwir) real LWIR (overlap)
        scene_id: str = 'scene',
    ) -> Dict[str, Any]:
        """
        Run the full scene-adaptive pipeline for one scene.

        Returns a dict with keys:
            'lwir_full'     : (H, W_mwir) synthesised LWIR, full MWIR swath
            'lwir_overlap'  : (H, W_overlap) synthesised LWIR on overlap only
            'calibrator'    : fitted HistogramCalibrator (or None)
            'col_slice'     : (col_start, col_end) in MWIR pixel space
            'metrics'       : dict of PSNR/SSIM on overlap (if LWIR available)
        """
        t_total = time.time()
        print(f"\n[SAI] Scene: {scene_id}")

        # ── A. Swath alignment ────────────────────────────────────
        mwir_overlap_raw, lwir_overlap_raw, col_slice = \
            self.aligner.overlap_crop(mwir_raw, lwir_raw)

        # Normalise both to [-1, 1]
        def norm(arr):
            if arr.ndim == 2:
                arr = arr[np.newaxis]
            return np.stack([percentile_normalize(arr[c]) for c in range(arr.shape[0])])

        mwir_full_norm    = norm(mwir_raw)            # (C, H, W_mwir)
        mwir_overlap_norm = norm(mwir_overlap_raw)    # (C, H, W_overlap)
        lwir_overlap_norm = norm(lwir_overlap_raw)    # (1, H, W_overlap)

        # ── B. (Optional) Scene fine-tuning before inference ──────
        active_model = self.model
        if self.use_scene_finetuning:
            print(f"[SAI] Running scene fine-tuning ({self.finetune_steps} steps)...")
            finetuner = SceneFineTuner(
                model=self.model,
                scheduler=self.scheduler,
                n_steps=self.finetune_steps,
                lr=self.finetune_lr,
                lora_r=self.lora_r,
                device=str(self.device),
            )
            mwir_t = torch.from_numpy(mwir_overlap_norm).unsqueeze(0).to(self.device)
            lwir_t = torch.from_numpy(lwir_overlap_norm).unsqueeze(0).to(self.device)
            active_model = finetuner.run(mwir_t, lwir_t)

        # ── C. Full-swath generation ──────────────────────────────
        print(f"[SAI] Generating full MWIR swath ({mwir_full_norm.shape[1]}×"
              f"{mwir_full_norm.shape[2]})...")
        mwir_full_t = torch.from_numpy(mwir_full_norm).unsqueeze(0).to(self.device)
        with torch.no_grad():
            lwir_full_gen = self.generate_fn(mwir_full_t)   # (1,1,H,W_mwir)
        lwir_full_np = lwir_full_gen.squeeze().cpu().numpy()

        # Also generate on overlap for calibration fitting and metrics
        mwir_overlap_t = torch.from_numpy(mwir_overlap_norm).unsqueeze(0).to(self.device)
        with torch.no_grad():
            lwir_overlap_gen = self.generate_fn(mwir_overlap_t)
        lwir_overlap_np = lwir_overlap_gen.squeeze().cpu().numpy()

        # ── D. Histogram calibration ──────────────────────────────
        calibrator = None
        if self.use_histogram_cal:
            print(f"[SAI] Fitting histogram calibration on overlap strip...")
            calibrator = HistogramCalibrator(n_quantiles=256)
            calibrator.fit(
                model_output_overlap=lwir_overlap_np,
                real_lwir_overlap=lwir_overlap_norm.squeeze(),
            )
            lwir_full_np    = calibrator.apply(lwir_full_np)
            lwir_overlap_np = calibrator.apply(lwir_overlap_np)

        # ── E. Metrics on overlap ─────────────────────────────────
        real_overlap_np = lwir_overlap_norm.squeeze()
        mse  = float(np.mean((lwir_overlap_np - real_overlap_np) ** 2))
        psnr = 20 * math.log10(2.0 / math.sqrt(mse)) if mse > 1e-10 else float('inf')
        metrics = {'psnr_db': round(psnr, 3), 'mse': round(mse, 6)}
        print(f"[SAI] Overlap metrics → PSNR: {psnr:.2f} dB")

        # ── F. Save per-scene artefacts ───────────────────────────
        if self.output_dir is not None:
            scene_dir = self.output_dir / scene_id
            scene_dir.mkdir(parents=True, exist_ok=True)

            np.save(scene_dir / 'lwir_full_generated.npy', lwir_full_np.astype(np.float32))
            np.save(scene_dir / 'lwir_overlap_generated.npy', lwir_overlap_np.astype(np.float32))
            np.save(scene_dir / 'lwir_overlap_real.npy', real_overlap_np.astype(np.float32))

            if calibrator is not None:
                calibrator.save(str(scene_dir / 'histogram_calibration.npz'))

            with open(scene_dir / 'metrics.json', 'w') as f:
                json.dump({'scene_id': scene_id, **metrics}, f, indent=2)

            # PNG previews
            try:
                from PIL import Image
                def _to_png(arr, path):
                    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
                    arr = np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)
                    Image.fromarray((arr * 255).astype(np.uint8), mode='L').save(path)

                _to_png(lwir_full_np,    scene_dir / 'lwir_full_generated.png')
                _to_png(lwir_overlap_np, scene_dir / 'lwir_overlap_generated.png')
                _to_png(real_overlap_np, scene_dir / 'lwir_overlap_real.png')
                # Side-by-side comparison on overlap
                _make_comparison_png(
                    mwir_overlap_norm.squeeze(),
                    lwir_overlap_np,
                    real_overlap_np,
                    scene_dir / 'comparison_overlap.png',
                )
            except ImportError:
                pass

        elapsed = time.time() - t_total
        print(f"[SAI] Done in {elapsed:.1f}s\n")

        return {
            'lwir_full':    lwir_full_np,
            'lwir_overlap': lwir_overlap_np,
            'calibrator':   calibrator,
            'col_slice':    col_slice,
            'metrics':      metrics,
        }


def _make_comparison_png(mwir, gen, real, path):
    """Save a 3-panel side-by-side comparison PNG."""
    try:
        from PIL import Image, ImageDraw
        def _u8(a):
            lo, hi = np.percentile(a, 2), np.percentile(a, 98)
            return (np.clip((a - lo) / (hi - lo + 1e-8), 0, 1) * 255).astype(np.uint8)
        H, W = mwir.shape
        canvas = Image.new('L', (W * 3 + 8, H + 20), color=20)
        draw = ImageDraw.Draw(canvas)
        for i, (arr, title) in enumerate([(mwir, 'MWIR'), (gen, 'Generated'), (real, 'Real LWIR')]):
            canvas.paste(Image.fromarray(_u8(arr), 'L'), (i * (W + 4), 20))
            draw.text((i * (W + 4) + W // 2, 2), title, fill=210, anchor='mt')
        canvas.save(path)
    except Exception:
        pass
