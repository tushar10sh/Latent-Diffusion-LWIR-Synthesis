# MWIR → LWIR Synthesis Pipeline — Complete Knowledge Summary

---

## 1. The Problem and Why It Is Hard

### Physical Background
Long-Wave Infrared (LWIR, 8–12 µm) and Mid-Wave Infrared (MWIR, 3–5 µm) are
fundamentally different physical channels:

- **MWIR** = thermal emission + solar reflection → high dynamic range, strong shadows
- **LWIR** = purely thermal emission → lower contrast, scene-temperature-dependent

The MWIR/LWIR ratio is governed by the Planck function:
  B(λ, T) = c₁ / [λ⁵ · (exp(c₂ / λT) - 1)]

For natural surfaces, BT_MWIR and BT_LWIR should be within ~15K. Larger deviations
indicate emissivity anomalies (metallic surfaces, specular reflection) or synthesis errors.

### Why Vanilla DDPM Fails
1. Thermal homogenisation: agricultural fields look identical in MWIR but have
   different temperatures in LWIR → pixel-MSE regresses to scene mean
2. Road/surface brightness errors: MWIR shows roads bright (solar reflection),
   LWIR shows them at their thermal temperature → model copies MWIR contrast incorrectly
3. Mean regression in low-contrast zones: vegetation, water bodies are nearly
   uniform in MWIR → model predicts average grey

---

## 2. The Three Pipeline Variants

### v1 — Baseline DDPM + Conditional UNet
- Entrypoint: `train.py --config configs/base.json`
- Pixel-space diffusion with cosine schedule
- Early fusion (concat MWIR at input) + cross-modal attention in decoder
- Loss: MSE + CFC + Spectral

### v2 — Improved Pixel-Space (targeted fixes from observed failures)
- Entrypoint: `train.py --config configs/improved_v2.json`
- Four additional fixes addressing specific failure modes:
  1. `LocalTextureGramLoss` — Gabor filter bank Gram matrices (subsampled to 64 patches)
  2. `SceneHistogramLoss` — vectorised KDE on 2048 pixel subsample
  3. `GlobalSceneContextEncoder` — scene-level stats via persistent forward hook
  4. `BridgeDiffusionScheduler` — starts from MWIR-derived prior

### LDM — Latent Diffusion + Conditional DiT (best quality)
- Entrypoint: `train_ldm.py --config configs/ldm.json`
- Two stages: VAE (Stage 1) + Conditional DiT with Flow Matching (Stage 2)

---

## 3. Architecture Decisions and Rationale

### Conditional UNet (v1/v2)
- **Spectral norm** on all convs EXCEPT zero-init layers (see Bug #1 below)
- **zero_conv()** for residual output layers — plain Conv2d, no spectral_norm
- **AdaGN** (Adaptive Group Norm, eps=1e-3): timestep + MWIR context modulates every ResBlock
- **GroupNorm eps=1e-3** everywhere: prevents 1/sqrt(0) in bfloat16 on homogeneous IR regions
- **Cross-modal attention** in decoder: queries MWIR features at each scale
- **Skip connection structure**: decoder needs (num_res_blocks + 1) pops per level.
  Bridge skip pushed AFTER downsample (resolution R/2), not before (R).
  With channel_mults=[1,2,4,8] and num_res_blocks=2: 12 pushes, 12 pops ✓

### KL-VAE (LDM Stage 1)
- **f=4 compression**: 256→64 spatial (not f=8) to preserve thermal edge structure
- **Gabor perceptual loss**: no VGG needed — 4 scales × 8 orientations covers IR frequencies
- **KL annealing**: β linearly increases 0 → 1e-4 over 10k steps (prevents posterior collapse)
- **Scale factor**: computed AFTER training as 1/std(z_raw), per-channel not global
- **GroupNorm eps=1e-3** throughout

### Conditional DiT (LDM Stage 2)
- **DiT-B/4**: hidden_dim=768, depth=12, heads=12, patch_size=4 → 256 tokens from 64×64 latent
- **2D RoPE**: ONLY applied to patch tokens, not register tokens
  `n_registers=self.num_registers` must be passed to every DiTBlock call
- **Register tokens** (4): prevent attention sink artefacts on flat IR regions
- **QK-Norm** (RMSNorm on Q and K): prevents attention logit spikes
- **adaLN-Zero**: timestep + MWIR global statistics modulate every DiT block
- **Learned null embedding**: `self.null_mwir = nn.Parameter(torch.zeros(1, C, H, W))`
  for CFG — spatial mean null causes tile artefact (see Bug #8 below)

---

## 4. Loss Functions

### Standard Losses
- **MSE / velocity MSE**: primary signal
- **CFC (Characteristic Function Consistency)**: empirical ECF patch-by-patch.
  IMPORTANT: max_freq=1.0 (not 5.0). CharDiff (WACV 2025) proves CF decays as
  O(1/(1+|u|^(d/2))). At u=5, |φ| ≈ 4e-6 — effectively zero. 81% of test
  points at max_freq=5.0 were wasted.
  Always computed in float32 — fft2/cos/sin are unstable in bfloat16.
- **Spectral consistency**: L1 on log power spectral density.
  Always computed in float32 — fft2 not in bfloat16 dispatch list.
- **Gram loss** (v2): Gabor filter Gram matrices, max_patches=64 subsample.
  Gabor kernels must be padded to uniform k_max before torch.stack.
- **Histogram loss** (v2): vectorised KDE soft histogram with S=2048 pixel subsample.
  Single (B, S, bins) broadcast op replaces 64-iteration Python loop.

### Relationship to CharDiff (Sinha & Moorthi, WACV 2025)
- ECF formula and squared distance metric are IDENTICAL to CharDiff Eq.3/6.
- KEY DIFFERENCE: CharDiff's training loss compares ECF of NOISY x_t against
  the ANALYTICAL Gaussian CF of p(x_t) = N(√ᾱ_t·μ, (1-ᾱ_t)·I).
  Our CFC compares ECF of clean x0_pred patches against real x0 patches.
- CharDiff also has a SAMPLING-TIME gradient correction (Alg. 2/3) applied at
  each denoising step — we do not implement this (would be a plug-and-play
  inference enhancement, no retraining needed).
- CharDiff notes its approach lags in latent space sampling — relevant for LDM.

### Flow Matching (replaces DDPM in LDM)
Straight-line interpolation: x_t = (1-t)·x₀ + t·ε
Model learns constant velocity u = ε - x₀
Training: L = E_t[||model(x_t, t, MWIR) - u_target||²]
x₀ recovery: x₀ = x_t - t·u_pred
Logit-Normal time sampling concentrates training at t≈0.5.
Heun's method at inference: 2nd-order ODE, same NFE as Euler but better quality.
null_cond MUST be passed to sample_heun/sample_euler when guidance_scale > 1.0.

### Planck Ratio Loss (physics-informed)
Converts normalised pixel values → DN → radiance → Brightness Temperature.
Penalises |BT_LWIR_generated - BT_MWIR| > allowed_delta_K.
Weighted by MWIR local thermal contrast.
Set lambda_planck=0.0 to disable. Fill sensor calibration values before enabling.

---

## 5. Latent Space Calibration (Critical)

### What scale_factor is for
The DiT's cosine/flow schedule assumes z ~ N(0,1). If the latent has std ≠ 1,
the schedule is miscalibrated and spatial collapse occurs.

### The fix (per-channel)
  scale_factor[c] = 1 / std(z_raw[c] - mean(z_raw[c]))
  latent_mean[c]  = mean(z_raw[c])
  z_DiT[c] = (z_raw[c] - latent_mean[c]) * scale_factor[c]  →  N(0,1)

Per-channel is correct because IR latent spaces are multimodal — land cover
clusters appear at different positions per channel.

### Calibration values (your run)
  ch0: raw_mean=1.3528  raw_std=0.3101  scale_factor=3.2238
  ch1: raw_mean=1.7968  raw_std=0.7346  scale_factor=1.3602
  ch2: raw_mean=-0.6939 raw_std=0.3050  scale_factor=3.2693
  ch3: raw_mean=-0.1852 raw_std=0.3949  scale_factor=2.5287

---

## 6. Data Pipeline

### Normalisation
Store raw physical DN values. The dataset's per-image p2-p98 stretch is the
only normalisation — do not pre-normalise.
One outlier (DN=3804) compressed 94% of typical scenes into 6% of [-1,1] when
using global min/max — the dataset's second percentile_normalize corrected it.
dataset._load() now sanitises NaN/Inf with per-channel median (defensive).

### Image Size
Data is 224×224 natively. Configs use image_size=256 (bilinear upsampling).
PSNR impact of upsampling: negligible (VAE achieved 42.46 dB).
attn_resolutions=[16, 8] matches the actual downsampled resolutions from 256.

---

## 7. Precision System

Config key: `"precision": "float32"` (default, recommended) or `"bfloat16"`.

All three trainers derive `self.use_amp` and `self.amp_dtype` from this key.
GradScaler is NEVER used in either mode. With bfloat16, GradScaler's default
init_scale=65536 causes it to detect false inf/nan and skip every optimizer step.

bfloat16 instabilities in this pipeline:
1. fft2 not in bfloat16 dispatch list → NaN in SpectralConsistencyLoss
2. GroupNorm eps=1e-5 underflows in bfloat16 on homogeneous IR (var=0)
3. CFC cos/sin chaotic at bfloat16's 7-bit mantissa precision

All three are fixed (float32 upcast in losses, eps=1e-3), but float32 training
is recommended when compute is not a constraint.

VAE trained correctly under bfloat16+GradScaler because:
- L1 recon loss has gradients in {-1, 0, +1} — always bounded
- KL regularisation prevents homogeneous activations (var≠0 in GroupNorm)
- VAE has neither fft2 nor CFC losses

---

## 8. Bugs Found and Fixed (Comprehensive)

| Bug | Root Cause | Fix | File |
|-----|-----------|-----|------|
| NaN from step 1 (PRIMARY) | zero_module(spectral_norm(conv)): σ=0, W/σ=NaN | zero_conv(): plain Conv2d, no spectral_norm | conditional_unet.py |
| DiT tile pattern | Spatial mean null conditioning trains model to output flat LWIR | Learned nn.Parameter null_mwir | ldm_trainer.py |
| NaN from step 1 (secondary) | GradScaler+bfloat16: skips every optimizer step | Remove GradScaler; plain loss.backward() | all trainers |
| fft2 NaN in bfloat16 | fft2 not in bfloat16 dispatch list | pred.float() before fft2 | diffusion_scheduler.py |
| GroupNorm inf in bfloat16 | eps=1e-5 underflows for homogeneous IR | eps=1e-3 everywhere | unet, vae |
| CFC instability at high t | x0_pred diverges to O(600) at t=999 | Hard clamp x0_pred to [-1,1] | diffusion_scheduler.py |
| CFC max_freq wrong | 81% of test points at near-zero CF signal | max_freq=1.0 (was 5.0) | diffusion_scheduler.py |
| Skip connection shape mismatch | Bridge skip pushed before downsample (R), consumed at R/2 | Push bridge AFTER downsample | conditional_unet.py |
| DiT RoPE size mismatch | n_registers=0 default; RoPE applied to 260 not 256 tokens | Pass n_registers=self.num_registers | ldm/dit.py |
| VAE scale_factor wrong | Stored std not 1/std; global scalar not per-channel | Per-channel latent_mean + 1/std | ldm/vae.py |
| model_fn argument error | ddim_sample_bridge() takes 'model', not 'model_fn' | Rename in _generate_fn | improved_trainer.py |
| PSNR=inf masks NaN | NaN > 1e-10 = False in Python → returned inf | Explicit isfinite check → -1 dB sentinel | visualizer.py |
| SceneHistogramLoss 100× slow | 64 sequential Python iterations → 384 CUDA kernels | Vectorised (B,S,bins) + 2048 pixel subsample | targeted_improvements.py |
| Gabor kernel torch.stack crash | Different kernel sizes (7,13,25) can't stack | Zero-pad all to k_max before stack | targeted_improvements.py |
| Per-step monkey-patch | New nn.Module created every step, breaks compile | Persistent register_forward_hook | improved_trainer.py |
| Global normalisation no-op | Outlier set global_max=3804, compressing 94% scenes | Use raw physical DN files | data/dataset.py |
| KL log interpretation | Eval summed over all 16,384 dims vs per-dim threshold | Report mean KL per dim | eval_vae.py |

---

## 9. Performance Analysis

### Why ImprovedTrainer was 100× slower than DiT (2600 vs 250K steps/48hr)

| Cause | Impact | Fix Applied |
|-------|--------|-------------|
| SceneHistogramLoss: 64-iter Python loop → 384 kernel launches | ~20× | Vectorised + subsampled |
| Pixel-space 256×256 vs latent 64×64 (16× more FLOPs per conv) | ~6× | Inherent — cannot eliminate |
| LocalTextureGramLoss: Gabor kernel size bug + all 392 patches | ~4× | Uniform kernels + max_patches=64 |
| SceneAugmentedTimeEmbed: new nn.Module every step | ~1.3× | Persistent forward hook |
| prior_net + scene_encoder at full 256×256 | ~1.1× | Inherent |

Expected after fixes: ~30,000-40,000 steps/48hr.
LDM remains faster because cause 2 is architectural and unavoidable.

### Step time reference
- DiT step (latent 64×64, FM loss): ~0.8 ms → 250K steps/48hr
- VAE step (pixel 256×256, L1+KL+Gabor): ~3 ms → 50K steps/48hr
- Improved step (before fixes): ~66 ms → 2,600 steps/48hr
- Improved step (after fixes): ~3-5 ms → 30-50K steps/48hr

---

## 10. VAE Evaluation Results (your actual numbers)

| Metric | Value | Status |
|--------|-------|--------|
| PSNR mean | 42.46 dB | PASS |
| PSNR p10 | 37.01 dB | PASS |
| SSIM mean | 0.9931 | PASS |
| KL/dim | 6.46 nats | PASS (healthy 2–10 range) |
| z_std (after recal) | 1.015 | PASS |
| z_mean_abs (after recal) | 0.131 | PASS |
| PSD error | 0.00116 | PASS |

**VAE is fully calibrated. Use vae_final_recal.pt for Stage 2 DiT training.**

---

## 11. File Inventory (8,171+ lines total)

### Models (core architecture)
- `models/conditional_unet.py` — v1/v2 UNet: zero_conv, eps=1e-3, bridge skip fix
- `models/diffusion_scheduler.py` — DDPM/DDIM + CFC (max_freq=1.0) + Spectral (float32)
- `models/targeted_improvements.py` — v2: vectorised histogram, patched Gabor, hook-based scene ctx
- `models/flow_matching.py` — FM scheduler + Heun/Euler with null_cond parameter
- `models/planck_loss.py` — Physics-informed Planck ratio loss
- `models/ldm/vae.py` — KL-VAE with per-channel calibration, eps=1e-3
- `models/ldm/dit.py` — Conditional DiT with RoPE + registers + QK-Norm

### Training
- `training/trainer.py` — v1: precision system, NaN guard, correct lwir shape
- `training/improved_trainer.py` — v2: hook-based scene ctx, fixed _generate_fn
- `training/ldm_trainer.py` — LDM: learned null_mwir, float32 backward, checkpoint save
- `training/visualizer.py` — NaN-safe PSNR (-1 dB sentinel), fixed sample sets

### Inference
- `inference/infer.py` — v1/v2 sliding-window inference
- `inference/ldm_infer.py` — LDM: loads null_mwir from checkpoint
- `inference/scene_adaptive.py` — SwathAligner + HistogramCalibrator + LoRA

### Data / Evaluation
- `data/dataset.py` — NaN sanitisation, percentile normalisation
- `eval_vae.py` — VAE reconstruction + latent + KL + PSD + verdict
- `diag_normalization.py` — Double-normalisation detection

---

## 12. Recommended Next Steps

1. Run improved v2 trainer (fixes applied) — expect 30K+ steps/48hr now
2. Monitor: `Total: <0.5 | MSE: <0.4 | CFC: <0.01 | Gram: <0.05` at convergence
3. For LDM Stage 2: start with `vae_final_recal.pt`, `precision: float32`
4. Monitor DiT: `FM: <0.1 | CFC: <0.01 | Planck: <0.02 | BT-MAE: <5K`
5. Fill sensor calibration coefficients in configs/ldm.json before enabling Planck loss
6. Consider CharDiff Algorithm 3 as inference-time plug-and-play enhancement
   (no retraining — wraps existing DDIM step with CF gradient correction)
7. Do NOT use CharDiff in latent space — paper notes it lags there

---

## 13. Key Configuration Reference

```json
{
  "precision": "float32",
  "use_flow_matching": true,
  "lambda_cfc": 0.10,
  "lambda_spectral": 0.05,
  "lambda_gram": 0.05,
  "lambda_hist": 0.05,
  "planck": { "lambda_planck": 0.0 }
}
```

Training log keys to monitor:
- `FM:` / `ddpm_mse:` — primary velocity/noise MSE — decreases from ~0.5 → <0.1
- `CFC:` — characteristic function distance — target <0.01
- `Planck:` — BT Huber penalty — <0.02 healthy; >0.1 = unphysical outputs
- `BT-MAE:` — mean absolute BT error vs real LWIR — target <5K
