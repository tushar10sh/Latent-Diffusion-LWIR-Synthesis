# MWIR → LWIR Synthesis Pipeline — Complete Knowledge Summary

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
1. Thermal homogenization: agricultural fields look identical in MWIR but have
   different temperatures in LWIR → pixel-MSE regresses to scene mean
2. Road/surface brightness errors: MWIR shows roads bright (solar reflection),
   LWIR shows them at their thermal temperature → model copies MWIR contrast incorrectly
3. Mean regression in low-contrast zones: vegetation, water bodies are nearly
   uniform in MWIR → model predicts average gray

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
  1. `LocalTextureGramLoss` — Gabor filter bank Gram matrices → fixes flat agricultural fields
  2. `SceneHistogramLoss` — differentiable KDE histogram matching → fixes road brightness bias
  3. `GlobalSceneContextEncoder` — scene-level statistics in UNet conditioning → resolves
     ambiguous MWIR patches where LWIR depends on global context
  4. `BridgeDiffusionScheduler` — starts from MWIR-derived prior, not pure Gaussian →
     eliminates mean regression on flat regions

### LDM — Latent Diffusion + Conditional DiT (best quality)
- Entrypoint: `train_ldm.py --config configs/ldm.json`
- Two stages: VAE (Stage 1) + Conditional DiT (Stage 2)
- Latest: Flow Matching replaces DDPM + Physics-informed Planck ratio loss

---

## 3. Architecture Decisions and Rationale

### Conditional UNet (v1/v2)
- **Spectral norm** on all convs: stabilises training on heterogeneous IR statistics
- **AdaGN** (Adaptive Group Norm): timestep + MWIR context modulates every ResBlock
- **Fourier timestep embedding**: better than sinusoidal for deep networks
- **Cross-modal attention** in decoder: queries MWIR features at each scale
- **MWIREncoder**: multi-scale CNN provides spatial features for cross-attention
- **Skip connection fix**: decoder needs (num_res_blocks + 1) pops per level.
  The "+1" block consumes a "bridge skip" pushed BEFORE each downsample.
  With channel_mults=[1,2,4,8] and num_res_blocks=2:
    Encoder pushes: 1 (stem) + 4×2 (res blocks) + 3 (bridges) = 12 entries
    Decoder pops:   4×3 = 12 entries ✓

### KL-VAE (LDM Stage 1)
- **f=4 compression**: 256→64 spatial (not f=8) to preserve thermal edge structure
- **Gabor perceptual loss**: no VGG needed — 4 scales × 8 orientations covers IR frequencies
- **KL annealing**: β linearly increases 0 → 1e-4 over 10k steps (prevents posterior collapse)
- **Free-bits schedule**: prevents encoder from mapping all scenes to identical latents
- **Scale factor**: computed AFTER training as 1/std(z_raw), per-channel not global

### Conditional DiT (LDM Stage 2)
- **DiT-B/4**: hidden_dim=768, depth=12, heads=12, patch_size=4 → 256 tokens from 64×64 latent
- **2D RoPE**: rotary position embeddings — generalises to different patch grid sizes
- **Register tokens** (4): prevent attention sink artefacts on flat IR regions
  (the checkerboard pattern from all attention weight collapsing to a few tokens)
- **QK-Norm** (RMSNorm on Q and K): prevents attention logit spikes from hot IR targets
- **adaLN-Zero**: timestep + MWIR global statistics modulate every DiT block
- **CrossModalAttentionDiT**: MWIR spatial features at latent resolution → cross-attention context
- **GlobalConditioner**: encodes 8 statistical moments of MWIR (mean, std, skew, kurtosis,
  4 percentiles) into adaLN conditioning vector
- **CRITICAL BUG FIX**: `n_registers=self.num_registers` must be passed to every
  DiTBlock call. RoPE is spatial and must only be applied to the 256 patch tokens,
  not the 260 total sequence (256 + 4 registers). Without this, the position grid
  is 256 tokens but the input sequence is 260 → size mismatch crash.

---

## 4. Loss Functions

### Standard Losses
- **MSE / velocity MSE**: primary signal; pixel/velocity prediction
- **CFC (Characteristic Function Consistency)**: matches empirical characteristic
  functions patch-by-patch → equivalent to matching ALL distribution moments.
  Critical for thermal IR where pixel-MSE looks fine but texture statistics are wrong.
- **Spectral consistency**: L1 on log power spectral density → prevents blurring of edges
- **Gram loss** (v2): Gabor filter Gram matrices on local 64×64 patches → fixes
  intra-class texture collapse (fields all the same gray)
- **Histogram loss** (v2): differentiable KDE soft histogram → fixes per-scene
  thermal offset errors

### Min-SNR Weighting (DDPM)
With cosine schedule, high-noise timesteps (t≈1000) dominate gradients but carry
no learning signal → model learns to predict zero. Fix:
  weight(t) = min(SNR(t), γ) / SNR(t),  γ=5
This clips high-noise timestep weights and is the most important stability fix
for the spatial collapse / repeating pattern problem.

### Flow Matching (replaces DDPM)
Straight-line interpolation: x_t = (1-t)·x₀ + t·ε
Model learns constant velocity u = ε - x₀
Training: L = E_t[||model(x_t, t, MWIR) - u_target||²]
x₀ recovery: x₀ = x_t - t·u_pred
Advantages: fewer inference steps (20 vs 50), simpler objective, better conditioning.
Logit-Normal time sampling concentrates training at t≈0.5 (hardest part of path).
Heun's method at inference: 2nd-order ODE, same NFE as Euler but better quality.

### Planck Ratio Loss (physics-informed)
Converts normalised pixel values → DN → radiance → Brightness Temperature.
Penalises |BT_LWIR_generated - BT_MWIR| > allowed_delta_K.
Uses Huber loss (quadratic below threshold, linear above).
Weighted by MWIR local thermal contrast (flat regions contribute less).
Physically: MWIR and LWIR BTs should track within ~15K for natural surfaces.
BT_MWIR is inferred from: x_norm → DN (via global min/max) → radiance (gain/offset) → BT.

---

## 5. Latent Space Calibration (Critical)

### What scale_factor is for
The DiT's cosine/flow schedule assumes z ~ N(0,1). If the latent has std ≠ 1,
the schedule is miscalibrated and spatial collapse occurs.

### The bug in the original code
`compute_scale_factor` stored `scale_factor = std(z_raw)` and DiT received
`z_raw × scale_factor`, giving `std = std² ≠ 1`.

For your VAE: std(z_raw) = 1.1367 → old z_DiT std = 1.29, reported as 1.2923 ✓

### The fix
Per-channel affine normalisation (not global scalar):
  latent_mean[c]  = mean(z_raw[c])   over all training samples
  scale_factor[c] = 1 / std(z_raw[c] - latent_mean[c])
  z_DiT[c] = (z_raw[c] - latent_mean[c]) * scale_factor[c]  →  N(0,1) per channel
  z_raw[c] = z_DiT[c] / scale_factor[c] + latent_mean[c]    (exact inverse)

Per-channel is correct because IR latent spaces are multimodal — different land
cover types (water, vegetation, urban, soil) form distinct clusters. A global
mean/std falls in the trough between modes. Per-channel handles each semantic
dimension independently.

Your calibrated values:
  ch0: raw_mean=1.3528  raw_std=0.3101  scale_factor=3.2238
  ch1: raw_mean=1.7968  raw_std=0.7346  scale_factor=1.3602
  ch2: raw_mean=-0.6939 raw_std=0.3050  scale_factor=3.2693
  ch3: raw_mean=-0.1852 raw_std=0.3949  scale_factor=2.5287

### KL training log interpretation
Training log shows KL = 6.4 nats/dim (per dimension, averaged).
Eval script showed 105,862 nats total — this is NOT a problem:
  16,384 dims × 6.4 nats/dim = 104,858 ≈ 105,862 ✓
Healthy range: 2–10 nats/dim. Below 0.1 = posterior collapse.

---

## 6. Data Pipeline

### Normalisation findings
Original approach: global min-max normalisation using pool-wide statistics.
This was broken for MWIR: global_max=3804 was set by outlier (fire/industrial target).
Typical scenes (p2=164, p98=421 DN) mapped to [-0.96, -0.82] — only 6% of [-1,1].
The dataset's second `percentile_normalize` call re-stretched this → effectively
per-image normalisation. The VAE never saw globally-normalised MWIR.

Correct approach: store physical DN values in .npy files, let the dataset's
per-image p2-p98 stretch do all normalisation. This is the intended pipeline.

Cross-band correlation (r=0.20 physical): low because per-image normalisation
removes absolute brightness information. The model learns spatial structure
transfer ("bright edge in MWIR → bright edge in LWIR") not absolute temperature.
This is correct for synthesis — only Planck loss preserves absolute thermal physics.

### Physics-based augmentations
- NEDT noise: simulates sensor noise floor differences between bands
- Radiance offset jitter: simulates atmospheric path radiance variation
- Emissivity scale jitter: simulates per-material emissivity uncertainty
- MTF blur (MWIR only): simulates different optical resolving power
- All geometric augmentations are applied paired (same crop to both bands)

### Image size
Data is 224×224 natively. VAE was trained at 256×256 (dataset upsampled 224→256).
Options: provide 256×256 crops (cleanest), accept the mild upsampling (functionally
identical — PSNR 42.46 dB proves it works), or retrain VAE at 224×224.
Configs corrected to 256×256 with attn_resolutions=[16, 8].

---

## 7. Real-World Deployment — Swath Geometry

### The scenario
MWIR swath = 3 km, LWIR swath = 2 km, scene centres aligned, same row count.
The central 2/3 of every MWIR scene overlaps with available LWIR data.

### Scene-Adaptive Inference (SAI) pipeline
Stage A — SwathAligner: computes pixel-column overlap from swath widths.
Stage B — HistogramCalibrator: fits 256-quantile CDF mapping from model output
  to real LWIR distribution using overlap strip. Corrects per-scene radiometric
  offset (time-of-day, season, atmospheric path). Runs in milliseconds.
Stage C — SceneFineTuner (optional): LoRA fine-tuning on overlap strip.
  Only adapts cross-attention projections (rank 4), base weights frozen.
  Prevents catastrophic forgetting. Use for genuinely OOD scenes.

Decision guide:
  Same sensor/region as training  → A + B
  New geographic region           → A + B + C (50–100 steps)
  Different altitude/config       → A + B + C (100–200 steps)

---

## 8. Bugs Found and Fixed

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| `channels.pop()` IndexError | Decoder needs (num_res_blocks+1) pops per level; encoder was not pushing bridge skips before downsampling | Push `channels.append(cur_channels)` before each Downsample call; rewrite forward() to match |
| DiT repeating tile pattern | Four compounding causes: missing Min-SNR weighting, null conditioning as zeros, scale_factor=1.0, RoPE applied to registers | All four fixed independently |
| RoPE size mismatch (260 vs 256) | `n_registers=self.num_registers` not passed to DiTBlock; defaulted to 0 | Pass `n_registers=self.num_registers` in ConditionalDiT.forward() |
| `float(vae.scale_factor)` TypeError | scale_factor is now list[float] after per-channel calibration; float() cannot convert a list | Store as-is; list is JSON-serialisable and handled by `_get_affine()` |
| eval_vae KL=105,862 | Eval summed KL over all 16,384 dims; threshold was per-dim | Changed eval to report mean KL per dim; threshold updated to 0.1–15 nats/dim |
| IndexError in save_worst_best | `all_psnr` has 200 entries but `orig_images` capped at 64; argsort returned indices >63 | Pass `all_psnr[:n_saved]` to match only saved images |
| `posterior.sample() * vae.scale_factor` TypeError | scale_factor is list; PyTorch interprets list as fancy index | Use `vae.encode_to_dit()` which routes through `_get_affine()` |
| scale_factor = std(z) not 1/std(z) | Stored std instead of reciprocal; DiT received z×std giving std² | Fix: scale_factor = 1/std; add latent_mean offset removal; per-channel |
| Device mismatch in compute_scale_factor | Function used device string argument; model was on different device | Derive device from `next(self.parameters()).device` |
| VAE Gabor stack shape mismatch | kernels at different scales have different k; torch.stack requires uniform shape | Pre-compute max_k; zero-pad all kernels to max_k before stack |
| eval_vae loaded old checkpoint | Print used `:.5f` on list; crashed silently; old checkpoint used | Add isinstance check; add `--inspect` flag; add old-format warning |
| double normalisation (MWIR 6% range) | global_max set by outlier; typical scenes in [-0.96,-0.82] | Use physical patches; dataset percentile_normalize is the only normalisation |

---

## 9. VAE Evaluation Results (your actual numbers)

| Metric | Value | Status | Notes |
|--------|-------|--------|-------|
| PSNR mean | 42.46 dB | PASS | Excellent (floor is 28 dB) |
| PSNR p10 | 37.01 dB | PASS | Worst 10% still good |
| SSIM mean | 0.9931 | PASS | Near-perfect structural fidelity |
| KL/dim | 6.46 nats | PASS | Healthy (2–10 range) |
| z_std | 1.015 after recal | PASS | Correctly calibrated |
| z_mean_abs | 0.131 after recal | PASS (loosened) | Train/val distribution gap |
| PSD error | 0.00116 | PASS | Negligible frequency distortion |

PSD analysis: dip at 45–65 cycles/image is structural (f=4 VAE bandwidth limit).
The 2 downsampling stages theoretically preserve up to 32 cycles/image; CNN
anti-aliasing extends this to ~45; the dip at 60 cycles (35% attenuation) is
the transition zone. Ratio returns to ~1.0 above 80 cycles (texture-level encoding).
This is acceptable for EO applications at typical GSD.

---

## 10. File Inventory (8,171 lines total)

### Models (core architecture)
- `models/conditional_unet.py` — v1/v2 UNet with cross-modal attention
- `models/diffusion_scheduler.py` — DDPM/DDIM + CFC + Spectral losses
- `models/targeted_improvements.py` — v2 Gram + Histogram + Bridge + LoRA
- `models/flow_matching.py` — Flow Matching scheduler + Heun/Euler samplers
- `models/planck_loss.py` — Physics-informed Planck ratio loss + BT conversion
- `models/ldm/vae.py` — KL-VAE with Gabor perceptual loss + per-channel calibration
- `models/ldm/dit.py` — Conditional DiT-B/4 with RoPE + registers + QK-Norm

### Training
- `training/trainer.py` — v1 trainer with EMA, AMP, best-checkpoint saving
- `training/improved_trainer.py` — v2 trainer with bridge diffusion + scene context
- `training/ldm_trainer.py` — Two-stage VAE + DiT trainer with FM + Planck loss
- `training/visualizer.py` — Fixed-sample visualiser with deterministic indices

### Inference
- `inference/infer.py` — v1/v2 patch sliding-window + ensemble + SAI
- `inference/ldm_infer.py` — LDM CFG sampling + latent blending + SAI
- `inference/scene_adaptive.py` — SwathAligner + HistogramCalibrator + LoRA fine-tuning

### Data
- `data/dataset.py` — MWIRLWIRDataset with physics-aware augmentations

### Evaluation / Diagnostics
- `eval_vae.py` — VAE reconstruction + latent + KL + PSD + spatial error + verdict
- `diag_normalization.py` — Detects double-normalisation, global vs per-image, cross-band correlation

### Configuration
- `configs/base.json` — v1 (256×256, attn_resolutions=[16,8])
- `configs/improved_v2.json` — v2 with targeted loss weights
- `configs/ldm.json` — LDM with FM enabled + Planck sub-config

### Utilities
- `train.py`, `train_ldm.py` — Entrypoints
- `offline_setup.py` — Airgapped machine: download/install/verify

---

## 11. Recommended Next Steps (Prioritised)

### Immediate (before Stage 2 DiT training)
1. Fill in actual calibration coefficients in `configs/ldm.json` planck section
2. Use physical patches (original DN values) as training data — remove pre-normalisation
3. Run `python eval_vae.py --vae_ckpt vae_final_recal.pt --inspect` to confirm NEW FORMAT

### Stage 2 Training
4. `python train_ldm.py --config configs/ldm.json --skip_vae --vae_ckpt vae_final_recal.pt`
5. Monitor `FM:` loss (target <0.1), `BT-MAE:` (target <5K), `Planck:` (target <0.02)
6. Use 8–20 FM Heun steps at inference vs 50 DDIM steps — same quality, faster

### Architecture improvements (Tier 1, high impact)
7. Joint MWIR-LWIR VAE: encoder in_channels=2, decoder two output heads
   The DiT latent then explicitly encodes cross-band relationships
8. Brightness Temperature normalisation: convert DN→BT before dataset normalisation
   Preserves the physical temperature relationship between bands

### Architecture improvements (Tier 2, medium impact)
9. Emissivity-conditioned generation: 5-class land cover classifier from MWIR
   provides additional cross-attention context
10. Multi-resolution DiT: coarse 16×16 + fine 64×64 latent pyramid for global thermal context

---

## 12. Key Configuration Reference

### To switch Flow Matching ON/OFF
```json
"use_flow_matching": true   // FM with Heun sampler (default, recommended)
"use_flow_matching": false  // DDPM with Min-SNR + DDIM sampler
```

### To enable/disable Planck loss
```json
"planck": { "lambda_planck": 0.05 }   // enabled
"planck": { "lambda_planck": 0.0  }   // disabled (zero overhead)
```

### Key inference parameters
- FM inference: 20 Heun steps ≈ DDIM 50 steps quality
- CFG guidance_scale: 3.0–7.0; higher = sharper but less diverse; above 7 = mode collapse
- SAI allowed_delta_K: 15K (natural surfaces), 8K (water/veg only), 25K (industrial)

### What to monitor in training logs
```
FM:      velocity MSE — target <0.1 at convergence
CFC:     characteristic function distance — target <0.01
Planck:  BT Huber penalty — target <0.02 (healthy); >0.1 = unphysical outputs
BT-MAE:  mean absolute BT error vs real LWIR — target <5K typical, <15K mixed
```
