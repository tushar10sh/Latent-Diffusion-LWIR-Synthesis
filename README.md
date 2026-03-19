# MWIR → LWIR Synthesis Pipeline

Conditional diffusion model pipeline for synthesising Long-Wave Infrared (LWIR, 8–12 µm) imagery from Mid-Wave Infrared (MWIR, 3–5 µm) spaceborne EO observations. Three progressively more capable model variants are provided, along with physics-informed training losses, scene-adaptive inference for real-world swath-mismatched deployment, and comprehensive VAE evaluation tooling.

---

## Table of Contents

1. [Background and Physics](#background-and-physics)
2. [Pipeline Variants](#pipeline-variants)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Data Preparation](#data-preparation)
6. [Training](#training)
7. [VAE Evaluation and Recalibration](#vae-evaluation-and-recalibration)
8. [Inference](#inference)
9. [Scene-Adaptive Inference](#scene-adaptive-inference)
10. [Visualisation](#visualisation)
11. [Hyperparameter Tuning](#hyperparameter-tuning)
12. [Important Fixes and Changes](#important-fixes-and-changes)
13. [Metrics](#metrics)
14. [Offline / Airgapped Setup](#offline--airgapped-setup)
15. [References](#references)

---

## Background and Physics

### Spectral physics
MWIR (3–5 µm) and LWIR (8–12 µm) are governed by different physical mechanisms. For a surface at temperature T, the spectral radiance at wavelength λ follows the Planck function:

```
B(λ, T) = c₁ / [λ⁵ · (exp(c₂ / λT) - 1)]
  c₁ = 1.19104 × 10⁸  W·µm⁴/(m²·sr)
  c₂ = 14387.8          µm·K
```

MWIR contains both thermal emission and solar-reflected components (high dynamic range, strong material contrast). LWIR is purely thermal emission (lower contrast, dominated by surface temperature and emissivity). For natural surfaces, Brightness Temperatures in the two bands should agree within ≈ 10–20 K; larger deviations indicate emissivity anomalies or synthesis errors.

### Why naive models fail
1. **Thermal homogenisation**: agricultural fields look uniform in MWIR but have distinct LWIR temperatures → pixel-MSE regresses to scene mean
2. **Emissivity brightness errors**: roads bright in MWIR (solar reflection) but at thermal temperature in LWIR → model copies MWIR contrast incorrectly
3. **Mean regression on low-contrast zones**: water bodies, vegetation are nearly uniform in MWIR → model outputs average grey

### How this pipeline addresses them
- CFC loss — matches full patch distribution (all moments), not just pixel means
- Spectral consistency — preserves spatial frequency content at structural edges
- Cross-modal attention — queries MWIR at every decoder/DiT scale
- Local Texture Gram loss — penalises texture collapse in homogeneous scenes
- Scene Histogram loss — corrects per-scene thermal offset/gain bias
- Bridge diffusion — seeds forward process from MWIR-derived prior, cuts mean regression
- Flow Matching — straight-line trajectories, fewer inference steps, simpler objective
- Planck ratio loss — physics-informed penalty for physically impossible MWIR/LWIR combinations

---

## Pipeline Variants

| Variant | Config | Entrypoint | Best for |
|---|---|---|---|
| **v1 — Baseline DDPM** | `configs/base.json` | `train.py` | Rapid prototyping, ablation |
| **v2 — Improved pixel-space** | `configs/improved_v2.json` | `train.py` | 500–2000 pairs, solid baseline |
| **LDM — Latent DiT + Flow Matching** | `configs/ldm.json` | `train_ldm.py` | Best quality, A100-class hardware |

All three share the same data pipeline, visualiser, diagnostics, and scene-adaptive inference layer.

---

## Project Structure

```
mwir2lwir/
│
├── train.py                        # v1 / v2 training entrypoint
├── train_ldm.py                    # LDM two-stage training entrypoint
├── offline_setup.py                # Airgapped machine: download / install / verify
├── diag_normalization.py           # Detects double-normalisation, cross-band correlation
├── eval_vae.py                     # VAE reconstruction + latent + KL + PSD evaluation
├── requirements.txt
│
├── configs/
│   ├── base.json                   # v1 hyperparameters (256×256, attn_resolutions=[16,8])
│   ├── improved_v2.json            # v2 targeted-improvement hyperparameters
│   └── ldm.json                    # LDM: VAE + DiT + Flow Matching + Planck loss config
│
├── data/
│   ├── dataset.py                  # MWIRLWIRDataset + physics-aware augmentations
│   └── ir_pairs/                   # Raw sensor DN files (not committed)
│       ├── mwir/  scene_001.npy …
│       └── lwir/  scene_001.npy …
│
├── models/
│   ├── conditional_unet.py         # v1/v2: UNet with cross-modal attention, AdaGN
│   ├── diffusion_scheduler.py      # DDPM/DDIM scheduler + CFC + Spectral losses
│   ├── targeted_improvements.py    # v2: Gram, Histogram, Bridge scheduler, LoRA
│   ├── flow_matching.py            # Rectified Flow / Flow Matching + Heun/Euler ODE sampler
│   ├── planck_loss.py              # Physics-informed Planck ratio loss + BT conversion
│   └── ldm/
│       ├── vae.py                  # KL-VAE (f=4): Gabor perceptual, per-channel calibration
│       └── dit.py                  # Conditional DiT-B/4: RoPE, register tokens, QK-Norm
│
├── training/
│   ├── trainer.py                  # v1: EMA, AMP, cosine LR, best-checkpoint saving
│   ├── improved_trainer.py         # v2: bridge diffusion, scene context encoder
│   ├── ldm_trainer.py              # LDM: VAETrainer (stage 1) + DiTTrainer (stage 2, FM)
│   └── visualizer.py               # Fixed-sample visualiser (deterministic across restarts)
│
├── inference/
│   ├── infer.py                    # v1/v2: patch sliding-window, ensemble, SAI integration
│   ├── ldm_infer.py                # LDM: CFG + FM sampling, latent blending, SAI integration
│   └── scene_adaptive.py           # SwathAligner, HistogramCalibrator, LoRA fine-tuning
│
├── pretrained_weights/             # Downloaded by offline_setup.py (not committed)
│   └── sd_vae/
│       └── diffusion_pytorch_model.safetensors
│
└── runs/                           # Generated at runtime (not committed)
    ├── mwir2lwir/
    ├── mwir2lwir_v2/
    └── mwir2lwir_ldm/
        ├── stage1_vae/
        │   ├── vae_final.pt        # raw checkpoint (scale_factor=1.0 — NOT ready for DiT)
        │   └── vae_final_recal.pt  # recalibrated checkpoint (use this for Stage 2)
        └── stage2_dit/
```

Each run directory contains:

```
runs/<run>/
├── config.json
├── train_log.jsonl                 # one JSON line per log interval
├── checkpoints/
│   ├── ckpt_step_NNNNNNN.pt       # periodic (last 3 kept, older auto-pruned)
│   ├── ckpt_best.pt               # best test-PSNR (never pruned)
│   └── ckpt_final.pt
├── train-results/step_NNNNNNN/
│   ├── overview.png               # MWIR | Generated LWIR | Real LWIR side-by-side grid
│   ├── sample_NN_<id>_{mwir,gen,real}.{npy,png}
│   └── psnr.json
└── test-results/step_NNNNNNN/     # same structure, separate fixed sample set
```

---

## Installation

```bash
pip install -r requirements.txt
```

Required packages: `torch>=2.1`, `torchvision`, `einops`, `numpy`, `pillow`, `scipy`, `rasterio`, `tqdm`, `safetensors`.

For airgapped machines see [Offline / Airgapped Setup](#offline--airgapped-setup).

---

## Data Preparation

### File format
Store **raw physical sensor values** (DN, radiance, or uncalibrated counts) in `.npy` files — do not pre-normalise. The dataset applies a robust p2–p98 percentile stretch per image internally, mapping each scene independently to `[-1, 1]`. Pre-normalising with a global min/max before storing will be silently overridden by this internal normalisation, causing a confusing double-normalisation effect.

```
data/ir_pairs/
  MWIR/
    MWIR-patch-0001.npy    # float32, shape (H, W) or (1, H, W), raw DN values
    MWIR-patch-0002.npy
  LWIR/
    LWIR-patch-0001.npy    # paired by filename stem
    LWIR-patch-0002.npy
```

Supported formats: `.npy`, `.npz`, `.tif`/`.tiff` (requires `rasterio`), `.png`. Files are paired by filename stem. The train/val/test split is deterministic by filename sort order.

### Checking your normalisation
If you are unsure whether your data is correctly formatted, run the diagnostic script:

```bash
python diag_normalization.py \
  --data_root data/ir_pairs \
  --n_samples 200
```

This script checks: (1) whether raw files are already in `[-1, 1]` (indicating pre-normalisation), (2) the magnitude of any double-normalisation effect, (3) cross-band mean correlation, and (4) whether global normalisation has collapsed per-scene dynamic range. Key indicator: if MWIR raw values are clustered in a narrow sub-range like `[-0.96, -0.82]` (6% of dynamic range), your global normalisation was dominated by outliers and the dataset is silently correcting it.

### Image size
The pipeline is configured for **256×256** images. If your native patches are 224×224, either extract 256×256 crops from source imagery (cleanest) or accept the mild bilinear upsampling that `dataset.py` applies automatically (PSNR impact is negligible — validated at 42.46 dB PSNR).

---

## Training

### v1 — Baseline DDPM

```bash
python train.py --config configs/base.json
```

### v2 — Improved pixel-space

```bash
python train.py --config configs/improved_v2.json
```

v2 adds four targeted fixes addressing specific observed failure modes:

- `LocalTextureGramLoss` — Gabor filter Gram matrices on 64×64 local patches → fixes flat fields
- `SceneHistogramLoss` — differentiable KDE histogram matching → fixes road brightness bias
- `GlobalSceneContextEncoder` — scene-level stats in AdaGN conditioning → resolves thermally ambiguous MWIR patches
- `BridgeDiffusionScheduler` — starts from MWIR-derived prior, not pure Gaussian → eliminates mean regression

### LDM — Latent DiT with Flow Matching (recommended)

**Stage 1: Fine-tune KL-VAE on LWIR only**

```bash
python train_ldm.py --config configs/ldm.json --stage1_only
```

**Both stages together:**

```bash
python train_ldm.py --config configs/ldm.json
```

**Skip Stage 1 (VAE already trained and recalibrated):**

```bash
python train_ldm.py --config configs/ldm.json \
  --skip_vae \
  --vae_ckpt runs/mwir2lwir_ldm/stage1_vae/vae_final_recal.pt
```

The LDM uses **Flow Matching** (Rectified Flow) instead of DDPM by default. This replaces the curved diffusion trajectory with a straight-line interpolation:

```
x_t = (1 - t) · x₀  +  t · ε,    t ∈ [0, 1]
u_target = ε - x₀                 (constant velocity — what the DiT learns)
```

Benefits over DDPM: 20 Heun steps match DDIM 50 steps in quality; no noise schedule to tune; simpler objective; Logit-Normal time sampling concentrates training at the hardest mid-trajectory region.

To revert to DDPM, set `"use_flow_matching": false` in `configs/ldm.json`.

### Resuming from checkpoint

```bash
python train.py --config configs/improved_v2.json \
  --resume runs/mwir2lwir_v2/checkpoints/ckpt_step_0050000.pt
```

### Training log metrics explained

```
Step    500 | FM: 0.0821 | CFC: 0.0041 | Planck: 0.0026 | LR: 1.00e-04 | 12.3 it/s
```

| Key | Meaning | Healthy range |
|---|---|---|
| `FM` | Flow matching velocity MSE (or `ddpm_mse` if DDPM) | Decreases from ~0.5 → <0.1 |
| `CFC` | Characteristic function consistency | <0.01 at convergence |
| `Planck` | Huber BT-difference penalty | <0.02 healthy; >0.1 = unphysical outputs |
| `recon` (VAE) | L1 pixel reconstruction | Decreases from ~0.3 → <0.05 |
| `kl` (VAE) | KL divergence per latent dimension | 2–10 nats/dim healthy |
| `kl_weight` (VAE) | Current KL annealing β | Linearly 0 → 1e-4 over 10k steps |

Note: the training log reports KL averaged across all latent dimensions. For a 256×256 image with f=4 VAE and z_channels=4, there are 16,384 latent dimensions. A KL of 6.4 nats/dim corresponds to 105,862 total nats — both numbers refer to the same healthy state.

---

## VAE Evaluation and Recalibration

### Why evaluation matters
A poorly trained or miscalibrated VAE is the most dangerous silent failure in the LDM pipeline. The DiT cannot recover information the VAE discarded, and bad latent calibration causes spatial collapse (repeating tile patterns) that looks like a DiT problem.

### Step 1: Evaluate the VAE

```bash
python eval_vae.py \
  --vae_ckpt runs/mwir2lwir_ldm/stage1_vae/vae_final.pt \
  --data_root data/ir_pairs \
  --output_dir eval/vae \
  --split val \
  --n_samples 200
```

This produces:
- `reconstruction_grid.png` — original vs reconstructed side-by-side (8 samples)
- `latent_distribution.png` — per-channel histogram of z_DiT vs N(0,1) reference
- `kl_distribution.png` — KL divergence distribution across samples
- `psd_comparison.png` — radially-averaged PSD: original vs reconstructed; ratio plot
- `spatial_error_map.png` — mean absolute error at each pixel position
- `worst/best_reconstructions.png` — 4 hardest and 4 easiest scenes
- `psnr_ssim_scatter.png` — per-sample quality scatter
- `metrics.json` — all metrics + go/no-go verdict

**Verdict thresholds:**

| Metric | Target | Notes |
|---|---|---|
| PSNR mean | ≥ 28 dB | 42+ dB is excellent |
| PSNR p10 | ≥ 24 dB | Worst-10% floor |
| SSIM | ≥ 0.85 | |
| z_std | 0.85–1.20 | After recalibration |
| z_mean_abs | < 0.20 | Train/val gap is normal |
| KL/dim | 0.1–15 nats/dim | 2–10 is ideal |
| PSD error | < 0.15 | Frequency fidelity |

### Step 2: Inspect checkpoint format

Before running Stage 2, verify the checkpoint contains per-channel calibration:

```bash
python eval_vae.py --vae_ckpt runs/mwir2lwir_ldm/stage1_vae/vae_final.pt --inspect
```

Expected output for a correctly recalibrated checkpoint:
```
✓ NEW FORMAT: per-channel calibration (len=4)
  scale_factor per channel: [3.2238, 1.3602, 3.2693, 2.5287]
  latent_mean  per channel: [1.3528, 1.7968, -0.6939, -0.1852]
  → Safe to use for Stage 2 DiT training.
```

If you see `✗ OLD FORMAT`, run Step 3.

### Step 3: Recalibrate the VAE

The following script recomputes per-channel `latent_mean` and `scale_factor` and saves a new checkpoint. The VAE weights are **not changed** — only the calibration values are updated.

```bash
python recalibrate.py
python eval_vae.py --vae_ckpt vae_final_recal.pt --inspect
```

### Understanding the latent distribution plot

The per-channel histogram shows each of the 4 latent channels after calibration. A multimodal (bimodal or trimodal) shape is **expected and correct** — different land cover types (water, vegetation, urban, bare soil) form distinct clusters in latent space. After per-channel calibration each channel should show mean ≈ 0 and std ≈ 1 while preserving the multimodal shape. The DiT learns the multimodal data distribution; the calibration only ensures the noise schedule is correctly scaled.

### Understanding the PSD plot

The radially-averaged PSD plot shows spatial frequency fidelity. The x-axis is in cycles/image: `k` cycles/image means `k` complete oscillations across the full 256-pixel width, i.e. features of size `256/k` pixels.

A dip around 45–65 cycles/image (features of 4–6 pixels) is structural — it is the VAE's bandwidth limit from f=4 compression. The 2 downsampling stages preserve frequencies up to ~32 cycles/image exactly; the CNN anti-aliasing extends this to ~45; above that there is a transition zone with ~35% attenuation. This is acceptable for EO applications. A PSD error below 0.15 passes the quality threshold.

---

## Inference

### v1 / v2 — Standard inference

```bash
python inference/infer.py \
  --checkpoint runs/mwir2lwir/checkpoints/ckpt_best.pt \
  --mwir path/to/scene_mwir.npy \
  --output outputs/scene_lwir.npy \
  --num_steps 50 \
  --patch_size 256 \
  --overlap 0.25
```

With ground-truth LWIR for metrics:

```bash
python inference/infer.py \
  --checkpoint runs/mwir2lwir/checkpoints/ckpt_best.pt \
  --mwir scene_mwir.npy \
  --lwir scene_lwir_gt.npy \
  --output outputs/scene_lwir.npy
```

### LDM inference with Flow Matching

```bash
python inference/ldm_infer.py \
  --vae_ckpt runs/mwir2lwir_ldm/stage1_vae/vae_final_recal.pt \
  --dit_ckpt runs/mwir2lwir_ldm/stage2_dit/checkpoints/dit_best.pt \
  --mwir scene_mwir.npy \
  --output outputs/scene_lwir.npy \
  --guidance_scale 5.0 \
  --num_steps 20
```

With Flow Matching, **20 steps** (Heun ODE) gives quality equivalent to DDIM 50 steps. The guidance scale (3.0–7.0) is the primary quality knob at inference. Above 7.0 causes mode collapse artefacts.

---

## Scene-Adaptive Inference

For real-world deployment where MWIR and LWIR sensors have different swath widths:

```
MWIR swath: 3 km  (full scene width)
LWIR swath: 2 km  (central strip, scene centres aligned, same row count)
```

**Stage A — Swath alignment** (always, microseconds): crops MWIR to the overlapping columns using swath widths and validates dimensions.

**Stage B — Histogram calibration** (default on, milliseconds): fits 256-quantile CDF mapping from model output to real LWIR. Corrects per-scene radiometric offset from time-of-day, season, atmospheric path.

**Stage C — LoRA scene fine-tuning** (optional, 1–5 min): gradient steps on the overlap strip, only adapting rank-4 cross-attention adapters. Base weights frozen.

| Scenario | Recommended stages |
|---|---|
| Same sensor, same region | A + B |
| Same sensor, new region | A + B + C (50–100 steps) |
| Different altitude / configuration | A + B + C (100–200 steps) |

```bash
# Standard deployment with histogram calibration
python inference/infer.py \
  --checkpoint runs/mwir2lwir/checkpoints/ckpt_best.pt \
  --mwir scene_mwir.npy \
  --lwir scene_lwir_overlap.npy \
  --mwir_swath_km 3.0 --lwir_swath_km 2.0 \
  --scene_id orbit_042_20240315 \
  --output outputs/orbit_042/lwir_full.npy

# OOD scene with LoRA fine-tuning
python inference/infer.py \
  --checkpoint runs/mwir2lwir/checkpoints/ckpt_best.pt \
  --mwir scene_mwir.npy \
  --lwir scene_lwir_overlap.npy \
  --mwir_swath_km 3.0 --lwir_swath_km 2.0 \
  --scene_finetune --finetune_steps 100 \
  --scene_id orbit_042_20240315 \
  --output outputs/orbit_042/lwir_full.npy
```

Per-scene outputs saved to `outputs/<scene_id>/`: full-swath `.npy`/`.png`, overlap comparison, histogram calibration `.npz`, and `metrics.json`.

---

## Visualisation

The `Visualizer` selects N fixed samples at init using deterministic even-spaced indices and a fixed seed. The same samples are shown at every visualisation step, including after checkpoint restarts, making quality progression directly comparable across training runs.

```json
"vis_n_samples": 8,
"vis_seed":      42,
"vis_every":     2000
```

Outputs per step: `overview.png` (MWIR | Generated | Real grid), individual `.npy`/`.png` per sample, `psnr.json`. PNG normalisation uses p2–p98 stretch — robust to hot targets and cold shadows in IR imagery.

---

## Hyperparameter Tuning

### Diagnosing LDM failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Repeating tile / checkerboard | Scale_factor=1.0 (uncalibrated VAE) | Run recalibration; use `vae_final_recal.pt` |
| Repeating tile even after recal | Min-SNR missing or RoPE bug | Confirm `n_registers` passed to DiTBlock; confirm `use_flow_matching: true` |
| Spatially uniform output | Null conditioning as zeros | Fixed — now uses per-scene MWIR spatial mean |
| Homogeneous fields look flat | Mean regression, CFC too weak | Increase `lambda_cfc` 0.1 → 0.3 |
| Road brightness wrong | Emissivity mismatch | Enable Planck loss; increase `lambda_hist` |
| Edges blurred at structures | Spectral loss too weak | Increase `lambda_spectral` 0.05 → 0.15 |
| Training instability early | Gram loss spikes | Reduce `lambda_gram` 0.05 → 0.01 first |
| OOD scene at inference | Distribution shift | Use SAI with `--scene_finetune` |

### Flow Matching tuning

| Parameter | Default | Effect |
|---|---|---|
| `fm_time_sampling` | `logit_normal` | `uniform` = flat; `logit_normal` = concentrate on hard mid-trajectory |
| `fm_logit_scale` | 1.0 | Higher = more concentration near t=0.5 |
| `fm_loss_weighting` | `constant` | `snr` = down-weight high-noise timesteps (rarely needed with FM) |
| `num_steps` at inference | 20 | 8 steps minimum; 20 recommended; 50 = diminishing returns |

### Planck loss configuration

Fill in your sensor's calibration coefficients in `configs/ldm.json` under the `"planck"` key before enabling:

```json
"planck": {
  "lambda_planck":    0.05,
  "mwir_wavelength_um": 4.0,
  "lwir_wavelength_um": 10.0,
  "mwir_gain":        <your gain>,
  "mwir_offset":      <your offset>,
  "lwir_gain":        <your gain>,
  "lwir_offset":      <your offset>,
  "mwir_norm_min":    88.0,
  "mwir_norm_max":    3804.0,
  "lwir_norm_min":    1622.0,
  "lwir_norm_max":    3836.0,
  "allowed_delta_K":  15.0
}
```

`norm_min/max` are the physical DN values corresponding to `[-1, 1]` in your normalised representation — use the global pool min/max from your sensor documentation. Set `lambda_planck: 0.0` to disable with zero overhead.

`allowed_delta_K` tolerance guide: 8K for water/vegetation-only, 15K (default) for mixed land cover, 25K for scenes with industrial targets.

### Architecture scaling

| GPU VRAM | v1 / v2 | LDM |
|---|---|---|
| 8 GB | `base_channels=64, channel_mults=[1,2,4], batch_size=4` | DiT-S/4, `dit_batch_size=4` |
| 16 GB | `base_channels=128, channel_mults=[1,2,4,8], batch_size=8` | DiT-B/4, `dit_batch_size=8` |
| 40 GB | `base_channels=192, num_res_blocks=3, batch_size=16` | DiT-L/4, `dit_batch_size=16` |

| Training pairs | Recommended variant | VAE strategy |
|---|---|---|
| < 500 | v2 with heavy augmentation | Skip LDM |
| 500–2000 | v2 or LDM | Fine-tune from SD KL-VAE |
| 2000–10000 | LDM DiT-B/4 | Fine-tune from SD KL-VAE |
| > 10000 | LDM DiT-L/4 | Train VAE from scratch |

---

## Important Fixes and Changes

This section documents substantive bugs and design changes made during development that are not obvious from the config files alone.

### ConditionalUNet skip connection structure (critical bug fix)
The decoder calls `channels.pop()` exactly `num_levels × (num_res_blocks + 1)` times. The encoder must push exactly that many entries: `num_res_blocks` entries from ResBlocks, plus one **bridge skip** pushed immediately before each Downsample (except the last level which feeds the bottleneck). The original code was short by `num_levels - 1` entries, causing `IndexError: pop from empty list`. The forward method also hardcoded `range(2)` instead of `range(self.num_res_blocks)`. Both `__init__` and `forward` were rewritten to correctly mirror each other.

### DiT RoPE and register token interaction (critical bug fix)
RoPE (Rotary Position Embeddings) is spatial — it must only be applied to the H×W patch tokens, not the register tokens prepended to the sequence. With `num_registers=4` and `patch_size=4`, the sequence has 260 tokens total (256 patches + 4 registers) but the RoPE grid has only 256 positions. The original code passed `n_registers=0` (default) to every DiTBlock, so RoPE tried to rotate all 260 tokens against a 256-position grid → `RuntimeError: size mismatch at dimension 2`. Fix: pass `n_registers=self.num_registers` in `ConditionalDiT.forward()`.

### VAE scale_factor was wrong in both direction and dimensionality
The original `compute_scale_factor` stored `scale_factor = std(z_raw)` (a scalar). The DiT then received `z_raw × std`, giving variance `std²` instead of 1. For `std=1.1367`: DiT z_std = `1.292` (matches reported 1.2923 exactly).

The fix uses **per-channel** normalisation (not global scalar):
```
scale_factor[c] = 1 / std(z_raw[c] - mean(z_raw[c]))
latent_mean[c]  = mean(z_raw[c])
z_DiT[c] = (z_raw[c] - latent_mean[c]) * scale_factor[c]  →  N(0,1)
```
Per-channel is correct because IR latent spaces are multimodal — land cover clusters appear at different positions per channel. A global mean falls in the trough between modes and inflates the measured std. The recalibration does not change VAE weights and does not require retraining.

### Null conditioning must not be zeros
Using `torch.zeros_like(mwir)` as the null conditioning for CFG trains the uncond path to predict noise for an "all-cold scene" rather than a truly neutral prior. This biases generated outputs away from cold-looking inputs and contributed to the spatial collapse pattern. Fixed to use the per-scene MWIR spatial mean as null conditioning, both in training and inference.

### KL divergence log interpretation
The training log reports KL averaged per latent dimension. The eval script originally summed over all 16,384 dimensions and compared against a per-dimension threshold, reporting KL=105,862 as a FAIL. This is not a problem:
```
16,384 dims × 6.4 nats/dim = 104,858 ≈ 105,862 total nats
```
Healthy range: 2–10 nats/dim. The eval script threshold and reporting were corrected.

### Global normalisation was a no-op due to outliers
Pre-normalising MWIR with a global pool min/max where `global_max=3804` (from a single hot outlier) compressed 94% of typical scenes into only 6% of the `[-1, 1]` range. The dataset's internal `percentile_normalize` re-expanded every scene back to `[-1, 1]`, silently overriding the global normalisation. The VAE was effectively trained with per-image normalisation regardless of any pre-processing applied to the stored files. Correct practice: store raw physical DN values and let the dataset normalise.

### VAE forward/encode/decode API change
The `decode()` method now applies the full inverse affine transform internally. DiTTrainer and inference code use `vae.encode_to_dit()` and `vae.decode()` rather than manually computing `posterior.sample() * scale_factor`. The helper `_get_affine()` handles both the old scalar format (backward compatible with old checkpoints) and the new per-channel list format.

### Image size corrected to 256×256
All three configs were temporarily changed to `image_size: 224` during development. The VAE was trained at 256×256 (the dataset upsampled 224→256 internally). All configs have been corrected back to `image_size: 256` with `attn_resolutions: [16, 8]` (the resolutions that actually appear when downsampling 256 four times: 256→128→64→32→16→8).

### Visualiser saves both train and test fixed sets
All three trainers save visualisation images to `train-results/` and `test-results/` separately at every `vis_every` step. The fixed sample indices are deterministic (evenly spaced + seeded shuffle) and identical across checkpoint restarts. The best-PSNR checkpoint (`ckpt_best.pt`) is saved based on the test-set visualisation PSNR, independently of the periodic checkpoint rotation.

---

## Metrics

| Metric | Description | Direction |
|---|---|---|
| PSNR (dB) | Peak Signal-to-Noise Ratio | Higher |
| SSIM | Structural Similarity Index | Higher |
| ERGAS | Relative global synthesis error | Lower |
| CFC Distance | Characteristic function L2 (full distribution match) | Lower |
| BT-MAE (K) | Mean absolute Brightness Temperature error vs real LWIR | Lower |
| PSD error | Log power-spectral-density L1 (frequency fidelity) | Lower |

CFC distance is the most informative metric for thermal IR synthesis — a model can have good PSNR while failing on CFC, indicating correct global brightness but wrong local texture distribution. BT-MAE is the most physically meaningful metric for deployment, directly measuring temperature accuracy in Kelvin.

---

## Offline / Airgapped Setup

The only external weight required is the SD KL-VAE (~335 MB), used only for LDM Stage 1 initialisation.

**On an internet-connected machine:**

```bash
python offline_setup.py --download \
  --cuda_version cu121 \
  --python_version 3.10

# Transfer pretrained_weights/ and offline_wheels/ to airgapped machine
```

**On the airgapped machine:**

```bash
python offline_setup.py --install
python offline_setup.py --patch_config   # updates configs/ldm.json with local VAE path
python offline_setup.py --verify         # confirms all imports and weights present
```

Setting `"sd_vae_path": null` in `configs/ldm.json` disables SD initialisation; increase `vae_total_steps` from 50000 to 100000 in that case. No other components auto-download at runtime — Gabor filter banks are computed analytically, no VGG or CLIP backbone is used.

---

## References

- Ho et al. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS.
- Song et al. (2020). *Denoising Diffusion Implicit Models.* ICLR 2021.
- Nichol & Dhariwal (2021). *Improved Denoising Diffusion Probabilistic Models.* ICML.
- Rombach et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR.
- Peebles & Xie (2023). *Scalable Diffusion Models with Transformers.* ICCV.
- Darcet et al. (2023). *Vision Transformers Need Registers.* ICLR 2024.
- Su et al. (2023). *RoFormer: Enhanced Transformer with Rotary Position Embedding.* Neurocomputing.
- Liu et al. (2022). *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.* ICLR 2023.
- Esser et al. (2024). *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis.* (SD3)
- Hang et al. (2023). *Efficient Diffusion Training via Min-SNR Weighting Strategy.* ICCV.
- Ansari et al. (2020). *Characteristic Function-based Methods for Generative Models.*
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.