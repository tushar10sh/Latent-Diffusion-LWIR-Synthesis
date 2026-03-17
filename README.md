# MWIR → LWIR Synthesis Pipeline

Conditional diffusion model pipeline for synthesising Long-Wave Infrared (LWIR, 8–12 µm) imagery from Mid-Wave Infrared (MWIR, 3–5 µm) spaceborne EO observations. Three progressively more capable model variants are provided, along with scene-adaptive inference for real-world deployment with mismatched sensor swath widths.

---

## Table of Contents

1. [Background](#background)
2. [Pipeline Variants](#pipeline-variants)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Data Preparation](#data-preparation)
6. [Training](#training)
7. [Inference](#inference)
8. [Scene-Adaptive Inference](#scene-adaptive-inference)
9. [Visualisation](#visualisation)
10. [Hyperparameter Tuning](#hyperparameter-tuning)
11. [Metrics](#metrics)
12. [Offline / Airgapped Setup](#offline--airgapped-setup)
13. [References](#references)

---

## Background

MWIR and LWIR channels have fundamentally different spectral statistics. MWIR (3–5 µm) contains both solar-reflected and thermal emission components, producing high dynamic range and strong shadows. LWIR (8–12 µm) is purely thermal emission — lower contrast, scene-temperature-dependent, and heavily influenced by surface emissivity. A naive pixel-space model regresses to the scene mean on homogeneous regions (agriculture, water) and systematically misassigns brightness to surfaces with different MWIR/LWIR emissivity ratios (roads, rooftops).

This pipeline addresses those failure modes specifically through:

- Characteristic Function Consistency (CFC) loss — matches the full empirical patch distribution, not just pixel-level MSE
- Spectral consistency loss — preserves spatial frequency content at structural edges
- Cross-modal attention — queries MWIR features at every decoder scale
- Local Texture Gram loss — penalises intra-class texture collapse in homogeneous scenes
- Scene Histogram loss — corrects systematic per-scene thermal offset/gain
- Bridge diffusion scheduler — seeds the forward process from an MWIR-derived prior, cutting mean-regression artefacts
- Latent diffusion (LDM variant) — operates in a perceptually compressed latent space, improving distribution coverage with limited training pairs

---

## Pipeline Variants

Three variants are provided. Choose based on dataset size and compute.

| Variant | Config | Entrypoint | Best for |
|---|---|---|---|
| **v1 — Baseline DDPM** | `configs/base.json` | `train.py` | Rapid prototyping, ablation |
| **v2 — Improved pixel-space** | `configs/improved_v2.json` | `train.py` | 500–2000 pairs, good GPU baseline |
| **LDM — Latent DiT** | `configs/ldm.json` | `train_ldm.py` | Best quality, A100-class hardware |

All three share the same data pipeline, visualiser, and scene-adaptive inference layer.

---

## Project Structure

```
mwir2lwir/
│
├── train.py                        # v1 / v2 training entrypoint
├── train_ldm.py                    # LDM two-stage training entrypoint
├── offline_setup.py                # Airgapped machine: download / install / verify
├── requirements.txt
│
├── configs/
│   ├── base.json                   # v1 baseline hyperparameters
│   ├── improved_v2.json            # v2 targeted-improvement hyperparameters
│   └── ldm.json                    # LDM stage-1 VAE + stage-2 DiT hyperparameters
│
├── data/
│   ├── dataset.py                  # MWIRLWIRDataset + physics-aware augmentations
│   └── ir_pairs/                   # Your data goes here (not committed)
│       ├── mwir/  scene_001.npy …
│       └── lwir/  scene_001.npy …
│
├── models/
│   ├── conditional_unet.py         # v1/v2: UNet with cross-modal attention, AdaGN, RoPE
│   ├── diffusion_scheduler.py      # DDPM/DDIM scheduler + CFC + Spectral losses
│   ├── targeted_improvements.py    # v2 additions: Gram loss, Histogram loss, Bridge scheduler, LoRA
│   └── ldm/
│       ├── vae.py                  # KL-VAE (f=4), Gabor perceptual loss, SD weight transfer
│       └── dit.py                  # Conditional DiT-B/4: RoPE, register tokens, QK-norm, CFG
│
├── training/
│   ├── trainer.py                  # v1: EMA, AMP, cosine LR, best-checkpoint saving
│   ├── improved_trainer.py         # v2: adds bridge diffusion, scene context encoder
│   ├── ldm_trainer.py              # LDM: VAETrainer (stage 1) + DiTTrainer (stage 2)
│   └── visualizer.py               # Fixed-sample visualiser (deterministic across restarts)
│
├── inference/
│   ├── infer.py                    # v1/v2 inference: patch sliding-window, ensemble, SAI
│   ├── ldm_infer.py                # LDM inference: CFG sampling, latent blending, SAI
│   └── scene_adaptive.py           # SceneAdaptiveInference: swath alignment, histogram
│                                   #   calibration, LoRA scene fine-tuning
│
├── pretrained_weights/             # Downloaded by offline_setup.py (not committed)
│   └── sd_vae/
│       └── diffusion_pytorch_model.safetensors
│
└── runs/                           # Generated at runtime (not committed)
    ├── mwir2lwir/                  # v1 run outputs
    ├── mwir2lwir_v2/               # v2 run outputs
    └── mwir2lwir_ldm/
        ├── stage1_vae/
        └── stage2_dit/
```

Each run directory contains:

```
runs/<run>/
├── config.json
├── train_log.jsonl
├── checkpoints/
│   ├── ckpt_step_NNNNNNN.pt       # periodic (last 3 kept)
│   ├── ckpt_best.pt               # best test-split PSNR (never pruned)
│   └── ckpt_final.pt
├── train-results/
│   └── step_NNNNNNN/
│       ├── overview.png           # MWIR | Generated LWIR | Real LWIR grid
│       ├── sample_00_<id>_gen.npy
│       ├── sample_00_<id>_gen.png
│       └── psnr.json
└── test-results/
    └── step_NNNNNNN/  (same structure)
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

Paired MWIR and LWIR files must share the same filename stem and live in `mwir/` and `lwir/` subdirectories under `data_root`:

```
data/ir_pairs/
  mwir/
    scene_001.npy    # float32, shape (H, W) or (1, H, W)
    scene_002.npy
  lwir/
    scene_001.npy
    scene_002.npy
```

Supported formats: `.npy`, `.npz`, `.tif` / `.tiff` (requires `rasterio`), `.png`.

Values should be raw sensor counts or calibrated radiance — the dataset applies per-channel percentile normalisation (p2–p98 stretch to `[-1, 1]`) internally. Do not pre-normalise.

The dataset splits into train / val / test via `val_frac` (default 10% each for val and test). The split is deterministic by filename sort order — no random seeding needed.

---

## Training

### v1 — Baseline (fastest to run, good for ablation)

```bash
python train.py --config configs/base.json
```

### v2 — Improved pixel-space (recommended starting point)

```bash
python train.py --config configs/improved_v2.json
```

v2 adds four targeted fixes over the baseline:

- `LocalTextureGramLoss` — Gabor filter bank Gram matrices on local patches; fixes intra-class texture collapse on homogeneous scenes (agriculture, water bodies)
- `SceneHistogramLoss` — differentiable KDE histogram matching; corrects systematic per-scene thermal offset (road brightness, surface emissivity errors)
- `GlobalSceneContextEncoder` — encodes scene-level statistical moments into the UNet's AdaGN conditioning; resolves ambiguous MWIR patches whose LWIR temperature depends on global scene context
- `BridgeDiffusionScheduler` — seeds the forward process from a lightweight MWIR-derived LWIR prior instead of pure Gaussian noise; eliminates mean-regression artefacts on flat regions

### LDM — Latent Diffusion (best quality, two-stage)

**Stage 1: Fine-tune the VAE on LWIR imagery (~50k steps)**

```bash
python train_ldm.py --config configs/ldm.json --stage1_only
```

**Stage 1 + Stage 2 together:**

```bash
python train_ldm.py --config configs/ldm.json
```

**Stage 2 only (VAE already trained):**

```bash
python train_ldm.py --config configs/ldm.json \
  --skip_vae \
  --vae_ckpt runs/mwir2lwir_ldm/stage1_vae/vae_final.pt
```

The LDM operates on 64×64×4 latents (f=4 VAE, 256×256 input). The DiT uses adaLN-Zero conditioning, 2D RoPE positional embeddings, register tokens to prevent attention sink artefacts on flat IR regions, QK-norm for stability on heterogeneous thermal statistics, and classifier-free guidance training (10% null conditioning) to enable quality-boosting CFG at inference.

### Resuming from checkpoint

```bash
python train.py --config configs/improved_v2.json \
  --resume runs/mwir2lwir_v2/checkpoints/ckpt_step_0050000.pt
```

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

Stochastic ensemble (averages N samples; improves PSNR by ~0.5–1 dB):

```bash
python inference/infer.py ... --ensemble 5 --eta 0.5
```

### LDM inference

```bash
python inference/ldm_infer.py \
  --vae_ckpt runs/mwir2lwir_ldm/stage1_vae/vae_final.pt \
  --dit_ckpt runs/mwir2lwir_ldm/stage2_dit/checkpoints/dit_best.pt \
  --mwir scene_mwir.npy \
  --output outputs/scene_lwir.npy \
  --guidance_scale 5.0 \
  --num_steps 50
```

CFG guidance scale (3.0–7.0) is the most impactful inference-time knob. Higher values produce sharper output but reduce diversity. Start at 5.0 and adjust based on visual quality.

---

## Scene-Adaptive Inference

In real-world deployment the MWIR sensor covers a wider swath than the LWIR sensor. The default configuration is:

```
MWIR swath:  3 km  (full scene width)
LWIR swath:  2 km  (central strip, scene centres aligned, same row count)
```

The central 2/3 of every MWIR column range overlaps exactly with the available LWIR data. `SceneAdaptiveInference` uses this overlap region to calibrate or fine-tune the model per scene before generating LWIR for the full 3 km MWIR swath.

### Adaptation stages

**Stage A — Swath alignment** (always runs, microseconds)
Computes the pixel-column overlap from swath widths and validates the array dimensions. No parameters.

**Stage B — Histogram calibration** (default on, milliseconds, no GPU)
Fits a 256-quantile piecewise-linear map from model output CDF to real LWIR CDF on the overlap strip. Corrects systematic per-scene radiometric offset caused by time-of-day, season, atmospheric path length, and viewing angle variations. Saved as `histogram_calibration.npz` alongside each scene output.

**Stage C — LoRA scene fine-tuning** (optional, 1–5 minutes on A100)
Runs N gradient steps on random crops from the overlap strip. Only low-rank adapters (rank 4) on cross-attention projections are trained — base weights are frozen, preventing catastrophic forgetting. Use for scenes that are genuinely out-of-distribution (new land cover, new sensor altitude, different atmospheric conditions).

### When to use each stage

| Scenario | Recommended stages |
|---|---|
| Same sensor, same region as training data | A + B |
| Same sensor, new geographic region | A + B + C (50–100 steps) |
| Different sensor altitude or configuration | A + B + C (100–200 steps) |
| Completely new sensor type | A + B + C + collect new training pairs |

### Commands

Standard inference (no adaptation):

```bash
python inference/infer.py \
  --checkpoint runs/mwir2lwir/checkpoints/ckpt_best.pt \
  --mwir scene_mwir.npy \
  --output outputs/scene.npy
```

With histogram calibration (typical deployment):

```bash
python inference/infer.py \
  --checkpoint runs/mwir2lwir/checkpoints/ckpt_best.pt \
  --mwir scene_mwir.npy \
  --lwir scene_lwir_overlap.npy \
  --mwir_swath_km 3.0 --lwir_swath_km 2.0 \
  --scene_id orbit_042_20240315 \
  --output outputs/orbit_042/lwir_full.npy
```

With LoRA scene fine-tuning (OOD scene):

```bash
python inference/infer.py \
  --checkpoint runs/mwir2lwir/checkpoints/ckpt_best.pt \
  --mwir scene_mwir.npy \
  --lwir scene_lwir_overlap.npy \
  --mwir_swath_km 3.0 --lwir_swath_km 2.0 \
  --scene_finetune --finetune_steps 100 \
  --scene_id orbit_042_20240315 \
  --output outputs/orbit_042/lwir_full.npy
```

LDM pipeline with SAI uses the same flags on `ldm_infer.py` with `--vae_ckpt` and `--dit_ckpt` instead of `--checkpoint`.

### Per-scene output directory

When SAI is active, the output directory for each `scene_id` contains:

```
outputs/<scene_id>/
├── lwir_full_generated.npy        # synthesised LWIR, full MWIR swath
├── lwir_full_generated.png
├── lwir_overlap_generated.npy     # synthesised LWIR on overlap strip (pre-calibration)
├── lwir_overlap_real.npy          # real LWIR (your input)
├── comparison_overlap.png         # MWIR | Generated | Real side-by-side
├── histogram_calibration.npz      # fitted quantile map (reusable)
└── metrics.json                   # PSNR, MSE on overlap strip
```

---

## Visualisation

The `Visualizer` class selects N samples from train and test datasets at init using deterministic even-spaced indices (with a fixed seed), caches them as tensors, and writes comparison grids at configurable intervals during training. The same indices are used on every run, including restarts from checkpoint.

Config keys (all optional, sensible defaults):

```json
"vis_n_samples": 8,
"vis_seed":      42,
"vis_every":     2000
```

Output per visualisation step:

- `overview.png` — N-row grid: MWIR input | Generated LWIR | Real LWIR
- `sample_<i>_<scene>_{mwir,gen,real}.npy` — individual float32 arrays
- `sample_<i>_<scene>_{mwir,gen,real}.png` — p2–p98 normalised PNG
- `psnr.json` — per-sample and mean PSNR

PNG normalisation uses p2–p98 percentile stretch rather than min-max to handle IR outliers (hot targets, cold shadows) without crushing scene contrast.

---

## Hyperparameter Tuning

### Diagnosing failure modes

| Symptom | Root cause | Fix |
|---|---|---|
| Homogeneous regions (fields, water) look flat / uniform | Mean regression in low-contrast zones | Increase `lambda_cfc` (0.15 → 0.3), enable v2 Gram loss |
| Road / rooftop brightness systematically wrong | MWIR/LWIR emissivity mismatch | Increase `lambda_hist` (0.05 → 0.15) |
| Edges blurred at structures | Spectral frequency loss | Increase `lambda_spectral` (0.10 → 0.20) |
| Intra-field texture variation absent | Gram loss too weak | Increase `lambda_gram` (0.05 → 0.10) |
| Training unstable early | Gram loss spikes | Reduce `lambda_gram` first; check learning rate |
| OOD scene at inference | Distribution shift | Use SAI with `--scene_finetune` |

### Architecture scaling by VRAM

| GPU VRAM | v1 / v2 config | LDM config |
|---|---|---|
| 8 GB | `base_channels=64, channel_mults=[1,2,4], batch_size=4` | DiT-S/4, `dit_batch_size=4` |
| 16 GB | `base_channels=128, channel_mults=[1,2,4,8], batch_size=8` | DiT-B/4, `dit_batch_size=8` |
| 40 GB | `base_channels=192, channel_mults=[1,2,4,8], batch_size=16, num_res_blocks=3` | DiT-L/4, `dit_batch_size=16` |

### Dataset size recommendations

| Training pairs | Recommended variant | VAE strategy |
|---|---|---|
| < 500 | v2 with heavy augmentation | Skip LDM |
| 500–2000 | v2 or LDM | Fine-tune from SD KL-VAE |
| 2000–10000 | LDM DiT-B/4 | Fine-tune from SD KL-VAE |
| > 10000 | LDM DiT-L/4 | Train VAE from scratch |

### LDM-specific tuning

The VAE `scale_factor` is computed empirically after Stage 1 (call `vae.compute_scale_factor(train_loader)`) and stored in the checkpoint. It normalises the latent space to approximately N(0,1) so the DiT's cosine noise schedule is correctly calibrated. Do not skip this step.

CFG guidance scale at inference is the most impactful quality knob post-training. Start at 5.0, increase toward 7.0 for sharper output. Above 7.0 typically causes mode collapse artefacts.

---

## Metrics

| Metric | Description | Direction |
|---|---|---|
| PSNR | Peak Signal-to-Noise Ratio | Higher is better |
| SSIM | Structural Similarity Index | Higher is better |
| ERGAS | Relative global error in synthesis | Lower is better |
| CFC Distance | Characteristic function L2 distance (distribution match) | Lower is better |
| MSE | Mean squared error in normalised [-1,1] space | Lower is better |

CFC distance is the most informative metric for thermal IR synthesis because it measures full distribution matching (all moments), not just second-order statistics. A model can achieve good PSNR while failing on CFC distance — this indicates correct global brightness but wrong local texture distribution, which is exactly the heterogeneity failure mode.

---

## Offline / Airgapped Setup

The pipeline requires one downloaded weight file (the SD KL-VAE, used only for the LDM variant) and all Python packages as offline wheels.

**On an internet-connected machine:**

```bash
# Downloads VAE weights (~335 MB) + all pip wheels (~3–5 GB)
python offline_setup.py --download \
  --cuda_version cu121 \
  --python_version 3.10

# Transfer to airgapped machine:
#   pretrained_weights/
#   offline_wheels/
```

**On the airgapped machine:**

```bash
python offline_setup.py --install        # installs from local wheels, no internet
python offline_setup.py --patch_config   # updates configs/ldm.json with local VAE path
python offline_setup.py --verify         # confirms imports and weights are present
```

If you choose not to use the SD VAE initialisation, set `"sd_vae_path": null` in `configs/ldm.json` and increase `vae_total_steps` from 50000 to 100000 to compensate for the lack of pretrained initialisation.

No other components auto-download at runtime. Gabor filter banks are computed analytically. No VGG, CLIP, or pretrained backbone is used anywhere.

---

## References

- Ho et al. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS.
- Song et al. (2020). *Denoising Diffusion Implicit Models.* ICLR 2021.
- Nichol & Dhariwal (2021). *Improved Denoising Diffusion Probabilistic Models.* ICML.
- Rombach et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR.
- Peebles & Xie (2023). *Scalable Diffusion Models with Transformers.* ICCV.
- Darcet et al. (2023). *Vision Transformers Need Registers.* ICLR 2024.
- Su et al. (2023). *RoFormer: Enhanced Transformer with Rotary Position Embedding.* Neurocomputing.
- Ansari et al. (2020). *Characteristic Function-based Methods for Generative Models.*
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.