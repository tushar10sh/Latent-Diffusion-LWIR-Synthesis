# MWIR → LWIR Synthesis Pipeline

Conditional diffusion models for synthesizing Long-Wave Infrared (LWIR, 8–12 µm) imagery from Mid-Wave Infrared (MWIR, 3–5 µm) observations from spaceborne EO platforms.

---

## Architecture Overview

This pipeline supports three primary architectures:

### 1. Conditional DDPM (Pixel Space)
A standard U-Net that operates directly on image pixels, conditioned on MWIR features via cross-modal attention. Best for smaller resolutions or when extreme pixel-level fidelity is required.

### 2. Improved DDPM (v2)
An evolution of the pixel-space model with targeted improvements in normalization (AdaGN), spectral loss, and local contrast normalization (LCN) to better handle thermal heterogeneity.

### 3. Latent Diffusion Model (LDM)
A two-stage approach for higher efficiency and resolution:
- **Stage 1: VAE** — Compresses LWIR imagery into a compact latent space (f=4).
- **Stage 2: DiT (Diffusion Transformer)** — Learns to generate LWIR latents conditioned on MWIR features.
- **Key benefit:** 16× reduction in computation during diffusion, allowing for deeper models (DiT-B/L) and better global coherence.

```
MWIR Image ──► MWIREncoder ──► Latent Features
                                     │
Noisy Latent + t ──► Conditional DiT ──┤ (cross-attention)
                                     │
                              ──► ε̂ (predicted noise) ──► DDIM sampling ──► VAE Decoder ──► Synthetic LWIR
```

---

## Problem: Low Heterogeneity in Thermal IR

MWIR and LWIR have fundamentally different spectral statistics:
- **MWIR** contains solar-reflected + thermal emission → high dynamic range, strong shadows.
- **LWIR** is purely thermal emission → lower contrast, scene-temperature-dependent.

This causes vanilla DDPM to:
1. Over-smooth homogeneous regions (vegetation, water).
2. Miss fine structural detail (roads, building edges) in low-contrast zones.

### Mitigations in this pipeline

1. **CFC loss** — penalizes distribution mismatch at patch level (not just pixel MSE).
2. **Spectral loss** — penalizes log-PSD mismatch, preserving spatial frequency content.
3. **Cross-modal attention** — lets the model query MWIR structural features at each scale.
4. **NEDT augmentation** — simulates sensor noise differences, prevents mode collapse on clean data.
5. **Local Contrast Normalization** (v2) — pre-normalizes each patch to reduce low-contrast problems.
6. **Scene-Adaptive Inference (SAI)** — uses real-world overlap strips to calibrate the model to specific scene statistics.

---

## Scene-Adaptive Inference (SAI)

Handles the real-world deployment scenario where MWIR and LWIR sensors have different swath widths (e.g., 3km vs 2km). SAI uses the central overlapping strip to calibrate the model to the specific atmospheric and radiometric conditions of a new scene.

### Adaptation Stages:
1. **SwathAligner**: Computes geometric overlap based on sensor parameters.
2. **HistogramCalibrator**: Fits a piecewise-linear quantile map to match real LWIR statistics (corrects bias/gain).
3. **SceneFineTuner**: Rapidly adapts the model via **LoRA (Low-Rank Adaptation)** on cross-attention layers using the overlap strip as a reference.

---

## Project Structure

```
.
├── models/
│   ├── conditional_unet.py       # Pixel-space UNet
│   ├── diffusion_scheduler.py    # DDPM/DDIM + CFC + Spectral losses
│   ├── targeted_improvements.py  # Improved v2 model logic
│   └── ldm/
│       ├── vae.py                # IR-optimized VAE
│       └── dit.py                # Diffusion Transformer (S/B/L sizes)
├── data/
│   └── dataset.py                # Paired dataset + physics augmentations
├── training/
│   ├── trainer.py                # Pixel-space trainer
│   ├── improved_trainer.py       # v2 trainer
│   └── ldm_trainer.py            # VAE and DiT trainer
├── inference/
│   ├── infer.py                  # Pixel-space inference
│   ├── ldm_infer.py              # LDM inference + SAI entrypoint
│   └── scene_adaptive.py         # Swath alignment, HistCal, and LoRA logic
├── configs/
│   ├── base.json                 # Pixel-space hyperparameters
│   ├── improved_v2.json          # v2 hyperparameters
│   └── ldm.json                  # LDM (VAE + DiT) hyperparameters
├── train.py                      # Pixel-space entrypoint
├── train_improved.py             # Improved v2 entrypoint
└── train_ldm.py                  # LDM entrypoint
```

---

## Quick Start

### 1. Prepare data

```
data/ir_pairs/
  mwir/
    scene_001.npy    # shape: (H, W) or (1, H, W) float32
    ...
  lwir/
    scene_001.npy
    ...
```
Files must be paired by filename stem. Supported formats: `.npy`, `.npz`, `.tif`, `.png`.

### 2. Training

**Pixel-space (DDPM):**
```bash
python train.py --config configs/base.json
```

**Latent Diffusion (LDM):**
1. Train VAE: `python train_ldm.py --config configs/ldm.json --stage1_only`
2. Train DiT: `python train_ldm.py --config configs/ldm.json --skip_vae --vae_ckpt runs/.../vae_final.pt`

### 3. Inference

**Standard Pixel-space:**
```bash
python inference/infer.py --checkpoint runs/.../ckpt_final.pt --mwir scene.npy --output out.npy
```

**LDM + Scene Adaptation (SAI):**
```bash
python inference/ldm_infer.py \
  --vae_ckpt ... --dit_ckpt ... \
  --mwir full_swath_mwir.npy \
  --lwir overlap_strip_lwir.npy \
  --mwir_swath_km 3.0 --lwir_swath_km 2.0 \
  --scene_finetune --output out.npy
```

---

## Hyperparameter Tuning Guide

### Addressing heterogeneity / blur

| Issue | Solution |
|---|---|
| Blurry homogeneous regions | Increase `lambda_cfc` (0.1 → 0.3) |
| Loss of fine edges | Increase `lambda_spectral` (0.05 → 0.15) |
| Mode collapse on low-contrast | Enable `use_lcn: true` |
| Temporal inconsistency | Increase `ema_decay` (0.9999 → 0.99995) |

### Architecture scaling

| GPU VRAM | Recommended config |
|---|---|
| 8 GB  | `base_channels=64, channel_mults=[1,2,4], batch_size=4` |
| 16 GB | `base_channels=128, channel_mults=[1,2,4,8], batch_size=8` |
| 40 GB | `base_channels=192, channel_mults=[1,2,4,8], batch_size=16, num_res_blocks=3` |

---

## Key design choices

| Component | Choice | Motivation |
|---|---|---|
| Noise schedule | Cosine | Avoids over-noising low-contrast thermal scenes |
| Conditioning | Cross-modal attention | Precisely aligns MWIR structural cues with LWIR synthesis |
| Normalization | Spectral norm + AdaGN | Stabilizes training on heterogeneous IR statistics |
| LDM Latents | f=4 (KL-VAE) | Balance between compression and fine thermal texture preservation |
| Guidance | Classifier-Free (CFG) | Critical for quality in data-constrained IR domains (LDM) |
| Adaptation | LoRA | Enables per-scene fine-tuning without catastrophic forgetting |
| Sampling | DDIM (50 steps) | 20× faster than full DDPM, near-identical quality |
| CFC loss | Char. Function Consist. | Matches full distribution, not just pixel means |
| Spectral loss | Log-PSD L1 | Prevents blur / ringing at structural edges |

---

## Metrics

| Metric | Meaning |
|---|---|
| PSNR | Peak Signal-to-Noise Ratio (higher = better) |
| SSIM | Structural Similarity Index (higher = better) |
| ERGAS | Relative global synthesis error (lower = better) |
| CFC Distance | Characteristic function distance (lower = better distribution match) |

---

## References

- **LDM**: Rombach et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*
- **DiT**: Peebles & Xie (2023). *Scalable Diffusion Models with Transformers*
- **DDIM**: Song et al. (2020). *Denoising Diffusion Implicit Models*
- **LoRA**: Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*
- **CFC loss**: Ansari et al. (2020). *Characteristic Function-based Methods for Generative Models*
