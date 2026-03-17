# MWIR → LWIR Synthesis Pipeline

Conditional diffusion model for synthesizing Long-Wave Infrared (LWIR, 8–12 µm) imagery from Mid-Wave Infrared (MWIR, 3–5 µm) observations from spaceborne EO platforms.

---

## Architecture Overview

```
MWIR Image ──► MWIREncoder ──► Multi-scale features
                                     │
Noisy LWIR + t ──► ConditionalUNet ──┤ (cross-modal attention at each decoder level)
                                     │
                              ──► ε̂ (predicted noise) ──► DDIM sampling ──► Synthetic LWIR
```

### Key design choices

| Component | Choice | Motivation |
|---|---|---|
| Noise schedule | Cosine | Avoids over-noising low-contrast thermal scenes |
| Conditioning | Early fusion + cross-modal attention | Combines pixel-level and semantic-level MWIR cues |
| Normalization | Spectral norm + AdaGN | Stabilises training on heterogeneous IR statistics |
| Timestep emb | Fourier features | Better gradient flow than sinusoidal for deep networks |
| Sampling | DDIM (50 steps) | 20× faster than full DDPM, near-identical quality |
| CFC loss | Characteristic Function Consistency | Matches full distribution, not just pixel means — critical for thermal heterogeneity |
| Spectral loss | Log-PSD L1 | Prevents blur / ringing at structural edges |

---

## Problem: Low Heterogeneity in Thermal IR

MWIR and LWIR have fundamentally different spectral statistics:
- **MWIR** contains solar-reflected + thermal emission → high dynamic range, strong shadows
- **LWIR** is purely thermal emission → lower contrast, scene-temperature-dependent

This causes vanilla DDPM to:
1. Over-smooth homogeneous regions (vegetation, water)
2. Miss fine structural detail (roads, building edges) in low-contrast zones

### Mitigations in this pipeline

1. **CFC loss** — penalises distribution mismatch at patch level (not just pixel MSE)
2. **Spectral loss** — penalises log-PSD mismatch, preserving spatial frequency content
3. **Cross-modal attention** — lets the model query MWIR structural features at each scale
4. **NEDT augmentation** — simulates sensor noise differences, prevents mode collapse on clean data
5. **Local Contrast Normalization** (optional) — pre-normalises each patch to reduce low-contrast problem before network input

---

## Project Structure

```
mwir2lwir/
├── models/
│   ├── conditional_unet.py       # Main UNet with cross-modal attention
│   └── diffusion_scheduler.py    # DDPM/DDIM + CFC + Spectral losses
├── data/
│   └── dataset.py                # Paired dataset + physics augmentations
├── training/
│   └── trainer.py                # EMA, mixed precision, checkpointing
├── inference/
│   └── infer.py                  # Patch inference, ensemble, metrics
├── configs/
│   └── base.json                 # Training hyperparameters
├── train.py                      # Entrypoint
└── requirements.txt
```

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Prepare data

```
data/ir_pairs/
  mwir/
    scene_001.npy    # shape: (H, W) or (1, H, W) float32
    scene_002.npy
    ...
  lwir/
    scene_001.npy
    scene_002.npy
    ...
```

Files must be paired by filename stem. Supported formats: `.npy`, `.npz`, `.tif`, `.png`.

### 3. Train

```bash
python train.py --config configs/base.json
```

Resume from checkpoint:
```bash
python train.py --config configs/base.json --resume runs/mwir2lwir/checkpoints/ckpt_step_0050000.pt
```

### 4. Inference

```bash
python inference/infer.py \
  --checkpoint runs/mwir2lwir/checkpoints/ckpt_final.pt \
  --mwir path/to/mwir_scene.npy \
  --output outputs/lwir_generated.npy \
  --lwir_gt path/to/lwir_real.npy \   # optional, for metrics
  --num_steps 50 \
  --overlap 0.25                       # for large images
```

Ensemble inference (better quality, higher cost):
```bash
python inference/infer.py ... --ensemble 5 --eta 0.5
```

---

## Hyperparameter Tuning Guide

### Addressing heterogeneity / blur

| Issue | Solution |
|---|---|
| Blurry homogeneous regions | Increase `lambda_cfc` (0.1 → 0.3) |
| Loss of fine edges | Increase `lambda_spectral` (0.05 → 0.15) |
| Mode collapse on low-contrast | Enable `use_lcn: true` |
| Temporal inconsistency (video) | Increase `ema_decay` (0.9999 → 0.99995) |

### Architecture scaling

| GPU VRAM | Recommended config |
|---|---|
| 8 GB  | `base_channels=64, channel_mults=[1,2,4], batch_size=4` |
| 16 GB | `base_channels=128, channel_mults=[1,2,4,8], batch_size=8` |
| 40 GB | `base_channels=192, channel_mults=[1,2,4,8], batch_size=16, num_res_blocks=3` |

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

- **DDPM**: Ho et al. (2020). *Denoising Diffusion Probabilistic Models*
- **DDIM**: Song et al. (2020). *Denoising Diffusion Implicit Models*
- **Cosine schedule**: Nichol & Dhariwal (2021). *Improved Denoising Diffusion Probabilistic Models*
- **Characteristic functions for generative models**: Ansari et al. (2020). *Characteristic Function-based Methods for Generative Models*
- **Cross-modal attention**: Rombach et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*
