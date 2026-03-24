"""
MWIR / LWIR paired dataset with physics-aware augmentation.

Design decisions for heterogeneous thermal IR data:
  - Radiometric normalization per band (not global ImageNet stats)
  - Paired augmentation only with geometrically valid transforms
    (no colour jitter; thermal radiance is physical, not aesthetic)
  - Thermal noise injection (NEDT simulation)
  - Atmospheric path radiance perturbation
  - Cross-band contrast normalisation to reduce heterogeneity before network input
"""

import os
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, List, Tuple, Callable
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Thermal physics augmentations
# ─────────────────────────────────────────────

class ThermalAugmentor:
    """
    Physics-motivated augmentations for thermal IR imagery.
    All augmentations are applied identically to MWIR and LWIR
    (or only to MWIR to simulate sensor noise differences).
    """

    @staticmethod
    def add_nedt_noise(img: torch.Tensor, nedt_sigma: float = 0.01) -> torch.Tensor:
        """Simulate Noise Equivalent Differential Temperature (NEDT)."""
        return img + torch.randn_like(img) * nedt_sigma

    @staticmethod
    def random_radiance_offset(img: torch.Tensor, max_offset: float = 0.05) -> torch.Tensor:
        """Simulate atmospheric path radiance variation across scenes.
        img: (B, C, H, W) — offset is per-sample, broadcast over C, H, W."""
        offset = torch.empty(img.shape[0], 1, 1, 1, device=img.device, dtype=img.dtype
                             ).uniform_(-max_offset, max_offset)
        return img + offset

    @staticmethod
    def random_emissivity_scale(img: torch.Tensor, scale_range=(0.9, 1.1)) -> torch.Tensor:
        """Simulate per-material emissivity uncertainty."""
        scale = random.uniform(*scale_range)
        return img * scale

    @staticmethod
    def simulate_sensor_blur(img: torch.Tensor, sigma_range=(0.3, 1.0)) -> torch.Tensor:
        """MWIR sensors often have slightly different MTF from LWIR."""
        sigma = random.uniform(*sigma_range)
        k = max(3, int(sigma * 4) | 1)   # ensure odd
        # Gaussian kernel
        coords = torch.arange(k, dtype=torch.float32) - k // 2
        g = torch.exp(-0.5 * (coords / sigma) ** 2)
        g = g / g.sum()
        kernel = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
        kernel = kernel.expand(img.shape[1], 1, k, k).to(img.device)
        pad = k // 2
        return F.conv2d(img, kernel, padding=pad, groups=img.shape[1])

    @staticmethod
    def random_horizontal_flip(mwir: torch.Tensor, lwir: torch.Tensor):
        if random.random() > 0.5:
            return mwir.flip(-1), lwir.flip(-1)
        return mwir, lwir

    @staticmethod
    def random_vertical_flip(mwir: torch.Tensor, lwir: torch.Tensor):
        if random.random() > 0.5:
            return mwir.flip(-2), lwir.flip(-2)
        return mwir, lwir

    @staticmethod
    def random_rotation_90(mwir: torch.Tensor, lwir: torch.Tensor):
        k = random.randint(0, 3)
        return torch.rot90(mwir, k, [-2, -1]), torch.rot90(lwir, k, [-2, -1])

    @staticmethod
    def random_crop(mwir: torch.Tensor, lwir: torch.Tensor, crop_size: int):
        _, _, H, W = mwir.shape
        if H <= crop_size or W <= crop_size:
            return (
                F.interpolate(mwir, size=(crop_size, crop_size), mode='bilinear', align_corners=False),
                F.interpolate(lwir, size=(crop_size, crop_size), mode='bilinear', align_corners=False),
            )
        top = random.randint(0, H - crop_size)
        left = random.randint(0, W - crop_size)
        return (
            mwir[..., top:top+crop_size, left:left+crop_size],
            lwir[..., top:top+crop_size, left:left+crop_size],
        )


# ─────────────────────────────────────────────
# Normalization utilities
# ─────────────────────────────────────────────

def percentile_normalize(
    img: np.ndarray,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
) -> np.ndarray:
    """
    Percentile-based normalization — robust to outliers in thermal scenes.
    Maps [p_low, p_high] → [-1, 1].

    NaN/Inf handling: invalid pixels are replaced with the channel median
    before computing percentiles, preventing silent NaN propagation that
    causes all downstream losses to be NaN and all optimizer steps to be
    silently skipped by GradScaler.
    """
    # Replace NaN/Inf with the finite median so percentile is computable
    finite_mask = np.isfinite(img)
    if not finite_mask.all():
        n_bad = (~finite_mask).sum()
        median_val = float(np.median(img[finite_mask])) if finite_mask.any() else 0.0
        img = img.copy()
        img[~finite_mask] = median_val
        # Warn only once per unique bad count to avoid log spam
        import warnings
        warnings.warn(
            f"[percentile_normalize] {n_bad} NaN/Inf pixel(s) replaced with "
            f"median ({median_val:.4f}). Check source data for sensor dropouts "
            f"or invalid no-data values.",
            RuntimeWarning, stacklevel=2,
        )

    lo = np.percentile(img, low_pct)
    hi = np.percentile(img, high_pct)
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-8)   # [0, 1]
    return img * 2.0 - 1.0                  # [-1, 1]


def local_contrast_normalize(
    img: torch.Tensor,
    kernel_size: int = 65,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Local contrast normalization (LCN) — reduces the low-heterogeneity
    problem by normalizing each patch by local mean and std.
    Critical for homogeneous vegetation / water bodies in LWIR.
    """
    pad = kernel_size // 2
    mean = F.avg_pool2d(img, kernel_size, stride=1, padding=pad)
    diff = img - mean
    std = (F.avg_pool2d(diff**2, kernel_size, stride=1, padding=pad) + eps).sqrt()
    return diff / std


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class MWIRLWIRDataset(Dataset):
    """
    Paired MWIR / LWIR dataset.

    Directory structure expected:
        root/
          mwir/
            scene_001.npy   (or .tif, .png)
            scene_002.npy
            ...
          lwir/
            scene_001.npy
            scene_002.npy
            ...

    Files should be single-channel float arrays.
    Paired by filename.

    Args:
        root:           Dataset root directory
        split:          'train' | 'val' | 'test'
        image_size:     Output spatial size (square)
        augment:        Whether to apply augmentation
        use_lcn:        Apply local contrast normalization
        nedt_sigma:     NEDT noise sigma (set 0 to disable)
        val_frac:       Fraction of data for validation
        file_ext:       File extension ('npy', 'tif', 'png')
        mwir_channels:  Number of MWIR channels to load
        lwir_channels:  Number of LWIR channels to load
    """

    SUPPORTED_EXTS = ('.npy', '.tif', '.tiff', '.png', '.npz')

    def __init__(
        self,
        root: str,
        split: str = 'train',
        image_size: int = 256,
        augment: bool = True,
        use_lcn: bool = False,
        nedt_sigma: float = 0.01,
        val_frac: float = 0.1,
        file_ext: str = 'npy',
        mwir_channels: int = 1,
        lwir_channels: int = 1,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == 'train')
        self.use_lcn = use_lcn
        self.nedt_sigma = nedt_sigma
        self.mwir_channels = mwir_channels
        self.lwir_channels = lwir_channels
        self.aug = ThermalAugmentor()

        # Discover paired files
        # mwir_dir = self.root / 'MWIR-normalized-globalstats'
        # lwir_dir = self.root / 'LWIR-normalized-globalstats'
        mwir_dir = self.root / 'MWIR'
        lwir_dir = self.root / 'LWIR'

        all_mwir_files = sorted([
            f.stem for f in mwir_dir.iterdir()
            if f.suffix.lower() in self.SUPPORTED_EXTS
        ])
        all_lwir_files = sorted([
            f.stem for f in lwir_dir.iterdir()
            if f.suffix.lower() in self.SUPPORTED_EXTS
        ])

        # ── Pair matching: only keep stems present in BOTH bands ──
        mwir_set = set(all_mwir_files)
        lwir_set = set(all_lwir_files)
        paired_stems = sorted(mwir_set & lwir_set)
        if len(paired_stems) < len(all_mwir_files) or len(paired_stems) < len(all_lwir_files):
            n_mwir_only = len(mwir_set - lwir_set)
            n_lwir_only = len(lwir_set - mwir_set)
            print(
                f"[Dataset] WARNING: {n_mwir_only} MWIR-only and {n_lwir_only} "
                f"LWIR-only files dropped (no matching pair)."
            )

        # ── Deterministic shuffle before split ──
        # Sorted stems are alphabetical (often temporal).  A seeded shuffle
        # ensures train/val/test share the same distribution of scenes.
        rng = np.random.RandomState(seed=42)
        rng.shuffle(paired_stems)

        # Train / val / test split
        n = len(paired_stems)
        n_val = max(1, int(n * val_frac))
        n_test = max(1, int(n * val_frac))
        if split == 'train':
            stems = paired_stems[:n - n_val - n_test]
        elif split == 'val':
            print(f"[MWIRLWIRDataset] val_fraction: {val_frac}")
            stems = paired_stems[n - n_val - n_test: n - n_test]
        else:
            stems = paired_stems[n - n_test:]

        self.mwir_paths = [mwir_dir / f'{f}.{file_ext}' for f in stems]
        self.lwir_paths = [lwir_dir / f'{f}.{file_ext}' for f in stems]

        print(f"[Dataset] {split}: {len(stems)} pairs found in {root}")

    def __len__(self):
        return len(self.mwir_paths)

    def _load(self, path: Path) -> np.ndarray:
        ext = path.suffix.lower()
        if ext == '.npy':
            arr = np.load(path)
            # arr[np.isnan(arr)] = 0 # explicitly set nan to zero
            # arr[np.isinf(arr)] = 0 # explicitly set nan to zero
            if np.any(np.isnan(arr)):
                print(f'[MWIRLWIRDataset] path: {path} has NaN')
            if np.any(np.isinf(arr)):
                print(f'[MWIRLWIRDataset] path: {path} has Inf')
        elif ext == '.npz':
            d = np.load(path)
            arr = d[list(d.keys())[0]]
        elif ext in ('.tif', '.tiff'):
            try:
                import rasterio
                with rasterio.open(path) as src:
                    arr = src.read().astype(np.float32)  # (C, H, W)
            except ImportError:
                raise ImportError("pip install rasterio for .tif support")
        elif ext == '.png':
            from PIL import Image
            arr = np.array(Image.open(path)).astype(np.float32)
            if arr.ndim == 2:
                arr = arr[None]  # (1, H, W)
            else:
                arr = arr.transpose(2, 0, 1)
        else:
            raise ValueError(f"Unsupported extension: {ext}")

        if arr.ndim == 2:
            arr = arr[None]  # ensure (C, H, W)
        arr = arr.astype(np.float32)

        # Defensive sanitisation: replace NaN/Inf with per-channel median.
        # GeoTIFF no-data regions and some sensor output files use NaN or large
        # sentinel values. A single NaN propagates through np.percentile → NaN
        # normalised array → NaN model input → NaN loss.
        # NOTE: This is a defensive guard, not the confirmed cause of the NaN
        # losses observed during diffusion training (see trainer.py comments for
        # the actual cause: GradScaler + bfloat16 incompatibility). However,
        # sanitising at load time is correct practice regardless.
        if not np.isfinite(arr).all():
            n_bad = (~np.isfinite(arr)).sum()
            for c in range(arr.shape[0]):
                ch = arr[c]
                finite_mask = np.isfinite(ch)
                if finite_mask.any():
                    fill = float(np.median(ch[finite_mask]))
                else:
                    fill = 0.0
                ch[~finite_mask] = fill
            import warnings
            warnings.warn(
                f"[Dataset] {path.name}: {n_bad} non-finite values (NaN/Inf) "
                f"replaced with per-channel median. Check your source data.",
                stacklevel=2,
            )

        return arr

    def _normalize(self, arr: np.ndarray) -> torch.Tensor:
        # Per-channel percentile normalization
        normed = np.stack([
            percentile_normalize(arr[c]) for c in range(arr.shape[0])
        ], axis=0)
        t = torch.from_numpy(normed)
        # Final guard: percentile_normalize is robust but can still produce
        # NaN if the channel is all-identical values (hi==lo, division by ~1e-8).
        # Replace any remaining NaN/Inf with 0.0 (neutral in [-1,1] space).
        if not torch.isfinite(t).all():
            t = torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)
        return t

    def _resize(self, t: torch.Tensor) -> torch.Tensor:
        if t.shape[-2] != self.image_size or t.shape[-1] != self.image_size:
            t = F.interpolate(
                t.unsqueeze(0), size=(self.image_size, self.image_size),
                mode='bilinear', align_corners=False
            ).squeeze(0)
        return t

    def __getitem__(self, idx: int):
        mwir_arr = self._load(self.mwir_paths[idx])
        lwir_arr = self._load(self.lwir_paths[idx])

        mwir = self._normalize(mwir_arr)
        lwir = self._normalize(lwir_arr)

        # Resize to target resolution
        mwir = self._resize(mwir)
        lwir = self._resize(lwir)

        if self.augment:
            # Add batch dim for ops that need it
            mwir = mwir.unsqueeze(0)
            lwir = lwir.unsqueeze(0)

            # Paired geometric augmentations
            mwir, lwir = self.aug.random_horizontal_flip(mwir, lwir)
            mwir, lwir = self.aug.random_vertical_flip(mwir, lwir)
            mwir, lwir = self.aug.random_rotation_90(mwir, lwir)
            mwir, lwir = self.aug.random_crop(mwir, lwir, self.image_size)

            mwir = mwir.squeeze(0)
            lwir = lwir.squeeze(0)

            # MWIR-only radiometric augmentation (sensor differences)
            mwir = self.aug.random_radiance_offset(mwir.unsqueeze(0)).squeeze(0)
            mwir = self.aug.random_emissivity_scale(mwir)
            if random.random() > 0.5:
                mwir = self.aug.simulate_sensor_blur(mwir.unsqueeze(0)).squeeze(0)

            # NEDT noise on both (different sensor noise floors)
            mwir = self.aug.add_nedt_noise(mwir, self.nedt_sigma * random.uniform(0.5, 2.0))
            lwir = self.aug.add_nedt_noise(lwir, self.nedt_sigma)

        # Optional local contrast normalization (helps heterogeneity)
        if self.use_lcn:
            mwir = local_contrast_normalize(mwir.unsqueeze(0)).squeeze(0)
            lwir = local_contrast_normalize(lwir.unsqueeze(0)).squeeze(0)

        return {
            'mwir': mwir.float(),          # (C, H, W), range ≈ [-1, 1]
            'lwir': lwir.float(),          # (C, H, W), range ≈ [-1, 1]
            'path': str(self.mwir_paths[idx].stem),
        }


def build_dataloaders(
    root: str,
    image_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
    use_lcn: bool = False,
    val_frac: float = 0.1,
    file_ext: str = 'npy',
):
    """Convenience function to create train/val DataLoaders."""
    train_ds = MWIRLWIRDataset(root, 'train', image_size, augment=True,
                                use_lcn=use_lcn, val_frac=val_frac, file_ext=file_ext)
    val_ds = MWIRLWIRDataset(root, 'val', image_size, augment=False,
                              use_lcn=use_lcn, val_frac=val_frac, file_ext=file_ext)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
