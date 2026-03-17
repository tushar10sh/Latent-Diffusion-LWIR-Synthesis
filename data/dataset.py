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
        """Simulate atmospheric path radiance variation across scenes."""
        offset = torch.empty(img.shape[0], 1, 1).uniform_(-max_offset, max_offset)
        return img + offset.to(img.device)

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
    Percentile-based normalization — robust to outliers in thermal scenes
    (e.g., hot targets like engines, cold shadows).
    Maps [p_low, p_high] → [-1, 1].
    """
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
        mwir_dir = self.root / 'MWIR-normalized-globalstats'
        lwir_dir = self.root / 'LWIR-normalized-globalstats'

        all_mwir_files = sorted([
            f.stem for f in mwir_dir.iterdir()
            if f.suffix.lower() in self.SUPPORTED_EXTS
        ])
        all_lwir_files = sorted([
            f.stem for f in lwir_dir.iterdir()
            if f.suffix.lower() in self.SUPPORTED_EXTS
        ])

        # Train / val / test split
        n = len(all_mwir_files)
        n_val = max(1, int(n * val_frac))
        n_test = max(1, int(n * val_frac))
        if split == 'train':
            mwir_files = all_mwir_files[:n - n_val - n_test]
            lwir_files = all_lwir_files[:n - n_val - n_test]
        elif split == 'val':
            mwir_files = all_mwir_files[n - n_val - n_test: n - n_test]
            lwir_files = all_lwir_files[n - n_val - n_test: n - n_test]
        else:
            mwir_files = all_mwir_files[n - n_test:]
            lwir_files = all_lwir_files[n - n_test:]

        self.mwir_paths = [mwir_dir / f'{f}.{file_ext}' for f in mwir_files]
        self.lwir_paths = [lwir_dir / f'{f}.{file_ext}' for f in lwir_files]

        print(f"[Dataset] {split}: {len(all_mwir_files)} pairs found in {root}")

    def __len__(self):
        return len(self.mwir_paths)

    def _load(self, path: Path) -> np.ndarray:
        ext = path.suffix.lower()
        if ext == '.npy':
            arr = np.load(path)
            # arr = np.fromfile(path, dtype=np.float32)
            # arr = np.reshape(arr, (224, 224))
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
        return arr.astype(np.float32)

    def _normalize(self, arr: np.ndarray) -> torch.Tensor:
        # Per-channel percentile normalization
        normed = np.stack([
            percentile_normalize(arr[c]) for c in range(arr.shape[0])
        ], axis=0)
        return torch.from_numpy(normed)

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
