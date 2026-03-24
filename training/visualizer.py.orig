"""
Visualizer — fixed-sample image saver for MWIR→LWIR training.

Selects N samples from train and test datasets at init using a
deterministic seed, caches them as CPU tensors, and writes them
to disk at any requested step via Visualizer.save().

Output layout per call:
    <output_dir>/train-results/step_<N>/
        overview.png          side-by-side grid: MWIR | Generated | Real LWIR
        sample_<i>_mwir.npy
        sample_<i>_gen.npy
        sample_<i>_real.npy
    <output_dir>/test-results/step_<N>/
        (same structure)

PNG normalisation:
    Each channel is independently mapped to [0, 255] via
    percentile stretch (p2–p98) so both homogeneous and
    high-contrast IR scenes render clearly.
"""

import math
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from PIL import Image, ImageDraw
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# ─────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────

def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Map float32 array of any range → uint8 [0,255] via p2–p98 stretch.
    Robust to IR outliers (hot targets, cold water) that crush contrast
    under naive min-max normalisation.
    """
    lo = np.percentile(arr, 2)
    hi = np.percentile(arr, 98)
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def _tensor_to_np(t: torch.Tensor) -> np.ndarray:
    """(1,H,W) or (H,W) tensor → (H,W) float32 numpy on CPU."""
    return t.squeeze().cpu().float().numpy()


def _make_grid_png(
    rows: List[List[np.ndarray]],
    col_titles: List[str],
    padding: int = 4,
    title_px: int = 18,
) -> "Image.Image":
    """
    Build a labelled side-by-side PNG grid.
    rows[i][j] = (H,W) float32 array for sample i, column j.
    """
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow required: pip install pillow")

    n_rows = len(rows)
    n_cols = len(rows[0])
    H, W = rows[0][0].shape

    canvas_w = n_cols * (W + padding) + padding
    canvas_h = n_rows * (H + padding) + padding + title_px

    canvas = Image.new("L", (canvas_w, canvas_h), color=20)
    draw = ImageDraw.Draw(canvas)

    # Column titles (centred above each column)
    for c, title in enumerate(col_titles):
        x = padding + c * (W + padding) + W // 2
        draw.text((x, 3), title, fill=210, anchor="mt")

    # Paste sample patches
    for r, row in enumerate(rows):
        for c, arr in enumerate(row):
            patch = Image.fromarray(_to_uint8(arr), mode="L")
            x = padding + c * (W + padding)
            y = title_px + padding + r * (H + padding)
            canvas.paste(patch, (x, y))

    return canvas


# ─────────────────────────────────────────────
# Visualizer
# ─────────────────────────────────────────────

class Visualizer:
    """
    Fixed-sample visualiser for MWIR→LWIR training.

    Parameters
    ----------
    train_dataset : MWIRLWIRDataset  (split='train')
    test_dataset  : MWIRLWIRDataset  (split='val' or 'test')
    n_samples     : number of fixed samples to track per split
    seed          : RNG seed — guarantees the same indices on every run,
                    including restarts from checkpoint
    device        : device string used when calling generate_fn
    """

    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset:  Dataset,
        n_samples: int = 8,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.n_samples = n_samples
        self.device = torch.device(device)

        self._cache: Dict[str, dict] = {
            "train": self._select_and_cache(train_dataset, n_samples, seed),
            "test":  self._select_and_cache(test_dataset,  n_samples, seed + 1),
        }

        # Best-PSNR tracking per split (for saving best-weights signal)
        self._best_psnr: Dict[str, float] = {"train": -float("inf"), "test": -float("inf")}

        for split, cache in self._cache.items():
            print(
                f"[Visualizer] {split}: {n_samples} fixed samples, "
                f"indices = {cache['indices']}"
            )

    # ── Index selection ───────────────────────────────────────────

    @staticmethod
    def _select_and_cache(dataset: Dataset, n: int, seed: int) -> dict:
        """
        Pick n indices deterministically:
          1. Evenly space across dataset length for scene diversity.
          2. Deterministically shuffle with `seed`.
          3. Re-sort so the grid is in index order (consistent across runs).
        """
        N = len(dataset)
        n = min(n, N)

        # Even spacing
        base = [int(i * N / n) for i in range(n)]

        # Deterministic shuffle
        rng = np.random.default_rng(seed)
        rng.shuffle(base)
        indices = sorted(base)

        mwir_list, lwir_list, path_list = [], [], []
        for idx in indices:
            sample = dataset[idx]
            mwir_list.append(sample["mwir"])          # (C, H, W) tensor
            lwir_list.append(sample["lwir"])          # (C, H, W) tensor
            path_list.append(sample.get("path", str(idx)))

        return {
            "indices": indices,
            "mwir":    torch.stack(mwir_list),        # (N, C, H, W) — CPU
            "lwir":    torch.stack(lwir_list),        # (N, C, H, W) — CPU
            "paths":   path_list,
        }

    # ── Public API ────────────────────────────────────────────────

    def save(
        self,
        step: int,
        generate_fn: Callable[[torch.Tensor], torch.Tensor],
        output_dir: Path,
        split: str = "train",
    ) -> Optional[float]:
        """
        Generate images for the fixed sample set and write to disk.

        Parameters
        ----------
        step        : current training step (embedded in directory name)
        generate_fn : callable that accepts a (B, C, H, W) MWIR batch on
                      self.device and returns a (B, C, H, W) generated-LWIR
                      batch (any device).  The Visualizer does NOT call
                      torch.no_grad() — wrap that in the caller or here.
        output_dir  : root run directory, e.g. Path('runs/mwir2lwir')
        split       : 'train' or 'test'

        Returns
        -------
        mean PSNR over the fixed samples, or None on failure.
        """
        cache = self._cache[split]
        split_dir = output_dir / f"{split}-results" / f"step_{step:07d}"
        split_dir.mkdir(parents=True, exist_ok=True)

        mwir_batch = cache["mwir"].to(self.device)    # (N, C, H, W)
        lwir_real  = cache["lwir"]                    # (N, C, H, W) — CPU

        # ── Generate ──────────────────────────────────────────────
        try:
            with torch.no_grad():
                gen_batch = generate_fn(mwir_batch).cpu()   # (N, C, H, W)
        except Exception as exc:
            print(f"[Visualizer] generate_fn raised at step {step}: {exc}")
            return None

        # ── Per-sample files + PSNR ───────────────────────────────
        psnr_vals: List[float] = []
        grid_rows: List[List[np.ndarray]] = []

        for i, (idx, scene_name) in enumerate(zip(cache["indices"], cache["paths"])):
            mwir_np = _tensor_to_np(mwir_batch[i])
            gen_np  = _tensor_to_np(gen_batch[i])
            real_np = _tensor_to_np(lwir_real[i])

            stem = f"sample_{i:02d}_{scene_name}"

            # .npy — raw float32 in network output range (≈ [-1, 1])
            np.save(split_dir / f"{stem}_mwir.npy",  mwir_np.astype(np.float32))
            np.save(split_dir / f"{stem}_gen.npy",   gen_np.astype(np.float32))
            np.save(split_dir / f"{stem}_real.npy",  real_np.astype(np.float32))

            # Individual .png files
            if _PIL_AVAILABLE:
                for tag, arr in [("mwir", mwir_np), ("gen", gen_np), ("real", real_np)]:
                    Image.fromarray(_to_uint8(arr), mode="L").save(
                        split_dir / f"{stem}_{tag}.png"
                    )

            # PSNR in [-1,1] space (consistent with training metrics)
            mse = float(np.mean((gen_np - real_np) ** 2))
            p = 20 * math.log10(2.0 / math.sqrt(mse)) if mse > 1e-10 else float("inf")
            psnr_vals.append(p)
            grid_rows.append([mwir_np, gen_np, real_np])

        mean_psnr = float(np.mean(psnr_vals))

        # ── Side-by-side overview PNG ──────────────────────────────
        if _PIL_AVAILABLE:
            try:
                grid = _make_grid_png(
                    grid_rows,
                    col_titles=["MWIR (input)", "Generated LWIR", "Real LWIR"],
                )
                grid.save(split_dir / "overview.png")
            except Exception as exc:
                print(f"[Visualizer] overview PNG failed: {exc}")

        # ── PSNR log ──────────────────────────────────────────────
        psnr_record = {
            "step":       step,
            "split":      split,
            "mean_psnr":  round(mean_psnr, 3),
            "per_sample": {
                cache["paths"][i]: round(psnr_vals[i], 3)
                for i in range(len(psnr_vals))
            },
        }
        with open(split_dir / "psnr.json", "w") as f:
            json.dump(psnr_record, f, indent=2)

        print(
            f"[Visualizer] {split:5s} step {step:>7,d} | "
            f"PSNR {mean_psnr:6.2f} dB | → {split_dir}"
        )
        return mean_psnr

    def save_both(
        self,
        step: int,
        generate_fn: Callable[[torch.Tensor], torch.Tensor],
        output_dir: Path,
    ) -> Dict[str, Optional[float]]:
        """Save train and test splits in one call. Returns {split: psnr}."""
        return {
            split: self.save(step, generate_fn, output_dir, split)
            for split in ("train", "test")
        }

    def is_best(self, psnr_val: Optional[float], split: str) -> bool:
        """
        Returns True (and updates internal tracker) if psnr_val is a
        new best for this split.  Pass to trainer to decide whether to
        save best-weights checkpoint.
        """
        if psnr_val is not None and psnr_val > self._best_psnr[split]:
            self._best_psnr[split] = psnr_val
            return True
        return False

    @property
    def best_psnr(self) -> Dict[str, float]:
        return dict(self._best_psnr)
