"""
Normalization diagnostic script.

Checks:
  1. What range your pre-normalized .npy files actually contain
  2. What range the dataset outputs after its internal percentile_normalize
  3. Whether MWIR and LWIR share the same normalization (or are independent)
  4. Recommendation on what to do

Run from project root:
    python diag_normalization.py \
        --data_root data/ir_pairs \
        --file_ext npy \
        --n_samples 100
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from data.dataset import MWIRLWIRDataset, percentile_normalize


def load_raw(path: Path) -> np.ndarray:
    """Load a .npy file without any normalisation."""
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    return arr


def stats(arr: np.ndarray, name: str):
    print(f"  {name:30s}  min={arr.min():10.4f}  max={arr.max():10.4f}"
          f"  mean={arr.mean():8.4f}  std={arr.std():7.4f}"
          f"  p2={np.percentile(arr, 2):8.4f}  p98={np.percentile(arr, 98):8.4f}")


def run(data_root: str, file_ext: str, n_samples: int, val_frac: float):
    root = Path(data_root)
    # mwir_dir = root / 'MWIR-normalized-globalstats'
    # lwir_dir = root / 'LWIR-normalized-globalstats'
    mwir_dir = root / 'MWIR'
    lwir_dir = root / 'LWIR'
    

    mwir_files = sorted(f.stem for f in mwir_dir.iterdir()
                   if f.suffix.lower() in ('.npy', '.npz', '.tif', '.png'))[:n_samples]
    lwir_files = sorted(f.stem for f in lwir_dir.iterdir()
                   if f.suffix.lower() in ('.npy', '.npz', '.tif', '.png'))[:n_samples]

    print(f"\n{'='*70}")
    print(f"  NORMALISATION DIAGNOSTIC  ({len(mwir_files)} samples)")
    print(f"{'='*70}\n")

    # ── 1. Raw file contents ───────────────────────────────────────
    print("1. RAW FILE VALUES (before any dataset processing):")
    all_mwir_raw, all_lwir_raw = [], []
    for mwir_stem, lwir_stem in zip(mwir_files, lwir_files):
        try:
            m = load_raw(mwir_dir / f'{mwir_stem}.{file_ext}')
            l = load_raw(lwir_dir / f'{lwir_stem}.{file_ext}')
            all_mwir_raw.append(m.ravel())
            all_lwir_raw.append(l.ravel())
        except Exception:
            continue

    mwir_raw = np.concatenate(all_mwir_raw)
    lwir_raw = np.concatenate(all_lwir_raw)
    stats(mwir_raw, 'MWIR raw')
    stats(lwir_raw, 'LWIR raw')

    # Detect pre-normalised data
    mwir_in_unit = mwir_raw.min() >= -1.05 and mwir_raw.max() <= 1.05
    lwir_in_unit = lwir_raw.min() >= -1.05 and lwir_raw.max() <= 1.05
    print()
    if mwir_in_unit and lwir_in_unit:
        print("  ⚠ Both MWIR and LWIR raw files are already in [-1, 1].")
        print("    This means your pre-normalisation has been applied to the .npy files.")
        print("    The dataset will apply percentile_normalize AGAIN on top of this.")
    elif not mwir_in_unit and not lwir_in_unit:
        print("  ✓ Raw files contain physical values (not pre-normalised).")
        print("    The dataset's percentile_normalize will be the primary normalisation.")
    else:
        print("  ⚠ MWIR and LWIR have different normalisation states — check consistency.")

    # ── 2. Dataset output (after internal percentile_normalize) ───
    print()
    print("2. DATASET OUTPUT VALUES (after dataset's internal percentile_normalize):")
    ds = MWIRLWIRDataset(
        root=data_root, split='train', image_size=256,
        augment=False, val_frac=val_frac, file_ext=file_ext
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False,
                        num_workers=0, drop_last=False)
    all_mwir_ds, all_lwir_ds = [], []
    for i, batch in enumerate(loader):
        if i * 8 >= n_samples:
            break
        all_mwir_ds.append(batch['mwir'].numpy().ravel())
        all_lwir_ds.append(batch['lwir'].numpy().ravel())

    mwir_ds = np.concatenate(all_mwir_ds)
    lwir_ds = np.concatenate(all_lwir_ds)
    stats(mwir_ds, 'MWIR after dataset')
    stats(lwir_ds, 'LWIR after dataset')

    # ── 3. Per-scene range analysis ────────────────────────────────
    print()
    print("3. PER-SCENE RANGE ANALYSIS (did pre-norm collapse contrast?):")
    raw_ranges_mwir, raw_ranges_lwir = [], []
    ds_ranges_mwir, ds_ranges_lwir   = [], []

    for mwir_stem, lwir_stem in zip(mwir_files[:20], lwir_files[:20]):
        try:
            m_r = load_raw(mwir_dir / f'{mwir_stem}.{file_ext}')
            l_r = load_raw(lwir_dir / f'{lwir_stem}.{file_ext}')
            raw_ranges_mwir.append(m_r.max() - m_r.min())
            raw_ranges_lwir.append(l_r.max() - l_r.min())
        except Exception:
            continue

    print(f"  MWIR per-scene value range in raw files:")
    print(f"    mean = {np.mean(raw_ranges_mwir):.4f}  "
          f"min = {np.min(raw_ranges_mwir):.4f}  "
          f"max = {np.max(raw_ranges_mwir):.4f}")
    print(f"  LWIR per-scene value range in raw files:")
    print(f"    mean = {np.mean(raw_ranges_lwir):.4f}  "
          f"min = {np.min(raw_ranges_lwir):.4f}  "
          f"max = {np.max(raw_ranges_lwir):.4f}")

    # ── 4. Double-normalisation check ─────────────────────────────
    print()
    print("4. DOUBLE-NORMALISATION CHECK:")
    if mwir_in_unit:
        # Simulate what percentile_normalize does to already-normalised data
        sample_raw = load_raw(mwir_dir / f'{mwir_files[0]}.{file_ext}').squeeze()
        sample_renorm = percentile_normalize(sample_raw)
        change = np.abs(sample_raw - sample_renorm).mean()
        print(f"  Mean absolute change from second normalisation on MWIR: {change:.5f}")
        if change < 0.01:
            print("  ✓ Second normalisation is a near-no-op: your data fills [-1,1] tightly.")
            print("    The global normalization and per-image normalization are effectively equivalent.")
        elif change < 0.1:
            print("  ⚠ Mild double-normalisation effect. Global norm is partially overridden.")
        else:
            print("  ✗ Strong double-normalisation effect. Global norm is substantially overridden.")
            print("    Your global normalization intent is NOT being preserved in training.")

    # ── 5. MWIR/LWIR scale consistency ─────────────────────────────
    print()
    print("5. MWIR / LWIR SCALE CONSISTENCY:")
    mwir_scene_mean = np.array([
        load_raw(mwir_dir / f'{s}.{file_ext}').mean() for s in mwir_files[:30]
    ])
    lwir_scene_mean = np.array([
        load_raw(lwir_dir / f'{s}.{file_ext}').mean() for s in lwir_files[:30]
    ])
    corr = np.corrcoef(mwir_scene_mean, lwir_scene_mean)[0, 1]
    print(f"  Scene-level mean correlation (MWIR vs LWIR): r = {corr:.4f}")
    if corr > 0.7:
        print("  ✓ Strong cross-band mean correlation — physically expected.")
    elif corr > 0.3:
        print("  ⚠ Moderate correlation. Normalisation may have partially decoupled the bands.")
    else:
        print("  ✗ Weak correlation. Independent per-band normalisation has likely decoupled "
              "the radiometric relationship — this makes the synthesis task harder.")

    # ── 6. Recommendation ─────────────────────────────────────────
    print()
    print("=" * 70)
    print("  RECOMMENDATION")
    print("=" * 70)

    if not mwir_in_unit and not lwir_in_unit:
        print("""
  Your raw files contain physical values. The dataset's per-image percentile
  normalisation is the only normalisation applied. This is the intended setup.
  No action needed.
""")
    elif mwir_in_unit and lwir_in_unit and change < 0.05:
        print("""
  Your files are pre-normalised AND the second normalisation is nearly a no-op.
  This means your global normalisation is tight (close to p2-p98 of each scene).
  Functionally this is equivalent to per-image normalisation.
  
  The VAE has already been trained with this setup. For Stage 2 DiT training,
  keep exactly the same preprocessing — consistency matters more than which
  specific normalisation you use.
  
  One recommendation: verify MWIR and LWIR were normalised independently
  (separate global_min/max per band), not with a shared value. Shared
  normalisation would distort the cross-band relationship the DiT needs to learn.
""")
    else:
        print("""
  ⚠ Your global normalisation is being partially overridden by the dataset's
  per-image normalisation. The VAE has learned with this double-normalisation
  in place. For Stage 2, keep the same setup — changing it now would be a
  distribution shift for the already-trained VAE.
  
  For future training runs consider one of:
    A) Store raw physical values in .npy files and let the dataset normalise.
    B) Pre-normalise with per-image p2-p98 stretch and disable dataset's internal
       normalisation (set use_precomputed_norm=True in dataset config).
    C) Keep global normalisation but set dataset's augment=False and verify the
       second normalisation is a no-op (change < 0.01).
""")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',  required=True)
    parser.add_argument('--file_ext',   default='npy')
    parser.add_argument('--n_samples',  type=int, default=100)
    parser.add_argument('--val_frac',   type=float, default=0.1)
    args = parser.parse_args()
    run(args.data_root, args.file_ext, args.n_samples, args.val_frac)
