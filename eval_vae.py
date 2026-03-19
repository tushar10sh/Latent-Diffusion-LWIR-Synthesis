"""
VAE Evaluation — Stage 1 quality diagnostics for MWIR→LWIR LDM pipeline.

Checks:
  1. Reconstruction fidelity    PSNR, SSIM, L1, per-sample scatter
  2. Latent calibration         z mean/std after scale_factor — must be ~N(0,1)
  3. Posterior collapse         KL divergence distribution — healthy range: 1–50 nats
  4. Frequency fidelity         log-PSD error: does the VAE preserve IR edge structure?
  5. Spatial error maps         where does the VAE fail spatially across your dataset?
  6. Worst/best reconstructions save the hardest and easiest scenes for inspection
  7. Go/no-go verdict           single summary with clear pass/fail per criterion

Usage:
    python eval_vae.py \
        --vae_ckpt runs/mwir2lwir_ldm/stage1_vae/vae_final.pt \
        --data_root data/ir_pairs \
        --output_dir eval/vae \
        --split val \
        --n_samples 200
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── optional plotting (graceful fallback if matplotlib absent) ───────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _PLOT = True
except ImportError:
    _PLOT = False
    print("[eval_vae] matplotlib not available — skipping plots, saving metrics only.")

from models.ldm.vae import IRVAE
from data.dataset import MWIRLWIRDataset, percentile_normalize


# ═════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ═════════════════════════════════════════════════════════════════════════════

def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 2.0) -> float:
    mse = F.mse_loss(pred, target).item()
    return 20 * math.log10(max_val / math.sqrt(mse)) if mse > 1e-10 else float('inf')


def ssim(pred: torch.Tensor, target: torch.Tensor,
         window: int = 11, C1: float = 0.0001, C2: float = 0.0009) -> float:
    mu_p  = F.avg_pool2d(pred,   window, 1, window // 2)
    mu_t  = F.avg_pool2d(target, window, 1, window // 2)
    sig_p  = F.avg_pool2d(pred**2,         window, 1, window // 2) - mu_p**2
    sig_t  = F.avg_pool2d(target**2,       window, 1, window // 2) - mu_t**2
    sig_pt = F.avg_pool2d(pred * target,   window, 1, window // 2) - mu_p * mu_t
    num = (2 * mu_p * mu_t + C1) * (2 * sig_pt + C2)
    den = (mu_p**2 + mu_t**2 + C1) * (sig_p + sig_t + C2)
    return float((num / den).mean())


def log_psd_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """L1 error between log power spectral densities."""
    psd_p = torch.fft.fft2(pred,   norm='ortho').abs().pow(2)
    psd_t = torch.fft.fft2(target, norm='ortho').abs().pow(2)
    return float(F.l1_loss(torch.log1p(psd_p), torch.log1p(psd_t)))


def radially_averaged_psd(arr: np.ndarray, n_bins: int = 64) -> tuple:
    """
    Compute radially-averaged power spectral density.
    Returns (frequencies, power) arrays for plotting.
    Useful for diagnosing which spatial frequency range the VAE blurs.
    """
    H, W   = arr.shape
    fft2   = np.fft.fft2(arr)
    psd    = np.abs(fft2) ** 2
    psd    = np.fft.fftshift(psd)

    cy, cx = H // 2, W // 2
    y_idx, x_idx = np.mgrid[-cy:H - cy, -cx:W - cx]
    r      = np.sqrt(y_idx**2 + x_idx**2).astype(int)

    max_r  = min(cy, cx)
    freqs  = np.arange(max_r)
    power  = np.array([psd[r == ri].mean() if (r == ri).any() else 0.0
                       for ri in range(max_r)])
    return freqs, power


def spatial_error_map(recons: list, originals: list) -> np.ndarray:
    """
    Compute mean absolute error at each pixel position across all samples.
    Reveals systematic spatial regions the VAE handles poorly.
    """
    errs = [np.abs(r - o) for r, o in zip(recons, originals)]
    return np.stack(errs).mean(axis=0)


# ═════════════════════════════════════════════════════════════════════════════
# Per-sample evaluation
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_batch(vae: IRVAE, batch_lwir: torch.Tensor, device: torch.device):
    """
    Run one batch through the VAE encode→decode cycle.
    Returns a dict of per-batch metrics and intermediate tensors.
    """
    lwir = batch_lwir.to(device)

    posterior  = vae.encode(lwir)
    z_raw      = posterior.mode()                        # (B, C, H, W)
    # Per-channel normalisation via _get_affine (handles both scalar and list formats)
    mean_t, scale_t = vae._get_affine(z_raw.device)
    z_scaled   = (z_raw - mean_t) * scale_t             # what DiT sees

    # Reconstruction via the corrected decode path
    recon = vae.decode(z_scaled)

    # Metrics
    psnr_val  = psnr(recon, lwir)
    ssim_val  = ssim(recon, lwir)
    l1_val    = float(F.l1_loss(recon, lwir))
    psd_err   = log_psd_error(recon, lwir)

    # Latent statistics (DiT-space: should be ~N(0,1))
    z_mean = float(z_scaled.mean())
    z_std  = float(z_scaled.std())
    z_min  = float(z_scaled.min())
    z_max  = float(z_scaled.max())

    z_channel_std = z_scaled.std(dim=(0, 2, 3)).cpu().tolist()

    # KL per latent DIMENSION (not summed) — matches the training log value
    mu     = posterior.mean
    logvar = posterior.logvar
    kl_elementwise = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)   # (B, C, H, W)
    n_dims = float(kl_elementwise.shape[1] * kl_elementwise.shape[2]
                   * kl_elementwise.shape[3])
    kl_per_dim_per_sample = kl_elementwise.mean(dim=(1, 2, 3))        # (B,) mean over dims
    kl_total_per_sample   = kl_elementwise.sum(dim=(1, 2, 3))         # (B,) sum over dims

    return {
        'psnr':                psnr_val,
        'ssim':                ssim_val,
        'l1':                  l1_val,
        'psd_err':             psd_err,
        'z_mean':              z_mean,
        'z_std':               z_std,
        'z_min':               z_min,
        'z_max':               z_max,
        'z_channel_std':       z_channel_std,
        'kl_per_dim':          float(kl_per_dim_per_sample.mean()),    # matches training log
        'kl_total':            float(kl_total_per_sample.mean()),      # for reference
        'n_latent_dims':       int(n_dims),
        'kl_per_dim_samples':  kl_per_dim_per_sample.cpu().tolist(),
        'recon':               recon.cpu(),
        'original':            lwir.cpu(),
        'z_scaled':            z_scaled.cpu(),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════════

def _u8(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)


def plot_reconstruction_grid(originals, reconstructions, paths, output_path, n=8):
    """Side-by-side original vs reconstruction for n samples."""
    if not _PLOT:
        return
    n = min(n, len(originals))
    fig, axes = plt.subplots(2, n, figsize=(n * 2.5, 5.5))
    fig.suptitle('VAE Reconstruction Quality\nTop: Original LWIR   Bottom: Reconstructed',
                 fontsize=11, y=1.01)
    for i in range(n):
        axes[0, i].imshow(_u8(originals[i].squeeze()), cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(paths[i][:12], fontsize=7)
        axes[0, i].axis('off')
        axes[1, i].imshow(_u8(reconstructions[i].squeeze()), cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [plot] reconstruction grid → {output_path}")


def plot_latent_distribution(z_all: np.ndarray, output_path, vae=None):
    """
    Per-channel histogram of z_DiT values — each channel should be ~N(0,1)
    after calibration. Shows each channel separately so multimodal structure
    is visible per-channel rather than hidden by global averaging.
    """
    if not _PLOT:
        return

    n_ch = z_all.shape[1] if z_all.ndim >= 4 else 1
    ncols = n_ch + 1      # one plot per channel + one summary
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 3.5, 4))
    xs = np.linspace(-4, 4, 200)
    normal_pdf = np.exp(-xs**2 / 2) / math.sqrt(2 * math.pi)

    # Per-channel histograms
    for c in range(n_ch):
        if z_all.ndim >= 4:
            vals = z_all[:, c, :, :].ravel()
        else:
            vals = z_all.ravel()
        axes[c].hist(vals, bins=80, density=True, alpha=0.65,
                     color=f'C{c}', label=f'ch{c}')
        axes[c].plot(xs, normal_pdf, 'k--', lw=1.5, alpha=0.7, label='N(0,1)')
        axes[c].set_title(
            f'Channel {c}\nmean={vals.mean():.3f}  std={vals.std():.3f}',
            fontsize=9
        )
        axes[c].set_xlim(-5, 5)
        axes[c].set_xlabel('z_DiT value')
        if c == 0:
            axes[c].set_ylabel('density')
        axes[c].legend(fontsize=7)

    # Summary: all channels overlaid
    ax_sum = axes[-1]
    for c in range(n_ch):
        if z_all.ndim >= 4:
            vals = z_all[:, c, :, :].ravel()
        else:
            vals = z_all.ravel()
        ax_sum.hist(vals, bins=80, density=True, alpha=0.4, label=f'ch{c}')
    ax_sum.plot(xs, normal_pdf, 'k--', lw=2, label='N(0,1)')
    ax_sum.set_title('All channels overlaid', fontsize=9)
    ax_sum.set_xlim(-5, 5)
    ax_sum.set_xlabel('z_DiT value')
    ax_sum.legend(fontsize=7)

    # Per-channel std bar chart note
    if z_all.ndim >= 4:
        ch_stds = z_all.std(axis=(0, 2, 3))
    else:
        ch_stds = np.array([z_all.std()])
    stds_str = '  '.join(f'ch{c}:{s:.3f}' for c, s in enumerate(ch_stds))
    fig.suptitle(
        f'Latent distribution after per-channel calibration\n'
        f'Per-channel std: {stds_str}\n'
        f'Multimodal shape is expected for IR (different land cover clusters)',
        fontsize=9, y=1.02
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [plot] latent distribution → {output_path}")


def plot_kl_distribution(kl_values: list, output_path):
    """KL divergence histogram. Healthy range: 1–50 nats."""
    if not _PLOT:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(kl_values, bins=50, color='darkorange', alpha=0.75, edgecolor='none')
    ax.axvline(1.0,  color='green', linestyle='--', lw=2, label='min healthy (1 nat)')
    ax.axvline(50.0, color='red',   linestyle='--', lw=2, label='max healthy (50 nats)')
    ax.set_title('KL Divergence Distribution\n'
                 'Too low → posterior collapse (encoder ignoring input)\n'
                 'Too high → encoder not regularised (latents off-manifold)')
    ax.set_xlabel('KL per sample (nats)')
    ax.set_ylabel('count')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [plot] KL distribution → {output_path}")


def plot_psd_comparison(orig_psds, recon_psds, output_path):
    """Mean radially-averaged PSD: original vs reconstructed."""
    if not _PLOT or not orig_psds:
        return
    min_len = min(len(p) for p in orig_psds + recon_psds)
    orig_mean  = np.stack([p[:min_len] for p in orig_psds]).mean(axis=0)
    recon_mean = np.stack([p[:min_len] for p in recon_psds]).mean(axis=0)
    freqs      = np.arange(min_len)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].semilogy(freqs[1:], orig_mean[1:],  label='Original',       lw=2)
    axes[0].semilogy(freqs[1:], recon_mean[1:], label='Reconstructed',  lw=2, linestyle='--')
    axes[0].set_title('Radially-averaged PSD (log scale)\nDivergence = frequency range VAE cannot reconstruct')
    axes[0].set_xlabel('Spatial frequency (cycles/image)')
    axes[0].set_ylabel('Power')
    axes[0].legend()

    ratio = recon_mean[1:] / (orig_mean[1:] + 1e-10)
    axes[1].plot(freqs[1:], ratio, lw=2, color='darkorange')
    axes[1].axhline(1.0, color='gray', linestyle='--', lw=1)
    axes[1].set_title('PSD Ratio (Reconstructed / Original)\n<1 = VAE attenuates; >1 = VAE amplifies')
    axes[1].set_xlabel('Spatial frequency')
    axes[1].set_ylabel('Ratio')
    axes[1].set_ylim(0, 3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [plot] PSD comparison → {output_path}")


def plot_spatial_error_map(error_map: np.ndarray, output_path):
    """Heatmap of mean absolute reconstruction error at each pixel position."""
    if not _PLOT:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(error_map.squeeze(), cmap='hot')
    plt.colorbar(im, ax=ax, label='Mean |original - recon|')
    ax.set_title('Spatial Error Map\n(averaged across all evaluation samples)')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [plot] spatial error map → {output_path}")


def plot_scatter(psnr_vals, ssim_vals, output_path):
    """PSNR vs SSIM scatter — outliers = scenes the VAE handles poorly."""
    if not _PLOT:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(psnr_vals, ssim_vals, alpha=0.5, s=20, color='steelblue')

    # Annotate the 5 worst-PSNR samples
    worst_idx = np.argsort(psnr_vals)[:5]
    for i in worst_idx:
        ax.annotate(f'#{i}', (psnr_vals[i], ssim_vals[i]), fontsize=7,
                    xytext=(3, 3), textcoords='offset points', color='red')

    ax.set_xlabel('PSNR (dB)')
    ax.set_ylabel('SSIM')
    ax.set_title('Per-sample PSNR vs SSIM\nRed = worst 5 samples by PSNR')
    ax.axvline(25, color='orange', linestyle='--', lw=1, label='25 dB (acceptable floor)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [plot] PSNR/SSIM scatter → {output_path}")


def save_worst_best(originals, reconstructions, paths, psnr_vals, output_dir, n=4):
    """
    Save PNG of the n worst and n best reconstructions for visual inspection.

    psnr_vals must be the subset corresponding to originals/reconstructions,
    not the full eval set (which may be larger if images are capped at 64).
    """
    if not _PLOT:
        return
    n_avail   = len(originals)
    n         = min(n, n_avail)
    if n == 0:
        return

    idx_sorted = np.argsort(psnr_vals[:n_avail])   # only index into saved images
    worst_idx  = idx_sorted[:n]
    best_idx   = idx_sorted[-n:][::-1]

    for label, indices in [('worst', worst_idx), ('best', best_idx)]:
        n_cols = len(indices)
        # squeeze=False guarantees axes is always 2D, even when n_cols==1
        fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 3, 6), squeeze=False)
        fig.suptitle(f'{label.capitalize()} {n_cols} reconstructions by PSNR', fontsize=11)
        for col, i in enumerate(indices):
            p = psnr_vals[i]
            axes[0, col].imshow(_u8(originals[i].squeeze()), cmap='gray')
            axes[0, col].set_title(f'{paths[i][:10]}\nPSNR={p:.1f}dB', fontsize=8)
            axes[0, col].axis('off')
            axes[1, col].imshow(_u8(reconstructions[i].squeeze()), cmap='gray')
            axes[1, col].axis('off')
        axes[0, 0].set_ylabel('Original',      fontsize=9)
        axes[1, 0].set_ylabel('Reconstructed', fontsize=9)
        plt.tight_layout()
        out = output_dir / f'{label}_reconstructions.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [plot] {label} reconstructions → {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Go/No-Go verdict
# ═════════════════════════════════════════════════════════════════════════════

# Thresholds — calibrated for single-channel thermal IR with f=4 compression
# KL thresholds are PER LATENT DIMENSION (matching the training log value)
# not summed over all dimensions (which would give ~16,384× larger numbers)
THRESHOLDS = {
    'psnr_mean':         {'min': 28.0,  'unit': 'dB',    'desc': 'Mean reconstruction PSNR'},
    'psnr_p10':          {'min': 24.0,  'unit': 'dB',    'desc': '10th-percentile PSNR (worst 10%)'},
    'ssim_mean':         {'min': 0.85,  'unit': '',      'desc': 'Mean SSIM'},
    'z_std_mean':        {'min': 0.85,  'max': 1.15,     'unit': '', 'desc': 'Mean latent std after calibration (target 1.0)'},
    'z_mean_abs':        {'max': 0.10,  'unit': '',      'desc': 'Abs latent mean after calibration (target 0.0)'},
    'kl_per_dim_mean':   {'min': 0.1,   'max': 15.0,     'unit': 'nats/dim',
                          'desc': 'Mean KL per latent dim (matches training log; 2-10 is ideal)'},
    'psd_err_mean':      {'max': 0.15,  'unit': '',      'desc': 'Mean log-PSD error (frequency fidelity)'},
}


def verdict(metrics: dict) -> dict:
    results = {}
    overall_pass = True

    for key, spec in THRESHOLDS.items():
        val = metrics.get(key)
        if val is None:
            results[key] = {'value': None, 'pass': None, 'note': 'not computed'}
            continue
        passed = True
        note_parts = []
        if 'min' in spec and val < spec['min']:
            passed = False
            note_parts.append(f"below minimum {spec['min']}")
        if 'max' in spec and val > spec['max']:
            passed = False
            note_parts.append(f"above maximum {spec['max']}")
        note = '; '.join(note_parts) if note_parts else 'OK'
        results[key] = {
            'value': round(float(val), 4),
            'unit':  spec.get('unit', ''),
            'pass':  passed,
            'desc':  spec['desc'],
            'note':  note,
        }
        if not passed:
            overall_pass = False

    results['_overall_pass'] = overall_pass
    return results


def print_verdict(v: dict):
    print('\n' + '=' * 65)
    print('  VAE EVALUATION VERDICT')
    print('=' * 65)
    for key, info in v.items():
        if key.startswith('_'):
            continue
        if info.get('pass') is None:
            status = '  ?  '
        elif info['pass']:
            status = ' PASS'
        else:
            status = ' FAIL'
        val_str = f"{info['value']}{info.get('unit','')}" if info['value'] is not None else 'N/A'
        print(f"  {status}  {info['desc']:<45s}  {val_str:>10s}   {info['note']}")
    print('=' * 65)
    if v.get('_overall_pass'):
        print('  ✓ OVERALL: VAE is ready for Stage 2 DiT training.')
    else:
        print('  ✗ OVERALL: VAE has issues. See FAIL rows above before proceeding.')
        print()
        print('  Remediation guide:')
        if v.get('psnr_mean', {}).get('pass') is False:
            print('    PSNR low   → train more steps (add 20k); check kl_weight schedule')
        if v.get('z_std_mean', {}).get('pass') is False:
            print('    z_std off  → rerun vae.compute_scale_factor(train_loader) and re-save')
        if v.get('kl_per_dim_mean', {}).get('pass') is False:
            kl = v.get('kl_per_dim_mean', {}).get('value', 0)
            if kl < 0.1:
                print('    KL/dim low  → posterior collapse; reduce kl_weight_max or add free-bits')
            else:
                print('    KL/dim high → reduce kl_weight_max; increase warmup steps')
        if v.get('psd_err_mean', {}).get('pass') is False:
            print('    PSD error  → VAE blurs high-frequency edges; add perceptual loss weight')
    print('=' * 65 + '\n')


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def run_eval(
    vae_ckpt:    str,
    data_root:   str,
    output_dir:  str,
    split:       str  = 'val',
    n_samples:   int  = 200,
    batch_size:  int  = 8,
    num_workers: int  = 4,
    file_ext:    str  = 'npy',
    val_frac:    float = 0.1,
    device_str:  str  = 'cuda',
):
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    out    = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load VAE ─────────────────────────────────────────────────
    ckpt = torch.load(vae_ckpt, map_location=device)
    cfg  = ckpt.get('config', {})
    vae  = IRVAE(
        in_channels  = cfg.get('lwir_channels', 1),
        ch           = cfg.get('vae_ch', 128),
        ch_mult      = tuple(cfg.get('vae_ch_mult', [1, 2, 4])),
        num_res_blocks = cfg.get('vae_num_res_blocks', 2),
        z_channels   = cfg.get('z_channels', 4),
    ).to(device)
    vae.load_state_dict(ckpt['vae'])
    vae.scale_factor = ckpt.get('scale_factor', 1.0)
    vae.latent_mean  = ckpt.get('latent_mean',  0.0)
    vae.eval()

    # Detect old scalar checkpoint format and warn
    is_old_format = not isinstance(vae.scale_factor, (list, tuple))
    if is_old_format:
        import warnings
        warnings.warn(
            "\n[eval_vae] *** OLD CHECKPOINT FORMAT DETECTED ***\n"
            "  scale_factor and latent_mean are scalars, not per-channel lists.\n"
            "  This checkpoint was saved BEFORE compute_scale_factor() was fixed.\n"
            "  z_std and z_mean metrics will be WRONG in this run.\n"
            "  Fix: run the recalibration script, then pass --vae_ckpt ...recal.pt",
            stacklevel=2,
        )

    # Format latent_mean for display — handles both list[float] and scalar float
    if isinstance(vae.latent_mean, (list, tuple)):
        lm_str = '[' + ', '.join(f'{v:.4f}' for v in vae.latent_mean) + ']'
        sf_str = '[' + ', '.join(f'{v:.4f}' for v in vae.scale_factor) + ']'
    else:
        lm_str = f'{vae.latent_mean:.5f}  ← SCALAR (old format, run recalibration)'
        sf_str = f'{vae.scale_factor:.5f}  ← SCALAR (old format)'

    print(
        f"[eval_vae] Loaded VAE\n"
        f"           scale_factor = {sf_str}\n"
        f"           latent_mean  = {lm_str}\n"
        f"           checkpoint   = {vae_ckpt}"
    )

    # ── Dataset ───────────────────────────────────────────────────
    ds = MWIRLWIRDataset(
        root       = data_root,
        split      = split,
        image_size = cfg.get('image_size', 256),
        augment    = False,
        val_frac   = val_frac,
        file_ext   = file_ext,
    )
    n_samples = min(n_samples, len(ds))
    # Deterministic subset: evenly spaced
    indices = [int(i * len(ds) / n_samples) for i in range(n_samples)]
    subset  = torch.utils.data.Subset(ds, indices)
    loader  = DataLoader(subset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
    print(f"[eval_vae] Evaluating {n_samples} samples from '{split}' split")

    # ── Collect results ───────────────────────────────────────────
    all_psnr, all_ssim, all_l1      = [], [], []
    all_psd_err                     = []
    all_kl                          = []
    all_z_stds                      = []
    all_z_means                     = []
    orig_images, recon_images       = [], []
    orig_psd_curves, recon_psd_curves = [], []
    paths                           = []

    for batch in loader:
        lwir = batch['lwir']
        res  = evaluate_batch(vae, lwir, device)

        B = lwir.shape[0]
        all_psnr.extend([psnr(res['recon'][i:i+1], res['original'][i:i+1])
                         for i in range(B)])
        all_ssim.extend([ssim(res['recon'][i:i+1], res['original'][i:i+1])
                         for i in range(B)])
        all_l1.append(res['l1'])
        all_psd_err.append(res['psd_err'])
        all_kl.extend(res['kl_per_dim_samples'])        # per-dimension KL
        all_z_stds.append(res['z_std'])
        all_z_means.append(abs(res['z_mean']))
        paths.extend(batch.get('path', [str(i) for i in range(B)]))

        # Collect images for plots (up to 64 to keep memory manageable)
        if len(orig_images) < 64:
            for i in range(B):
                orig_images.append(res['original'][i].numpy())
                recon_images.append(res['recon'][i].numpy())
                o_np = res['original'][i].squeeze().numpy()
                r_np = res['recon'][i].squeeze().numpy()
                _, op = radially_averaged_psd(o_np)
                _, rp = radially_averaged_psd(r_np)
                orig_psd_curves.append(op)
                recon_psd_curves.append(rp)

    # Collect all z values for distribution plot (one full batch)
    with torch.no_grad():
        sample_lwir = next(iter(loader))['lwir'].to(device)
        # Use encode_to_dit — applies per-channel (mean, scale) correctly
        # whether scale_factor is a scalar (old ckpt) or list (new ckpt)
        z_all_np = vae.encode_to_dit(sample_lwir).cpu().numpy()

    # ── Aggregate metrics ─────────────────────────────────────────
    metrics = {
        'psnr_mean':        float(np.mean(all_psnr)),
        'psnr_std':         float(np.std(all_psnr)),
        'psnr_min':         float(np.min(all_psnr)),
        'psnr_p10':         float(np.percentile(all_psnr, 10)),
        'psnr_median':      float(np.median(all_psnr)),
        'ssim_mean':        float(np.mean(all_ssim)),
        'ssim_std':         float(np.std(all_ssim)),
        'l1_mean':          float(np.mean(all_l1)),
        'psd_err_mean':     float(np.mean(all_psd_err)),
        'kl_per_dim_mean':  float(np.mean(all_kl)),          # matches training log
        'kl_per_dim_p5':    float(np.percentile(all_kl, 5)),
        'kl_per_dim_p95':   float(np.percentile(all_kl, 95)),
        'z_std_mean':       float(np.mean(all_z_stds)),
        'z_mean_abs':       float(np.mean(all_z_means)),
        'n_samples':        n_samples,
        'split':            split,
        'vae_ckpt':         str(vae_ckpt),
        'scale_factor':     vae.scale_factor,   # list[float] or float — both JSON-serialisable
        'latent_mean':      vae.latent_mean,    # list[float] or float
    }

    # ── Print summary ─────────────────────────────────────────────
    n_dims = res.get('n_latent_dims', '?')
    print('\n[eval_vae] ── Summary ─────────────────────────────────────')
    print(f"  PSNR   mean={metrics['psnr_mean']:.2f} dB  "
          f"p10={metrics['psnr_p10']:.2f} dB  "
          f"min={metrics['psnr_min']:.2f} dB")
    print(f"  SSIM   mean={metrics['ssim_mean']:.4f}  std={metrics['ssim_std']:.4f}")
    print(f"  L1     mean={metrics['l1_mean']:.5f}")
    print(f"  PSD err     {metrics['psd_err_mean']:.5f}")
    print(f"  KL/dim mean={metrics['kl_per_dim_mean']:.3f} nats  "
          f"(×{n_dims} dims = {metrics['kl_per_dim_mean']*int(n_dims):.0f} total nats)")
    print(f"  Note: training log KL should match KL/dim = {metrics['kl_per_dim_mean']:.3f}")
    print(f"  z_std  mean={metrics['z_std_mean']:.4f}  (target 1.0 after calibration)")
    print(f"  z_mean abs ={metrics['z_mean_abs']:.5f}  (target 0.0 after calibration)")

    # ── Verdict ───────────────────────────────────────────────────
    v = verdict(metrics)
    print_verdict(v)
    metrics['verdict'] = v

    # ── Save metrics JSON ─────────────────────────────────────────
    with open(out / 'metrics.json', 'w') as f:
        # Remove non-serialisable keys before saving
        save_metrics = {k: v for k, v in metrics.items() if k != 'verdict'}
        save_verdict = {k: vv for k, vv in v.items()}
        json.dump({'metrics': save_metrics, 'verdict': save_verdict}, f, indent=2)
    print(f"[eval_vae] Metrics saved → {out / 'metrics.json'}")

    # ── Plots ─────────────────────────────────────────────────────
    plot_reconstruction_grid(orig_images, recon_images, paths[:len(orig_images)],
                             out / 'reconstruction_grid.png', n=8)
    plot_latent_distribution(z_all_np, out / 'latent_distribution.png', vae=vae)
    plot_kl_distribution(all_kl, out / 'kl_distribution.png')
    plot_psd_comparison(orig_psd_curves, recon_psd_curves, out / 'psd_comparison.png')
    err_map = spatial_error_map(orig_images, recon_images)
    plot_spatial_error_map(err_map, out / 'spatial_error_map.png')
    # Only pass psnr values for samples that have a saved image (orig_images capped at 64)
    n_saved = len(orig_images)
    plot_scatter(all_psnr[:n_saved], all_ssim[:n_saved], out / 'psnr_ssim_scatter.png')
    save_worst_best(orig_images, recon_images, paths[:n_saved],
                    all_psnr[:n_saved], out, n=4)

    # Save spatial error map as npy for further analysis
    np.save(out / 'spatial_error_map.npy', err_map.astype(np.float32))

    print(f"\n[eval_vae] All outputs saved to {out}/")
    return metrics


# ═════════════════════════════════════════════════════════════════════════════

def inspect_checkpoint(vae_ckpt: str):
    """
    Quick inspection of a VAE checkpoint without loading data.
    Use this to verify a checkpoint has the correct per-channel calibration
    before running the full eval.
    """
    ckpt = torch.load(vae_ckpt, map_location='cpu')
    print(f"\n[inspect] Checkpoint: {vae_ckpt}")
    print(f"  Keys: {list(ckpt.keys())}")

    sf = ckpt.get('scale_factor', 'MISSING')
    lm = ckpt.get('latent_mean',  'MISSING')

    print(f"\n  scale_factor : {sf}")
    print(f"  latent_mean  : {lm}")

    if isinstance(sf, (list, tuple)):
        print(f"\n  ✓ NEW FORMAT: per-channel calibration (len={len(sf)})")
        print(f"    scale_factor per channel: {[round(v, 5) for v in sf]}")
        print(f"    latent_mean  per channel: {[round(v, 5) for v in lm]}")
        print(f"    These will produce z_DiT with std≈1.0 and mean≈0.0 per channel.")
        print(f"\n  → Safe to use for Stage 2 DiT training.")
    elif isinstance(sf, float):
        print(f"\n  ✗ OLD FORMAT: scalar calibration")
        print(f"    scale_factor = {sf:.5f}  (was std, should be 1/std)")
        print(f"    latent_mean  = {lm}")
        print(f"    z_DiT will have std≈{sf**2:.3f} and mean≈{sf * float(lm or 0):.3f}")
        print(f"\n  → Run the recalibration script first, then use vae_final_recal.pt")
    else:
        print(f"\n  ? Unknown format for scale_factor: {type(sf)}")

    cfg = ckpt.get('config', {})
    if cfg:
        print(f"\n  Config: z_channels={cfg.get('z_channels')}, "
              f"image_size={cfg.get('image_size')}, "
              f"vae_ch={cfg.get('vae_ch')}, "
              f"vae_ch_mult={cfg.get('vae_ch_mult')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate VAE reconstruction quality')
    parser.add_argument('--vae_ckpt',   required=True,
                        help='Path to VAE checkpoint (vae_final.pt or vae_final_recal.pt)')
    parser.add_argument('--data_root',  default=None,
                        help='Dataset root (contains mwir/ and lwir/ subdirs). '
                             'Not needed for --inspect.')
    parser.add_argument('--output_dir', default='eval/vae',
                        help='Where to save evaluation outputs')
    parser.add_argument('--split',      default='val',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--n_samples',  type=int, default=200,
                        help='Number of samples to evaluate')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers',type=int, default=4)
    parser.add_argument('--file_ext',   default='npy')
    parser.add_argument('--val_frac',   type=float, default=0.1)
    parser.add_argument('--device',     default='cuda')
    parser.add_argument('--inspect',    action='store_true',
                        help='Only inspect checkpoint contents — no data needed, runs instantly.')
    args = parser.parse_args()

    if args.inspect:
        inspect_checkpoint(args.vae_ckpt)
    else:
        if args.data_root is None:
            parser.error("--data_root is required unless --inspect is used")
        run_eval(
            vae_ckpt    = args.vae_ckpt,
            data_root   = args.data_root,
            output_dir  = args.output_dir,
            split       = args.split,
            n_samples   = args.n_samples,
            batch_size  = args.batch_size,
            num_workers = args.num_workers,
            file_ext    = args.file_ext,
            val_frac    = args.val_frac,
            device_str  = args.device,
        )
