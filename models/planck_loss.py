"""
Physics-informed Planck Ratio Loss for MWIR→LWIR synthesis.

Physical background
───────────────────
For a blackbody surface at temperature T (Kelvin), the spectral radiance
at wavelength λ (µm) is given by the Planck function:

    B(λ, T) = c₁ / [λ⁵ · (exp(c₂ / (λ·T)) - 1)]

where:
    c₁ = 1.19104 × 10⁸  W·µm⁴ / (m²·sr)   (first radiation constant)
    c₂ = 14387.8         µm·K               (second radiation constant)

For a real surface with spectral emissivity ε(λ):
    L(λ, T) = ε(λ) · B(λ, T) + (1 - ε(λ)) · L_down(λ)

where L_down is the downwelling sky radiance.

The cross-band physical constraint:
    Given a surface temperature T inferred from MWIR BT, the LWIR
    radiance should be consistent with the same T (adjusted for emissivity
    difference ε_LWIR vs ε_MWIR). For unit emissivity surfaces (water,
    vegetation, most soils have ε_LWIR ≈ 0.97–0.99):

    BT_LWIR ≈ BT_MWIR  (temperatures should track each other)

    For surfaces where ε_MWIR << ε_LWIR (metallic targets, some minerals):
    BT_LWIR can exceed BT_MWIR significantly.

Loss design
───────────
We implement a soft physical constraint rather than a hard equality:

  1. Convert MWIR DN → BT_MWIR using sensor calibration coefficients.
  2. Convert generated LWIR DN → BT_LWIR.
  3. Compute the expected LWIR radiance at BT_MWIR (unit emissivity).
  4. Compute the actual LWIR radiance at BT_LWIR.
  5. Loss = soft penalty for |BT_LWIR - BT_MWIR| > allowed_delta_K,
     weighted by local MWIR confidence (high-contrast regions carry
     more weight than uniform/homogeneous areas).

This allows the model to learn emissivity differences (LWIR BT ≠ MWIR BT
is physically valid) while penalising clearly unphysical outputs such as
MWIR showing a warm target but LWIR showing it cold, or vice versa.

Configuration example (configs/ldm.json):
    "planck": {
        "mwir_wavelength_um": 4.0,
        "lwir_wavelength_um": 10.0,
        "mwir_gain":  0.01,
        "mwir_offset": -0.5,
        "lwir_gain":  0.005,
        "lwir_offset": 200.0,
        "mwir_norm_min": 88.0,
        "mwir_norm_max": 3804.0,
        "lwir_norm_min": 1622.0,
        "lwir_norm_max": 3836.0,
        "allowed_delta_K": 15.0,
        "lambda_planck": 0.05
    }

The norm_min / norm_max are the physical DN values that correspond to
[-1, 1] in your normalised representation — i.e. the global min/max
from your sensor pool. The loss internally converts normalised images
back to physical DN, then to BT.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# Planck constants
_C1 = 1.19104e8    # W·µm⁴ / (m²·sr)
_C2 = 14387.8      # µm·K


# ─────────────────────────────────────────────
# Planck function utilities
# ─────────────────────────────────────────────

def planck_radiance(wavelength_um: float, T: torch.Tensor) -> torch.Tensor:
    """
    Planck spectral radiance B(λ, T) in W/(m²·sr·µm).

    Args:
        wavelength_um: central wavelength in µm (scalar)
        T:             surface temperature in Kelvin (any shape, all > 0)
    """
    T = T.clamp(min=100.0, max=400.0)   # physical guard: 100 K → 400 K
    return _C1 / (wavelength_um**5 * (torch.exp(_C2 / (wavelength_um * T)) - 1.0))


def bt_from_radiance(radiance: torch.Tensor, wavelength_um: float) -> torch.Tensor:
    """
    Invert Planck function: convert spectral radiance to Brightness Temperature.

    BT(λ, L) = c₂ / [λ · ln(c₁ / (λ⁵ · L) + 1)]

    Args:
        radiance:      spectral radiance in W/(m²·sr·µm)
        wavelength_um: central wavelength in µm
    """
    radiance = radiance.clamp(min=1e-10)
    arg = _C1 / (wavelength_um**5 * radiance) + 1.0
    return _C2 / (wavelength_um * torch.log(arg))


def dn_to_bt(
    dn:             torch.Tensor,
    gain:           float,
    offset:         float,
    wavelength_um:  float,
) -> torch.Tensor:
    """
    Convert raw sensor DN to Brightness Temperature.

    radiance = DN · gain + offset   (linear radiometric calibration)
    BT       = Planck inversion of radiance at wavelength_um

    Args:
        dn:            raw DN values (any shape)
        gain:          radiometric gain  (W/(m²·sr·µm) per DN)
        offset:        radiometric offset (W/(m²·sr·µm))
        wavelength_um: central wavelength in µm
    """
    radiance = dn * gain + offset
    radiance = radiance.clamp(min=1e-10)
    return bt_from_radiance(radiance, wavelength_um)


def norm_to_dn(
    x_norm:   torch.Tensor,
    dn_min:   float,
    dn_max:   float,
) -> torch.Tensor:
    """
    Convert network-normalised image ([-1, 1]) back to physical DN.
    Inverts the linear stretch applied during preprocessing.

    DN = (x_norm + 1) / 2 · (dn_max - dn_min) + dn_min
    """
    return (x_norm + 1.0) * 0.5 * (dn_max - dn_min) + dn_min


def norm_to_bt(
    x_norm:        torch.Tensor,
    dn_min:        float,
    dn_max:        float,
    gain:          float,
    offset:        float,
    wavelength_um: float,
) -> torch.Tensor:
    """Full pipeline: normalised image → DN → radiance → BT."""
    dn = norm_to_dn(x_norm, dn_min, dn_max)
    return dn_to_bt(dn, gain, offset, wavelength_um)


# ─────────────────────────────────────────────
# MWIR Confidence Weight
# ─────────────────────────────────────────────

def mwir_confidence_weight(mwir_bt: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    """
    Per-pixel confidence weight based on local MWIR thermal contrast.

    Pixels in homogeneous regions (low local std) get lower weight because:
    - The BT estimate is noisier in flat regions
    - The physical constraint is looser for near-uniform scenes

    High-contrast pixels (edges, structure boundaries, thermal targets)
    get weight ≈ 1.0 — these are where the physical constraint matters most.

    Uses local standard deviation computed with a small Gaussian kernel.
    """
    # Local std in a 7×7 neighbourhood
    k = 7
    pad = k // 2
    mu  = F.avg_pool2d(mwir_bt, k, stride=1, padding=pad)
    var = F.avg_pool2d(mwir_bt**2, k, stride=1, padding=pad) - mu**2
    local_std = var.clamp(min=0).sqrt()

    # Soft sigmoid gate: low contrast → weight→0, high contrast → weight→1
    weight = torch.sigmoid((local_std - sigma) / sigma)
    return weight.detach()   # do not backprop through the weight


# ─────────────────────────────────────────────
# Planck Ratio Loss
# ─────────────────────────────────────────────

class PlanckRatioLoss(nn.Module):
    """
    Physics-informed Planck ratio loss for MWIR→LWIR synthesis.

    Penalises generated LWIR images where the implied surface brightness
    temperature is inconsistent with the MWIR conditioning image.

    The loss is a soft Huber-style penalty:
        Δ = |BT_LWIR - BT_MWIR|
        loss = huber(Δ, allowed_delta_K) · confidence_weight(MWIR)

    This permits physically valid emissivity-driven BT differences
    (|Δ| < allowed_delta_K) but penalises clearly unphysical reversals.

    Args:
        mwir_wavelength_um: MWIR central wavelength (µm), e.g. 4.0
        lwir_wavelength_um: LWIR central wavelength (µm), e.g. 10.0
        mwir_gain:          MWIR radiometric gain (W/(m²·sr·µm) per DN)
        mwir_offset:        MWIR radiometric offset (W/(m²·sr·µm))
        lwir_gain:          LWIR radiometric gain
        lwir_offset:        LWIR radiometric offset
        mwir_norm_min:      DN value that maps to -1 in normalised MWIR
        mwir_norm_max:      DN value that maps to +1 in normalised MWIR
        lwir_norm_min:      DN value that maps to -1 in normalised LWIR
        lwir_norm_max:      DN value that maps to +1 in normalised LWIR
        allowed_delta_K:    physical tolerance in K before penalty activates
                            (default 15 K — covers emissivity variation for
                             most natural surfaces; tighten for water/veg only)
        use_confidence:     weight by MWIR local thermal contrast (recommended)
        confidence_sigma:   local std threshold for confidence weighting (K)
    """

    def __init__(
        self,
        mwir_wavelength_um: float = 4.0,
        lwir_wavelength_um: float = 10.0,
        mwir_gain:          float = 0.01,
        mwir_offset:        float = -0.5,
        lwir_gain:          float = 0.005,
        lwir_offset:        float = 200.0,
        mwir_norm_min:      float = 88.0,
        mwir_norm_max:      float = 3804.0,
        lwir_norm_min:      float = 1622.0,
        lwir_norm_max:      float = 3836.0,
        allowed_delta_K:    float = 15.0,
        use_confidence:     bool  = True,
        confidence_sigma:   float = 2.0,
    ):
        super().__init__()
        self.mwir_wl        = mwir_wavelength_um
        self.lwir_wl        = lwir_wavelength_um
        self.mwir_gain      = mwir_gain
        self.mwir_offset    = mwir_offset
        self.lwir_gain      = lwir_gain
        self.lwir_offset    = lwir_offset
        self.mwir_dn_min    = mwir_norm_min
        self.mwir_dn_max    = mwir_norm_max
        self.lwir_dn_min    = lwir_norm_min
        self.lwir_dn_max    = lwir_norm_max
        self.allowed_delta  = allowed_delta_K
        self.use_confidence = use_confidence
        self.conf_sigma     = confidence_sigma

    def mwir_to_bt(self, mwir_norm: torch.Tensor) -> torch.Tensor:
        """Normalised MWIR image → BT in Kelvin."""
        return norm_to_bt(
            mwir_norm,
            self.mwir_dn_min, self.mwir_dn_max,
            self.mwir_gain, self.mwir_offset,
            self.mwir_wl,
        )

    def lwir_to_bt(self, lwir_norm: torch.Tensor) -> torch.Tensor:
        """Normalised LWIR image (generated or real) → BT in Kelvin."""
        return norm_to_bt(
            lwir_norm,
            self.lwir_dn_min, self.lwir_dn_max,
            self.lwir_gain, self.lwir_offset,
            self.lwir_wl,
        )

    def forward(
        self,
        pred_lwir: torch.Tensor,   # (B, 1, H, W) generated LWIR in [-1, 1]
        mwir:      torch.Tensor,   # (B, C, H, W) conditioning MWIR in [-1, 1]
        real_lwir: Optional[torch.Tensor] = None,  # if provided, also compute vs real
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the Planck ratio loss.

        Returns dict with keys 'planck_loss' and optionally 'planck_real_loss'.
        """
        # Use first MWIR channel if multi-channel
        mwir_1ch = mwir[:, :1]   # (B, 1, H, W)

        # Convert to BT
        bt_mwir = self.mwir_to_bt(mwir_1ch)      # (B, 1, H, W) in K
        bt_lwir = self.lwir_to_bt(pred_lwir)      # (B, 1, H, W) in K

        # Temperature difference: how far is generated LWIR from MWIR BT?
        delta = bt_lwir - bt_mwir                 # (B, 1, H, W) in K

        # Soft penalty: Huber loss with threshold = allowed_delta_K
        # |δ| ≤ allowed_delta → loss = 0.5·δ²/allowed_delta  (quadratic)
        # |δ| > allowed_delta → loss = |δ| - 0.5·allowed_delta (linear)
        # This smoothly penalises violations without hard clipping.
        penalty = F.huber_loss(
            delta,
            torch.zeros_like(delta),
            delta=self.allowed_delta,
            reduction='none',
        )

        # Confidence weighting: down-weight homogeneous MWIR regions
        if self.use_confidence:
            weight = mwir_confidence_weight(bt_mwir, self.conf_sigma)
            loss = (penalty * weight).mean()
        else:
            loss = penalty.mean()

        out = {'planck_loss': loss}

        # Optionally compute BT error vs real LWIR (for validation diagnostics)
        if real_lwir is not None:
            bt_real = self.lwir_to_bt(real_lwir)
            bt_error_mean = (bt_lwir - bt_real).abs().mean()
            bt_error_std  = (bt_lwir - bt_real).std()
            out['bt_mae_K']   = bt_error_mean
            out['bt_std_K']   = bt_error_std

        return out

    @torch.no_grad()
    def diagnostics(
        self,
        pred_lwir: torch.Tensor,
        mwir:      torch.Tensor,
        real_lwir: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Extended per-scene diagnostics — call at validation time.
        Returns human-readable BT statistics.
        """
        mwir_1ch = mwir[:, :1]
        bt_mwir  = self.mwir_to_bt(mwir_1ch)
        bt_lwir  = self.lwir_to_bt(pred_lwir)
        delta    = (bt_lwir - bt_mwir)

        d = {
            'bt_mwir_mean_K':   float(bt_mwir.mean()),
            'bt_lwir_pred_K':   float(bt_lwir.mean()),
            'delta_mean_K':     float(delta.mean()),
            'delta_abs_mean_K': float(delta.abs().mean()),
            'delta_p95_K':      float(delta.abs().flatten().quantile(0.95)),
            'pct_violation':    float((delta.abs() > self.allowed_delta).float().mean() * 100),
        }
        if real_lwir is not None:
            bt_real = self.lwir_to_bt(real_lwir)
            d['bt_lwir_real_K']   = float(bt_real.mean())
            d['bt_mae_pred_K']    = float((bt_lwir - bt_real).abs().mean())
        return d

    @classmethod
    def from_config(cls, cfg: dict) -> 'PlanckRatioLoss':
        """Instantiate from the 'planck' sub-dict in your config JSON."""
        p = cfg.get('planck', {})
        return cls(
            mwir_wavelength_um = p.get('mwir_wavelength_um', 4.0),
            lwir_wavelength_um = p.get('lwir_wavelength_um', 10.0),
            mwir_gain          = p.get('mwir_gain',   0.01),
            mwir_offset        = p.get('mwir_offset', -0.5),
            lwir_gain          = p.get('lwir_gain',   0.005),
            lwir_offset        = p.get('lwir_offset', 200.0),
            mwir_norm_min      = p.get('mwir_norm_min',  88.0),
            mwir_norm_max      = p.get('mwir_norm_max',  3804.0),
            lwir_norm_min      = p.get('lwir_norm_min',  1622.0),
            lwir_norm_max      = p.get('lwir_norm_max',  3836.0),
            allowed_delta_K    = p.get('allowed_delta_K', 15.0),
            use_confidence     = p.get('use_confidence', True),
            confidence_sigma   = p.get('confidence_sigma', 2.0),
        )
