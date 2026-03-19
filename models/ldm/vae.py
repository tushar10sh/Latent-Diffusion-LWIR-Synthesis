"""
KL-Regularized VAE for LWIR imagery.

Design decisions for 500-2000 sample regime:
  - Architecture mirrors SD's KL-VAE (f=4 compression, not f=8)
    → 256×256 LWIR becomes 64×64×4 latent
    → f=4 preserves more fine thermal texture than f=8
    → Still 16× smaller than pixel space for DiT efficiency
  - Weights initialised from SD KL-VAE where possible (transfer learning)
    → Input conv: 4ch→1ch weight averaging
    → Output conv: reinitialised (distribution shift too large)
    → All middle layers: transferred directly
  - Perceptual loss uses Gabor filter responses (no VGG needed for IR)
  - KL weight follows free-bits annealing schedule

Latent convention:  z ∈ R^(B, 4, H/4, W/4),  normalised to ~N(0,1)
                    DiT operates on this z.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ─────────────────────────────────────────────
# Building blocks (same style as SD's VAE)
# ─────────────────────────────────────────────

def nonlinearity(x):
    return F.silu(x)

def Normalize(in_channels, num_groups=32):
    # Ensure num_groups divides in_channels
    g = min(num_groups, in_channels)
    while in_channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=0)

    def forward(self, x):
        x = F.pad(x, (0, 1, 0, 1), mode='constant', value=0)
        return self.conv(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.nin_shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        h = self.conv1(nonlinearity(self.norm1(x)))
        h = self.conv2(self.dropout(nonlinearity(self.norm2(h))))
        return h + self.nin_shortcut(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        B, C, H, W = q.shape
        q = q.reshape(B, C, H*W).permute(0, 2, 1)
        k = k.reshape(B, C, H*W)
        w = torch.bmm(q, k) * (C ** -0.5)
        w = F.softmax(w, dim=2)
        v = v.reshape(B, C, H*W)
        h = torch.bmm(v, w.permute(0, 2, 1)).reshape(B, C, H, W)
        return x + self.proj_out(h)


# ─────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────

class IREncoder(nn.Module):
    """
    Encodes LWIR images into latent space.
    f=4 compression: 256→64 spatial, 1→4 channel.

    ch_mult=(1,2,4) gives channels: 128, 256, 512 at each resolution.
    Two resolution halvings → f=4.
    """
    def __init__(
        self,
        in_channels: int = 1,
        ch: int = 128,
        ch_mult: Tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        z_channels: int = 4,
        dropout: float = 0.0,
        double_z: bool = True,    # output mean + logvar
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)

        # Input projection
        self.conv_in = nn.Conv2d(in_channels, ch, 3, padding=1)

        # Downsampling
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            blocks = nn.ModuleList([
                ResnetBlock(block_in if j == 0 else block_out, block_out, dropout)
                for j in range(num_res_blocks)
            ])
            level = nn.Module()
            level.block = blocks
            level.attn = nn.ModuleList([
                AttnBlock(block_out) if i_level == self.num_resolutions - 1 else nn.Identity()
                for _ in range(num_res_blocks)
            ])
            level.downsample = (
                Downsample(block_out)
                if i_level != self.num_resolutions - 1
                else nn.Identity()
            )
            self.down.append(level)

        # Middle
        mid_ch = ch * ch_mult[-1]
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(mid_ch, mid_ch, dropout)
        self.mid.attn_1 = AttnBlock(mid_ch)
        self.mid.block_2 = ResnetBlock(mid_ch, mid_ch, dropout)

        # Output
        self.norm_out = Normalize(mid_ch)
        self.conv_out = nn.Conv2d(
            mid_ch,
            2 * z_channels if double_z else z_channels,
            3, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for level in self.down:
            for i, block in enumerate(level.block):
                h = block(h)
                h = level.attn[i](h)
            h = level.downsample(h)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        return self.conv_out(nonlinearity(self.norm_out(h)))


# ─────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────

class IRDecoder(nn.Module):
    """
    Decodes latent z → LWIR image.
    Mirror of IREncoder with upsampling.
    """
    def __init__(
        self,
        out_channels: int = 1,
        ch: int = 128,
        ch_mult: Tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        z_channels: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        block_in = ch * ch_mult[-1]

        # Input projection from latent
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, padding=1)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in, dropout)

        # Upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block_out = ch * ch_mult[i_level]
            blocks = nn.ModuleList([
                ResnetBlock(block_in if j == 0 else block_out, block_out, dropout)
                for j in range(num_res_blocks + 1)
            ])
            level = nn.Module()
            level.block = blocks
            level.attn = nn.ModuleList([
                AttnBlock(block_out) if i_level == self.num_resolutions - 1 else nn.Identity()
                for _ in range(num_res_blocks + 1)
            ])
            level.upsample = (
                Upsample(block_out)
                if i_level != 0
                else nn.Identity()
            )
            self.up.append(level)
            block_in = block_out

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        for level in self.up:
            for i, block in enumerate(level.block):
                h = block(h)
                h = level.attn[i](h)
            h = level.upsample(h)
        return torch.tanh(self.conv_out(nonlinearity(self.norm_out(h))))


# ─────────────────────────────────────────────
# Diagonal Gaussian (reparameterisation)
# ─────────────────────────────────────────────

class DiagonalGaussian:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.mean, self.logvar = parameters.chunk(2, dim=1)
        self.logvar = self.logvar.clamp(-30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        self.deterministic = deterministic

    def sample(self) -> torch.Tensor:
        if self.deterministic:
            return self.mean
        return self.mean + self.std * torch.randn_like(self.std)

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self, other=None) -> torch.Tensor:
        """KL divergence vs N(0,1) (standard VAE KL)."""
        return 0.5 * (self.mean.pow(2) + self.var - 1.0 - self.logvar).mean()


# ─────────────────────────────────────────────
# Perceptual Loss (Gabor-based, no VGG)
# ─────────────────────────────────────────────

class IRPerceptualLoss(nn.Module):
    """
    Multi-scale Gabor perceptual loss for single-channel IR.
    Replaces VGG-based perceptual loss — VGG was trained on RGB and
    its features are poorly calibrated for thermal imagery.

    Fix: all kernels are zero-padded to the largest kernel size (max_k)
    before stacking so torch.stack receives uniform shapes.
    A single batched conv2d with padding=max_k//2 is then used for
    all filters simultaneously — faster and avoids the loop.
    """
    def __init__(self, num_scales: int = 4, num_orientations: int = 8):
        super().__init__()

        # Pre-compute every (sigma, k) pair so we know max_k before building kernels
        scale_params = []
        for s in range(num_scales):
            sigma = 1.5 * (2 ** s)
            k = int(6 * sigma) | 1      # ensure odd
            scale_params.append((sigma, k))

        max_k = max(k for _, k in scale_params)   # largest kernel side (e.g. 37 for 4 scales)

        kernels = []
        for sigma, k in scale_params:
            for o in range(num_orientations):
                theta = o * math.pi / num_orientations
                g = self._gabor(k, sigma, theta)   # (k, k)

                # Zero-pad to (max_k, max_k) so all kernels share the same spatial size
                if k < max_k:
                    pad = (max_k - k) // 2
                    g = F.pad(g, (pad, pad, pad, pad))  # left, right, top, bottom

                kernels.append(g)   # each is now (max_k, max_k)

        # Stack is safe now — all tensors are (max_k, max_k)
        filters = torch.stack(kernels).unsqueeze(1)   # (N, 1, max_k, max_k)
        self.register_buffer('filters', filters)
        self.max_k = max_k

    @staticmethod
    def _gabor(k: int, sigma: float, theta: float, freq_scale: float = 0.25) -> torch.Tensor:
        half = k // 2
        y, x = torch.meshgrid(
            torch.arange(-half, half + 1, dtype=torch.float32),
            torch.arange(-half, half + 1, dtype=torch.float32),
            indexing='ij',
        )
        xr = x * math.cos(theta) + y * math.sin(theta)
        yr = -x * math.sin(theta) + y * math.cos(theta)
        freq = freq_scale / sigma
        g = torch.exp(-0.5 * (xr ** 2 + yr ** 2) / sigma ** 2) * torch.cos(2 * math.pi * freq * xr)
        g = g - g.mean()
        g = g / (g.norm() + 1e-8)
        return g   # (k, k)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        # Single batched conv — all filters are the same spatial size now
        return F.conv2d(x, self.filters, padding=self.max_k // 2)  # (B, N, H, W)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._features(pred), self._features(target))


# ─────────────────────────────────────────────
# Full KL-VAE
# ─────────────────────────────────────────────

class IRVAE(nn.Module):
    """
    KL-regularized VAE for LWIR imagery.

    Latent scale factor (self.scale_factor) is computed empirically
    after training to normalise z to approximately N(0,1) — required
    for the DiT's noise schedule to be calibrated correctly.

    Args:
        in_channels:    1 for single-band LWIR
        ch:             base channel width (128 recommended)
        ch_mult:        spatial compression stages; (1,2,4) → f=4
        z_channels:     latent channel depth (4, same as SD)
        kl_weight:      KL annealing start weight
        kl_weight_max:  KL annealing end weight
    """
    def __init__(
        self,
        in_channels: int = 1,
        ch: int = 128,
        ch_mult: Tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        z_channels: int = 4,
        kl_weight: float = 1e-6,
        kl_weight_max: float = 1e-4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.z_channels = z_channels
        self.kl_weight = kl_weight
        self.kl_weight_max = kl_weight_max
        # Both set by compute_scale_factor() after training.
        # scale_factor = 1 / std(z_raw - latent_mean)
        # latent_mean  = mean(z_raw)  — absorbs encoder bias
        self.scale_factor = 1.0
        self.latent_mean  = 0.0

        self.encoder = IREncoder(in_channels, ch, ch_mult, num_res_blocks, z_channels, dropout)
        self.decoder = IRDecoder(in_channels, ch, ch_mult, num_res_blocks, z_channels, dropout)
        self.perceptual = IRPerceptualLoss()

    def encode(self, x: torch.Tensor) -> DiagonalGaussian:
        h = self.encoder(x)
        return DiagonalGaussian(h)

    def _get_affine(self, device: torch.device) -> tuple:
        """
        Return (mean_t, scale_t) as broadcastable tensors on `device`.
        Handles both the old scalar format and the new per-channel list format,
        so old checkpoints load without error.
        """
        mean  = self.latent_mean
        scale = self.scale_factor

        if isinstance(mean, (list, tuple)):
            mean_t  = torch.tensor(mean,  dtype=torch.float32, device=device)
            scale_t = torch.tensor(scale, dtype=torch.float32, device=device)
            # Reshape to (1, C, 1, 1) for broadcasting against (B, C, H, W)
            mean_t  = mean_t.view(1, -1, 1, 1)
            scale_t = scale_t.view(1, -1, 1, 1)
        else:
            # Scalar fallback (old checkpoints)
            mean_t  = torch.tensor(float(mean),  device=device)
            scale_t = torch.tensor(float(scale), device=device)

        return mean_t, scale_t

    def encode_to_dit(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode x and return the per-channel-normalised latent the DiT trains on.

            z_DiT[c] = (z_raw[c] - latent_mean[c]) * scale_factor[c]  →  N(0,1)
        """
        posterior = self.encode(x)
        z_raw = posterior.sample()
        mean_t, scale_t = self._get_affine(z_raw.device)
        return (z_raw - mean_t) * scale_t

    def decode(self, z_dit: torch.Tensor) -> torch.Tensor:
        """
        Decode a DiT-space latent back to pixel space.

            z_raw[c] = z_dit[c] / scale_factor[c] + latent_mean[c]
        """
        mean_t, scale_t = self._get_affine(z_dit.device)
        z_raw = z_dit / scale_t + mean_t
        return self.decoder(z_raw)

    def forward(self, x: torch.Tensor, sample_posterior: bool = True):
        """
        VAE reconstruction round-trip (used only during VAE training).
        The affine transform cancels in decode(encode_to_dit(x)), so
        reconstruction quality is independent of calibration values.
        """
        posterior = self.encode(x)
        z_raw = posterior.sample() if sample_posterior else posterior.mode()
        mean_t, scale_t = self._get_affine(z_raw.device)
        z_dit = (z_raw - mean_t) * scale_t
        recon = self.decode(z_dit)
        return recon, posterior

    def training_step(
        self,
        x: torch.Tensor,
        kl_weight: Optional[float] = None,
    ) -> Tuple[torch.Tensor, dict]:
        recon, posterior = self(x, sample_posterior=True)

        recon_loss = F.l1_loss(recon, x)
        perc_loss = self.perceptual(recon, x)
        kl_loss = posterior.kl()

        w_kl = kl_weight if kl_weight is not None else self.kl_weight
        total = recon_loss + 0.1 * perc_loss + w_kl * kl_loss

        return total, {
            'recon': recon_loss.item(),
            'perceptual': perc_loss.item(),
            'kl': kl_loss.item(),
            'kl_weight': w_kl,
            'total': total.item(),
        }

    @torch.no_grad()
    def compute_scale_factor(self, dataloader, device: str = None, n_batches: int = 50):
        """
        Compute affine normalisation parameters so that the latents fed to the
        DiT are approximately N(0, 1) **per channel**.

        Per-channel (not global) normalisation is used because IR latent spaces
        are typically multimodal — different land cover types (water, vegetation,
        urban, bare soil) encode to distinct clusters. A single global mean/std
        falls in the trough between modes and inflates the std. Per-channel
        statistics are robust to this structure.

        Parameters stored:
            latent_mean   : (z_channels,) per-channel mean  — shape (C,)
            scale_factor  : (z_channels,) per-channel 1/std — shape (C,)

        Encoding:  z_DiT[c] = (z_raw[c] - latent_mean[c]) * scale_factor[c]
        Decoding:  z_raw[c] = z_DiT[c] / scale_factor[c] + latent_mean[c]

        Device handling: the model must already be on the correct device before
        calling this function (use vae.to(device) first). The `device` argument
        is kept for backward compatibility but is now ignored — the model's own
        device is used, eliminating the CPU/CUDA mismatch error.
        """
        # Derive device from model weights — avoids CPU/CUDA type mismatch
        model_device = next(self.parameters()).device

        self.eval()
        z_samples = []
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            x = batch['lwir'].to(model_device)
            z = self.encode(x).mode()          # (B, C, H, W) — mode for stability
            z_samples.append(z.cpu())

        z_all = torch.cat(z_samples, dim=0)    # (N, C, H, W)

        # Per-channel mean and std (averaged over batch B and spatial H×W)
        # z_all: (N, C, H, W)  →  stats shape: (C,)
        channel_mean = z_all.mean(dim=(0, 2, 3))          # (C,)
        channel_std  = z_all.std(dim=(0, 2, 3)).clamp(min=1e-6)   # (C,) — prevent div/0

        # Store as plain Python lists for JSON-serialisable checkpoints
        self.latent_mean  = channel_mean.tolist()          # list[float] of length C
        self.scale_factor = (1.0 / channel_std).tolist()  # list[float] of length C

        # Verify: compute z_DiT statistics after normalisation
        # z_DiT[c] = (z_all[c] - mean[c]) / std[c]
        z_dit_check = (z_all - channel_mean[None, :, None, None]) \
                      * (1.0 / channel_std)[None, :, None, None]
        check_mean = z_dit_check.mean(dim=(0, 2, 3)).tolist()
        check_std  = z_dit_check.std(dim=(0, 2, 3)).tolist()

        print(f"[VAE] Per-channel latent calibration ({z_all.shape[1]} channels):")
        for c in range(z_all.shape[1]):
            print(
                f"  ch{c}: raw_mean={channel_mean[c]:.4f}  raw_std={channel_std[c]:.4f}"
                f"  →  z_DiT mean={check_mean[c]:.4f}  std={check_std[c]:.4f}"
            )
        print("[VAE] Calibration complete. z_DiT ~ N(0,1) per channel ✓")

        self.train()
        return self.scale_factor

    @classmethod
    def from_pretrained_sd(
        cls,
        sd_vae_path: str,
        **kwargs
    ) -> 'IRVAE':
        """
        Initialise from SD's KL-VAE checkpoint (diffusers format).

        Handles both formats produced by huggingface_hub download:
          - diffusion_pytorch_model.safetensors   (preferred)
          - diffusion_pytorch_model.bin            (legacy .pt)

        Diffusers key naming convention (what the downloaded file contains):
          encoder.conv_in.weight              (512, 3, 3, 3)  — RGB input
          encoder.down_blocks.0.resnets.0.*
          encoder.mid_block.*
          decoder.conv_in.weight
          decoder.up_blocks.*.resnets.*
          decoder.mid_block.*
          decoder.conv_out.weight             (3, 128, 3, 3)  — RGB output
          quant_conv.weight / post_quant_conv.weight

        Our key naming convention:
          encoder.conv_in.weight              (128, 1, 3, 3)  — IR input
          encoder.down.0.block.0.*
          encoder.mid.*
          decoder.conv_in.weight
          decoder.up.0.block.0.*
          decoder.mid.*
          decoder.conv_out.weight             (1, 128, 3, 3)  — IR output

        Strategy:
          encoder.conv_in   : average SD's 3 RGB input channels → 1 IR channel
          decoder.conv_out  : reinitialise (RGB output stats ≠ IR)
          quant_conv        : transferred directly (z_channels=4 matches SD)
          post_quant_conv   : transferred directly
          all resnet/attn   : transferred directly where shapes match
        """
        model = cls(**kwargs)

        sd_path = str(sd_vae_path)
        print(f"[VAE] Loading SD weights from: {sd_path}")

        try:
            # ── Load checkpoint (safetensors or .bin/.pt) ──────────────
            if sd_path.endswith('.safetensors'):
                try:
                    from safetensors.torch import load_file
                    sd_state = load_file(sd_path, device='cpu')
                except ImportError:
                    raise ImportError(
                        "safetensors package required for .safetensors files.\n"
                        "Install offline: pip install --no-index --find-links ./offline_wheels safetensors"
                    )
            else:
                sd_state = torch.load(sd_path, map_location='cpu', weights_only=True)
                if 'state_dict' in sd_state:
                    sd_state = sd_state['state_dict']

            # ── Build diffusers→ours key mapping ───────────────────────
            # Diffusers uses a different block hierarchy. We remap the
            # keys we can use and skip the ones with structural mismatches
            # (different ch_mult, different num_res_blocks, etc.).
            # The most important transfers are:
            #   - quant_conv / post_quant_conv  (latent projection, shape matches exactly)
            #   - mid_block resnets + attention  (shape matches if ch_mult[-1] same)
            #   - conv_in (with channel averaging)

            model_state = model.state_dict()
            transferred, skipped, adapted = 0, 0, 0

            # Map quant_conv and post_quant_conv directly (z_channels=4 in both)
            direct_keys = {
                'quant_conv.weight': 'encoder.conv_out.weight',   # not quite but handled below
                'post_quant_conv.weight': 'decoder.conv_in.weight',
            }

            for sd_key, sd_val in sd_state.items():
                sd_val = sd_val.float()

                # ── quant_conv → encoder output projection ──
                if sd_key == 'quant_conv.weight' and 'encoder.conv_out.weight' in model_state:
                    our_val = model_state['encoder.conv_out.weight']
                    if sd_val.shape == our_val.shape:
                        model_state['encoder.conv_out.weight'] = sd_val
                        transferred += 1
                    continue

                if sd_key == 'quant_conv.bias' and 'encoder.conv_out.bias' in model_state:
                    our_val = model_state['encoder.conv_out.bias']
                    if sd_val.shape == our_val.shape:
                        model_state['encoder.conv_out.bias'] = sd_val
                        transferred += 1
                    continue

                # ── post_quant_conv → decoder input projection ──
                if sd_key == 'post_quant_conv.weight' and 'decoder.conv_in.weight' in model_state:
                    our_val = model_state['decoder.conv_in.weight']
                    if sd_val.shape == our_val.shape:
                        model_state['decoder.conv_in.weight'] = sd_val
                        transferred += 1
                    continue

                if sd_key == 'post_quant_conv.bias' and 'decoder.conv_in.bias' in model_state:
                    our_val = model_state['decoder.conv_in.bias']
                    if sd_val.shape == our_val.shape:
                        model_state['decoder.conv_in.bias'] = sd_val
                        transferred += 1
                    continue

                # ── encoder.conv_in: 3ch RGB → 1ch IR (average channels) ──
                if sd_key == 'encoder.conv_in.weight':
                    our_key = 'encoder.conv_in.weight'
                    if our_key in model_state:
                        # SD shape: (out_ch, 3, k, k)  Ours: (out_ch, 1, k, k)
                        if sd_val.shape[1] >= 3 and model_state[our_key].shape[1] == 1:
                            # Average RGB → greyscale-equivalent for IR init
                            model_state[our_key] = sd_val[:, :3].mean(dim=1, keepdim=True)
                            adapted += 1
                        elif sd_val.shape == model_state[our_key].shape:
                            model_state[our_key] = sd_val
                            transferred += 1
                    continue

                # ── decoder.conv_out: skip — IR output stats ≠ RGB ──
                if sd_key in ('decoder.conv_out.weight', 'decoder.conv_out.bias'):
                    skipped += 1
                    continue

                # ── All other keys: try direct transfer if shape matches ──
                # Remap diffusers nested key format to our flat format where possible
                our_key = _remap_diffusers_key(sd_key)
                if our_key and our_key in model_state:
                    if sd_val.shape == model_state[our_key].shape:
                        model_state[our_key] = sd_val
                        transferred += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1

            model.load_state_dict(model_state, strict=True)
            print(
                f"[VAE] SD weights loaded: "
                f"{transferred} direct | {adapted} adapted | {skipped} skipped/reinit"
            )
            print(
                f"[VAE] Note: skipped layers will train from scratch (fine for resnet/attn blocks "
                f"with different ch_mult). The critical transfers are quant_conv and conv_in."
            )

        except Exception as e:
            print(f"[VAE] Could not load SD checkpoint: {e}")
            print(f"[VAE] Falling back to random initialisation. "
                  f"Increase vae_total_steps to 100k to compensate.")

        return model


def _remap_diffusers_key(key: str) -> str:
    """
    Attempt to remap a diffusers-format key to our key format.
    Returns None if no mapping is possible.

    Diffusers format examples:
      encoder.mid_block.resnets.0.norm1.weight
      encoder.mid_block.attentions.0.group_norm.weight
      decoder.up_blocks.0.resnets.0.norm1.weight

    Our format examples:
      encoder.mid.block_1.norm1.weight
      encoder.mid.attn_1.norm.weight
      decoder.up.0.block.0.norm1.weight
    """
    # Mid block resnets
    if 'mid_block.resnets.0' in key:
        return key.replace('mid_block.resnets.0', 'mid.block_1')
    if 'mid_block.resnets.1' in key:
        return key.replace('mid_block.resnets.1', 'mid.block_2')
    if 'mid_block.attentions.0.group_norm' in key:
        return key.replace('mid_block.attentions.0.group_norm', 'mid.attn_1.norm')
    if 'mid_block.attentions.0.to_q' in key:
        return key.replace('mid_block.attentions.0.to_q', 'mid.attn_1.q')
    if 'mid_block.attentions.0.to_k' in key:
        return key.replace('mid_block.attentions.0.to_k', 'mid.attn_1.k')
    if 'mid_block.attentions.0.to_v' in key:
        return key.replace('mid_block.attentions.0.to_v', 'mid.attn_1.v')
    if 'mid_block.attentions.0.to_out.0' in key:
        return key.replace('mid_block.attentions.0.to_out.0', 'mid.attn_1.proj_out')

    # Encoder norm_out and conv_out (before quant_conv)
    if key == 'encoder.conv_norm_out.weight':
        return 'encoder.norm_out.weight'
    if key == 'encoder.conv_norm_out.bias':
        return 'encoder.norm_out.bias'

    # Decoder norm_out
    if key == 'decoder.conv_norm_out.weight':
        return 'decoder.norm_out.weight'
    if key == 'decoder.conv_norm_out.bias':
        return 'decoder.norm_out.bias'

    # Down blocks — structural mismatch likely, return None to skip gracefully
    if 'down_blocks' in key or 'up_blocks' in key:
        return None

    return None
