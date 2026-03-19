"""
Conditional UNet for MWIR → LWIR synthesis.

Architecture improvements over vanilla DDPM:
  - Cross-modal attention blocks (MWIR conditions each decoder stage)
  - Spectral normalization on all conv layers (stabilizes heterogeneous IR data)
  - Fourier feature positional embeddings for timestep
  - BigGAN-style residual blocks
  - Adaptive Group Normalization (AdaGN) conditioned on both timestep & MWIR features
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def exists(x):
    return x is not None

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

def spectral_conv(in_c, out_c, kernel=3, stride=1, padding=1, bias=True):
    return nn.utils.spectral_norm(
        nn.Conv2d(in_c, out_c, kernel, stride=stride, padding=padding, bias=bias)
    )

# ─────────────────────────────────────────────
# Timestep / Fourier Embeddings
# ─────────────────────────────────────────────

class FourierTimestepEmbedding(nn.Module):
    """
    Random Fourier feature embedding for timestep (better than sinusoidal for
    diffusion models operating on heterogeneous sensor data).
    """
    def __init__(self, dim: int, num_fourier_features: int = 64):
        super().__init__()
        self.dim = dim
        self.register_buffer('freqs', torch.randn(num_fourier_features) * 0.1)
        self.proj = nn.Sequential(
            nn.Linear(num_fourier_features * 2, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float().unsqueeze(-1)                      # (B, 1)
        freqs = self.freqs.unsqueeze(0)                  # (1, F)
        x = t * freqs * 2 * math.pi
        x = torch.cat([x.sin(), x.cos()], dim=-1)       # (B, 2F)
        return self.proj(x)                              # (B, dim)


# ─────────────────────────────────────────────
# Adaptive Group Norm
# ─────────────────────────────────────────────

class AdaGN(nn.Module):
    """
    Adaptive Group Normalization conditioned on both timestep embedding
    and MWIR context vector.
    """
    def __init__(self, num_channels: int, context_dim: int, groups: int = 32):
        super().__init__()
        self.gn = nn.GroupNorm(groups, num_channels, affine=False)
        self.scale_proj = nn.Linear(context_dim, num_channels)
        self.shift_proj = nn.Linear(context_dim, num_channels)
        nn.init.zeros_(self.scale_proj.weight); nn.init.ones_(self.scale_proj.bias)
        nn.init.zeros_(self.shift_proj.weight); nn.init.zeros_(self.shift_proj.bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)   context: (B, ctx_dim)
        x = self.gn(x)
        scale = self.scale_proj(context)[:, :, None, None]
        shift = self.shift_proj(context)[:, :, None, None]
        return x * scale + shift


# ─────────────────────────────────────────────
# Cross-Modal Attention
# ─────────────────────────────────────────────

class CrossModalAttention(nn.Module):
    """
    Multi-head cross-attention: query from LWIR noise features,
    key/value from MWIR encoder features.
    Designed for heterogeneous spectral bands where spatial statistics differ.
    """
    def __init__(self, query_dim: int, context_dim: int, heads: int = 8, dim_head: int = 64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = zero_module(nn.Linear(inner_dim, query_dim))
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_k = nn.LayerNorm(context_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        Bc, Cc, Hc, Wc = context.shape

        # Flatten spatial
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        ctx_flat = rearrange(context, 'b c h w -> b (h w) c')

        x_norm = self.norm_q(x_flat)
        ctx_norm = self.norm_k(ctx_flat)

        q = rearrange(self.to_q(x_norm), 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(self.to_k(ctx_norm), 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(self.to_v(ctx_norm), 'b n (h d) -> b h n d', h=self.heads)

        # Scaled dot-product (flash-attention compatible)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = rearrange(attn, 'b h n d -> b n (h d)')
        out = self.to_out(attn)
        out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        return x + out


# ─────────────────────────────────────────────
# Residual Block (BigGAN-style + AdaGN)
# ─────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        context_dim: int,
        groups: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = AdaGN(in_channels, context_dim, groups)
        self.conv1 = spectral_conv(in_channels, out_channels)
        self.norm2 = AdaGN(out_channels, context_dim, groups)
        self.conv2 = zero_module(spectral_conv(out_channels, out_channels))
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            spectral_conv(in_channels, out_channels, kernel=1, padding=0)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x, context)))
        h = self.dropout(h)
        h = self.conv2(F.silu(self.norm2(h, context)))
        return h + self.skip(x)


# ─────────────────────────────────────────────
# Self-Attention Block
# ─────────────────────────────────────────────

class SelfAttentionBlock(nn.Module):
    def __init__(self, channels: int, heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.proj_out = zero_module(nn.Conv2d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        h = rearrange(h, 'b c h w -> b (h w) c')
        h, _ = self.attn(h, h, h)
        h = rearrange(h, 'b (h w) c -> b c h w', h=H, w=W)
        return x + self.proj_out(h)


# ─────────────────────────────────────────────
# Down / Up sampling
# ─────────────────────────────────────────────

class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = spectral_conv(channels, channels, kernel=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = spectral_conv(channels, channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# ─────────────────────────────────────────────
# MWIR Encoder (extracts multi-scale features for conditioning)
# ─────────────────────────────────────────────

class MWIREncoder(nn.Module):
    """
    Lightweight encoder that extracts multi-scale spatial features from the
    MWIR conditioning image.  Features are injected via cross-modal attention
    at each decoder stage.
    """
    def __init__(self, in_channels: int, base_channels: int, channel_mults: tuple):
        super().__init__()
        self.stem = spectral_conv(in_channels, base_channels)
        self.downs = nn.ModuleList()
        c = base_channels
        for mult in channel_mults[:-1]:
            out_c = base_channels * mult
            self.downs.append(nn.Sequential(
                spectral_conv(c, out_c),
                nn.SiLU(),
                Downsample(out_c),
            ))
            c = out_c
        # bottleneck
        final_mult = channel_mults[-1]
        self.bottleneck = nn.Sequential(
            spectral_conv(c, base_channels * final_mult),
            nn.SiLU(),
        )

    def forward(self, mwir: torch.Tensor):
        feats = []
        x = self.stem(mwir)
        feats.append(x)
        for down in self.downs:
            x = down(x)
            feats.append(x)
        x = self.bottleneck(x)
        feats.append(x)
        return feats  # list of (B, C, H, W) at decreasing resolutions


# ─────────────────────────────────────────────
# Main Conditional UNet
# ─────────────────────────────────────────────

class ConditionalUNet(nn.Module):
    """
    Full conditional UNet for MWIR→LWIR diffusion.

    Args:
        in_channels:     noisy LWIR channels (typically 1)
        mwir_channels:   MWIR input channels
        base_channels:   feature map width at full resolution
        channel_mults:   multipliers per resolution level
        attn_resolutions: resolutions at which to apply self-attention
        num_res_blocks:  residual blocks per resolution
        dropout:         dropout rate
        use_cross_attn:  inject MWIR via cross-modal attention in decoder
    """

    def __init__(
        self,
        in_channels: int = 1,
        mwir_channels: int = 1,
        base_channels: int = 128,
        channel_mults: tuple = (1, 2, 4, 8),
        attn_resolutions: tuple = (16, 8),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        use_cross_attn: bool = True,
        image_size: int = 256,
    ):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.num_res_blocks = num_res_blocks          # stored for forward()
        context_dim = base_channels * 4  # timestep emb dim

        # Timestep embedding
        self.time_embed = FourierTimestepEmbedding(context_dim)

        # MWIR encoder
        self.mwir_enc = MWIREncoder(mwir_channels, base_channels, channel_mults)

        # Concat MWIR at input (early fusion)
        self.input_conv = spectral_conv(in_channels + mwir_channels, base_channels)

        # ── Encoder ──
        self.enc_blocks = nn.ModuleList()
        self.enc_downs  = nn.ModuleList()
        self.enc_attns  = nn.ModuleList()

        # channels tracks skip-connection channel widths in push order.
        # The decoder pops (num_res_blocks + 1) times per level in reverse,
        # so we must push (num_res_blocks + 1) entries per level:
        #   • num_res_blocks  entries: one after each res block
        #   • 1 bridge entry:          the level's final feature, pushed
        #                              BEFORE the downsample
        # Exception: the last (deepest) level feeds the bottleneck directly
        # and has no downsample, so it does not need a bridge entry.
        channels = [base_channels]                    # stem output is first skip
        cur_channels = base_channels
        cur_res = image_size

        for level, mult in enumerate(channel_mults):
            out_c = base_channels * mult
            for _ in range(num_res_blocks):
                self.enc_blocks.append(ResBlock(cur_channels, out_c, context_dim, dropout=dropout))
                if cur_res in attn_resolutions:
                    self.enc_attns.append(SelfAttentionBlock(out_c))
                else:
                    self.enc_attns.append(nn.Identity())
                channels.append(out_c)
                cur_channels = out_c

            is_last = (level == len(channel_mults) - 1)
            if not is_last:
                # Bridge entry: pushed before downsampling, consumed by the
                # "+1" decoder block at this resolution level.
                channels.append(cur_channels)
                self.enc_downs.append(Downsample(cur_channels))
                cur_res //= 2
            else:
                self.enc_downs.append(nn.Identity())

        # ── Bottleneck ──
        self.mid_block1 = ResBlock(cur_channels, cur_channels, context_dim, dropout=dropout)
        self.mid_attn   = SelfAttentionBlock(cur_channels)
        self.mid_block2 = ResBlock(cur_channels, cur_channels, context_dim, dropout=dropout)

        # ── Decoder ──
        self.dec_blocks = nn.ModuleList()
        self.dec_ups    = nn.ModuleList()
        self.dec_attns  = nn.ModuleList()
        self.cross_attns = nn.ModuleList() if use_cross_attn else None

        # MWIR encoder channel sizes for cross-attn (one per decoder level)
        mwir_enc_channels = (
            [base_channels]
            + [base_channels * m for m in channel_mults[:-1]]
            + [base_channels * channel_mults[-1]]
        )

        for level, mult in reversed(list(enumerate(channel_mults))):
            out_c = base_channels * mult
            # num_res_blocks + 1 blocks per decoder level (the +1 consumes
            # the bridge entry pushed by the corresponding encoder level)
            for i in range(num_res_blocks + 1):
                skip_c = channels.pop()               # now always succeeds
                self.dec_blocks.append(ResBlock(cur_channels + skip_c, out_c, context_dim, dropout=dropout))
                if cur_res in attn_resolutions:
                    self.dec_attns.append(SelfAttentionBlock(out_c))
                else:
                    self.dec_attns.append(nn.Identity())
                if use_cross_attn:
                    ctx_c = mwir_enc_channels[min(level, len(mwir_enc_channels) - 1)]
                    self.cross_attns.append(CrossModalAttention(out_c, ctx_c))
                cur_channels = out_c

            if level > 0:
                self.dec_ups.append(Upsample(cur_channels))
                cur_res *= 2
            else:
                self.dec_ups.append(nn.Identity())

        assert len(channels) == 0, (
            f"channels stack not fully consumed: {len(channels)} entries remain. "
            "This is a bug in the UNet init — encoder pushes and decoder pops must balance."
        )

        # Output
        self.out_norm = nn.GroupNorm(32, base_channels)
        self.out_conv = zero_module(spectral_conv(base_channels, in_channels))

    def forward(
        self,
        x: torch.Tensor,       # (B, 1, H, W) noisy LWIR
        t: torch.Tensor,       # (B,) timestep indices
        mwir: torch.Tensor,    # (B, 1, H, W) conditioning MWIR
    ) -> torch.Tensor:
        # ── Context vectors ──────────────────────────────────────
        ctx        = self.time_embed(t)        # (B, context_dim)
        mwir_feats = self.mwir_enc(mwir)       # list of (B, C, H, W) at decreasing resolutions

        # ── Stem (early fusion) ───────────────────────────────────
        h = self.input_conv(torch.cat([x, mwir], dim=1))

        # ── Encoder ──────────────────────────────────────────────
        # Push the stem output as the very first skip.
        skips   = [h]
        blk_i   = 0

        for level_i, down in enumerate(self.enc_downs):
            is_last = (level_i == len(self.enc_downs) - 1)

            # num_res_blocks res blocks — push a skip after each
            for _ in range(self.num_res_blocks):
                h = self.enc_blocks[blk_i](h, ctx)
                h = self.enc_attns[blk_i](h)
                skips.append(h)
                blk_i += 1

            # Bridge skip: push h BEFORE downsampling so the decoder's
            # "+1" block at this resolution has a skip to consume.
            # Only non-last levels actually downsample.
            if not is_last:
                skips.append(h)   # bridge — same tensor as last res block output
                h = down(h)       # Downsample
            # last level: down is nn.Identity(), no bridge needed

        # ── Bottleneck ────────────────────────────────────────────
        h = self.mid_block1(h, ctx)
        h = self.mid_attn(h)
        h = self.mid_block2(h, ctx)

        # ── Decoder ──────────────────────────────────────────────
        num_levels   = len(self.enc_downs)
        dec_i        = 0
        cross_i      = 0

        for level_i, up in enumerate(self.dec_ups):
            # level_i=0 corresponds to the deepest (highest-channel) decoder level
            # which is the mirror of the last encoder level.
            enc_level = num_levels - 1 - level_i   # encoder level this mirrors

            for j in range(self.num_res_blocks + 1):
                skip = skips.pop()
                h    = torch.cat([h, skip], dim=1)
                h    = self.dec_blocks[dec_i](h, ctx)
                h    = self.dec_attns[dec_i](h)

                if self.use_cross_attn and cross_i < len(self.cross_attns):
                    # Use MWIR features from the matching encoder level.
                    # mwir_feats is ordered coarsest-to-finest so index from end.
                    mwir_feat_idx = min(enc_level, len(mwir_feats) - 1)
                    mwir_f = mwir_feats[mwir_feat_idx]
                    if mwir_f.shape[-2:] != h.shape[-2:]:
                        mwir_f = F.interpolate(
                            mwir_f, size=h.shape[-2:],
                            mode='bilinear', align_corners=False,
                        )
                    h = self.cross_attns[cross_i](h, mwir_f)
                    cross_i += 1

                dec_i += 1

            h = up(h)   # Upsample (or Identity at the shallowest level)

        return self.out_conv(F.silu(self.out_norm(h)))
