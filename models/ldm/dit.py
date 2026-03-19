"""
Conditional Diffusion Transformer (DiT) for latent MWIR→LWIR synthesis.

Architecture: DiT-B/4 adapted for cross-modal EO conditioning.

Key differences from vanilla DiT (Peebles & Xie 2023):
  1. Dual conditioning:
       - adaLN-Zero  ← global scene context (timestep + MWIR global stats)
       - Cross-attention ← local MWIR spatial features at latent resolution
  2. RoPE (Rotary Position Embeddings) instead of learned 2D sine/cos
     → Better generalises to different image sizes / patch grids
  3. Register tokens (Darcet et al. 2023) to prevent attention sink artefacts
     which manifest as checkerboard patterns in low-contrast IR regions
  4. QK-Norm for training stability on heterogeneous thermal statistics

Latent input:  (B, 4, 64, 64) for 256×256 input with f=4 VAE
Patch size:    4 → sequence length = (64/4)² = 256 tokens
               (manageable on single A100; could go patch=2 on cluster)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple


# ─────────────────────────────────────────────
# Rotary Position Embedding (2D)
# ─────────────────────────────────────────────

class RotaryEmbedding2D(nn.Module):
    """
    2D Rotary Position Embedding.
    Applied separately to H and W axes of the patchified latent.
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 4 == 0, "dim must be divisible by 4 for 2D RoPE"
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim // 2, 2).float() / (dim // 2)))
        self.register_buffer('inv_freq', inv_freq)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """x: (B, heads, N, dim_head), N = H*W"""
        device = x.device
        h_pos = torch.arange(H, device=device).float()
        w_pos = torch.arange(W, device=device).float()

        h_sin = torch.outer(h_pos, self.inv_freq).sin()
        h_cos = torch.outer(h_pos, self.inv_freq).cos()
        w_sin = torch.outer(w_pos, self.inv_freq).sin()
        w_cos = torch.outer(w_pos, self.inv_freq).cos()

        # Tile to patch grid
        h_sin = h_sin.unsqueeze(1).expand(-1, W, -1).reshape(H*W, -1)
        h_cos = h_cos.unsqueeze(1).expand(-1, W, -1).reshape(H*W, -1)
        w_sin = w_sin.unsqueeze(0).expand(H, -1, -1).reshape(H*W, -1)
        w_cos = w_cos.unsqueeze(0).expand(H, -1, -1).reshape(H*W, -1)

        # Concatenate H and W embeddings
        sin = torch.cat([h_sin, w_sin], dim=-1).unsqueeze(0).unsqueeze(0)
        cos = torch.cat([h_cos, w_cos], dim=-1).unsqueeze(0).unsqueeze(0)

        # Apply to query/key
        d = sin.shape[-1]
        x_rot = x[..., :d] * cos + self._rotate_half(x[..., :d]) * sin
        x_pass = x[..., d:]
        return torch.cat([x_rot, x_pass], dim=-1)


# ─────────────────────────────────────────────
# adaLN-Zero modulation
# ─────────────────────────────────────────────

class AdaLNZero(nn.Module):
    """
    Adaptive LayerNorm-Zero conditioning (DiT paper).
    Projects global conditioning vector to (scale, shift, gate) × 2
    (one set for self-attn, one for FFN).

    The 'Zero' means the gate is initialised to 0 → identity at init,
    making training more stable especially for IR data.
    """
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_dim),
        )
        # Zero-init the output projection
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """Returns modulated x and (alpha_attn, alpha_ffn) gates."""
        params = self.proj(c).unsqueeze(1)   # (B, 1, 6*H)
        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = params.chunk(6, dim=-1)
        x_norm = self.norm(x)
        x_sa = x_norm * (1 + scale_sa) + shift_sa
        x_ff = x_norm * (1 + scale_ff) + shift_ff
        return x_sa, x_ff, gate_sa, gate_ff


# ─────────────────────────────────────────────
# Cross-Modal Attention (MWIR spatial features)
# ─────────────────────────────────────────────

class CrossModalAttentionDiT(nn.Module):
    """
    Multi-head cross-attention: latent noise tokens attend to MWIR features.

    Uses QK-Norm for training stability — critical for IR data where
    attention logits can spike on high-contrast targets (hot pixels).
    """
    def __init__(self, hidden_dim: int, context_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(context_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(context_dim, hidden_dim, bias=False)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)

        # QK-Norm
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_ctx = nn.LayerNorm(context_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm_q(x)
        context = self.norm_ctx(context)

        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(self.to_k(context), 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(self.to_v(context), 'b n (h d) -> b h n d', h=self.num_heads)

        q = self.q_norm(q)
        k = self.k_norm(k)

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return residual + self.to_out(out)


# ─────────────────────────────────────────────
# Self-Attention with RoPE + QK-Norm
# ─────────────────────────────────────────────

class SelfAttentionRoPE(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.to_qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.rope = RotaryEmbedding2D(self.head_dim)

    def forward(self, x: torch.Tensor, H: int, W: int,
                n_registers: int = 0) -> torch.Tensor:
        """
        x: (B, N_total, hidden_dim)  where N_total = n_registers + H*W

        RoPE is spatial — it must be applied only to the H*W patch tokens.
        Register tokens sit at positions [:n_registers] and have no spatial
        meaning, so we slice them off, rotate the patch slice, then concat.
        """
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if n_registers > 0:
            # Split: register slice has no position, patch slice gets RoPE
            q_reg,   q_patch   = q[:, :, :n_registers], q[:, :, n_registers:]
            k_reg,   k_patch   = k[:, :, :n_registers], k[:, :, n_registers:]

            q_patch = self.rope(q_patch, H, W)
            k_patch = self.rope(k_patch, H, W)

            q = torch.cat([q_reg, q_patch], dim=2)
            k = torch.cat([k_reg, k_patch], dim=2)
        else:
            q = self.rope(q, H, W)
            k = self.rope(k, H, W)

        out = F.scaled_dot_product_attention(q, k, v)
        return self.to_out(rearrange(out, 'b h n d -> b n (h d)'))


# ─────────────────────────────────────────────
# Feed-Forward Network
# ─────────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, hidden_dim: int, expand: int = 4):
        super().__init__()
        inner = hidden_dim * expand
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.GELU(),
            nn.Linear(inner, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# DiT Block
# ─────────────────────────────────────────────

class DiTBlock(nn.Module):
    """
    Single DiT block with:
      1. adaLN-Zero self-attention (modulated by global cond)
      2. Cross-modal attention (attends to MWIR spatial features)
      3. adaLN-Zero FFN
    """
    def __init__(self, hidden_dim: int, num_heads: int, context_dim: int, cond_dim: int):
        super().__init__()
        self.adaln = AdaLNZero(hidden_dim, cond_dim)
        self.self_attn = SelfAttentionRoPE(hidden_dim, num_heads)
        self.cross_attn = CrossModalAttentionDiT(hidden_dim, context_dim, num_heads)
        self.ffn = FFN(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        context: torch.Tensor,
        H: int,
        W: int,
        n_registers: int = 0,
    ) -> torch.Tensor:
        x_sa, x_ff, gate_sa, gate_ff = self.adaln(x, c)

        # Self-attention (with adaLN-Zero gate)
        # n_registers tells SelfAttentionRoPE how many leading tokens to skip
        x = x + gate_sa * self.self_attn(x_sa, H, W, n_registers=n_registers)

        # Cross-attention to MWIR (no gate — always active)
        x = self.cross_attn(x, context)

        # FFN (with adaLN-Zero gate)
        x = x + gate_ff * self.ffn(x_ff)
        return x


# ─────────────────────────────────────────────
# Global Conditioning (timestep + MWIR global)
# ─────────────────────────────────────────────

class GlobalConditioner(nn.Module):
    """
    Produces the global conditioning vector c for adaLN-Zero.
    Fuses:
      - Fourier timestep embedding
      - MWIR global scene statistics (mean, std, skew, kurtosis, percentiles)
    """
    def __init__(self, cond_dim: int, mwir_channels: int = 1):
        super().__init__()
        # Timestep
        self.t_freq = 256
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_freq * 2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        # MWIR global stats (8 stats × mwir_channels)
        n_stats = 8 * mwir_channels
        self.stat_proj = nn.Sequential(
            nn.Linear(n_stats, cond_dim // 2),
            nn.SiLU(),
            nn.Linear(cond_dim // 2, cond_dim),
        )
        self.fuse = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.SiLU(),
        )
        # Register buffer for Fourier freqs
        freqs = torch.randn(self.t_freq) * 0.1
        self.register_buffer('freqs', freqs)

    def _mwir_stats(self, mwir: torch.Tensor) -> torch.Tensor:
        B = mwir.shape[0]
        flat = mwir.reshape(B, mwir.shape[1], -1)   # (B, C, N)
        mean = flat.mean(-1)
        std = flat.std(-1) + 1e-8
        skew = ((flat - mean.unsqueeze(-1)) / std.unsqueeze(-1)).pow(3).mean(-1)
        kurt = ((flat - mean.unsqueeze(-1)) / std.unsqueeze(-1)).pow(4).mean(-1) - 3
        p10 = flat.quantile(0.10, dim=-1)
        p25 = flat.quantile(0.25, dim=-1)
        p75 = flat.quantile(0.75, dim=-1)
        p90 = flat.quantile(0.90, dim=-1)
        return torch.cat([mean, std, skew, kurt, p10, p25, p75, p90], dim=-1)

    def forward(self, t: torch.Tensor, mwir: torch.Tensor) -> torch.Tensor:
        # Timestep Fourier embedding
        t_f = t.float().unsqueeze(-1) * self.freqs.unsqueeze(0) * 2 * math.pi
        t_emb = self.t_proj(torch.cat([t_f.sin(), t_f.cos()], dim=-1))

        # MWIR global stats
        stats = self._mwir_stats(mwir)
        stat_emb = self.stat_proj(stats)

        return self.fuse(torch.cat([t_emb, stat_emb], dim=-1))


# ─────────────────────────────────────────────
# MWIR Spatial Feature Extractor
# ─────────────────────────────────────────────

class MWIRSpatialEncoder(nn.Module):
    """
    Extracts spatial features from MWIR at the latent resolution (H/4, W/4).
    Output is the cross-attention context sequence for DiT blocks.

    Uses a simple CNN — deeper than needed would overfit on 500-2000 pairs.
    """
    def __init__(self, in_channels: int = 1, context_dim: int = 512, f: int = 4):
        super().__init__()
        layers = []
        ch = 64
        c_in = in_channels
        n_down = int(math.log2(f))   # f=4 → 2 downsamples
        for i in range(n_down):
            c_out = ch * (2 ** i)
            layers += [
                nn.utils.spectral_norm(nn.Conv2d(c_in, c_out, 3, stride=2, padding=1)),
                nn.SiLU(),
                nn.utils.spectral_norm(nn.Conv2d(c_out, c_out, 3, padding=1)),
                nn.SiLU(),
            ]
            c_in = c_out
        layers += [
            nn.Conv2d(c_in, context_dim, 1),
        ]
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(context_dim)

    def forward(self, mwir: torch.Tensor) -> torch.Tensor:
        """Returns (B, H_lat * W_lat, context_dim) token sequence."""
        feat = self.net(mwir)                   # (B, context_dim, H/f, W/f)
        B, C, H, W = feat.shape
        feat = feat.reshape(B, C, H*W).permute(0, 2, 1)   # (B, N, C)
        return self.norm(feat)


# ─────────────────────────────────────────────
# Full Conditional DiT
# ─────────────────────────────────────────────

class ConditionalDiT(nn.Module):
    """
    Conditional Diffusion Transformer for MWIR→LWIR latent synthesis.

    DiT-B/4 configuration:
      hidden_dim=768, num_heads=12, depth=12  → ~86M parameters

    DiT-S/4 (lighter, for ablation):
      hidden_dim=384, num_heads=6, depth=6   → ~22M parameters

    Args:
        in_channels:    latent channels (4 from VAE)
        patch_size:     spatial patch size for sequence (4 recommended for f=4 VAE)
        hidden_dim:     transformer hidden dimension
        depth:          number of DiT blocks
        num_heads:      attention heads
        context_dim:    MWIR spatial feature dim (cross-attn)
        cond_dim:       global conditioning dim (adaLN-Zero)
        mwir_channels:  MWIR input channels
        vae_f:          VAE spatial compression factor (4)
        num_registers:  register tokens to prevent attention sink artefacts
    """

    def __init__(
        self,
        in_channels: int = 4,
        patch_size: int = 4,
        hidden_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        context_dim: int = 512,
        cond_dim: int = 1024,
        mwir_channels: int = 1,
        vae_f: int = 4,
        num_registers: int = 4,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_registers = num_registers
        self.hidden_dim = hidden_dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # Register tokens (prevent attention collapse on flat IR regions)
        self.register_tokens = nn.Parameter(torch.randn(1, num_registers, hidden_dim) * 0.02)

        # Global conditioner (timestep + MWIR stats → adaLN-Zero)
        self.global_cond = GlobalConditioner(cond_dim, mwir_channels)

        # MWIR spatial encoder (→ cross-attention context)
        self.mwir_encoder = MWIRSpatialEncoder(mwir_channels, context_dim, vae_f)

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, context_dim, cond_dim)
            for _ in range(depth)
        ])

        # Final layer (adaLN-Zero → unpatchify)
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_dim),
        )
        nn.init.zeros_(self.final_adaln[-1].weight)
        nn.init.zeros_(self.final_adaln[-1].bias)
        self.final_proj = nn.Linear(hidden_dim, patch_size * patch_size * in_channels)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def unpatchify(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        p = self.patch_size
        c = self.in_channels
        x = x.reshape(x.shape[0], H, W, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.reshape(x.shape[0], c, H * p, W * p)

    def forward(
        self,
        z: torch.Tensor,        # (B, 4, H_lat, W_lat) noisy latent
        t: torch.Tensor,        # (B,) timestep
        mwir: torch.Tensor,     # (B, 1, H, W) full-res MWIR
    ) -> torch.Tensor:

        B, C, H_lat, W_lat = z.shape
        p = self.patch_size

        # Patchify latent: (B, 4, H_lat, W_lat) → (B, N, hidden_dim)
        x = self.patch_embed(z)                             # (B, hidden_dim, H_lat/p, W_lat/p)
        pH, pW = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)                   # (B, pH*pW, hidden_dim)

        # Prepend register tokens
        reg = repeat(self.register_tokens, '1 r d -> b r d', b=B)
        x = torch.cat([reg, x], dim=1)                     # (B, R+N, hidden_dim)
        N_total = x.shape[1]

        # Global conditioning: timestep + MWIR stats
        c = self.global_cond(t, mwir)                      # (B, cond_dim)

        # MWIR spatial context for cross-attention
        ctx = self.mwir_encoder(mwir)                      # (B, H_lat*W_lat/vae_f², context_dim)

        # DiT blocks
        for block in self.blocks:
            x = block(x, c, ctx, pH, pW, n_registers=self.num_registers)

        # Strip register tokens
        x = x[:, self.num_registers:, :]                   # (B, N, hidden_dim)

        # Final modulation
        params = self.final_adaln(c).unsqueeze(1)
        shift, scale = params.chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale) + shift

        # Project and unpatchify
        x = self.final_proj(x)                             # (B, N, p*p*C)
        return self.unpatchify(x, pH, pW)                  # (B, C, H_lat, W_lat)


# ─────────────────────────────────────────────
# Model configurations
# ─────────────────────────────────────────────

def DiT_S_4(**kwargs):
    """Small: ~22M params. Good for ablation and limited GPU."""
    return ConditionalDiT(hidden_dim=384, depth=6, num_heads=6, patch_size=4, **kwargs)

def DiT_B_4(**kwargs):
    """Base: ~86M params. Recommended for A100 cluster with 500-2000 pairs."""
    return ConditionalDiT(hidden_dim=768, depth=12, num_heads=12, patch_size=4, **kwargs)

def DiT_L_4(**kwargs):
    """Large: ~178M params. Use if you can get >5k pairs or heavy augmentation."""
    return ConditionalDiT(hidden_dim=1024, depth=16, num_heads=16, patch_size=4, **kwargs)
