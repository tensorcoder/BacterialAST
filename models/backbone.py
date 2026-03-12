"""ViT-Small backbone adapted for 96x96 grayscale microscopy images."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic depth (drop path) regularisation.

    During training, each sample's residual branch is dropped with probability
    *drop_prob*.  At test time the module is an identity.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Shape (B, 1, 1, ...) – works for any number of trailing dims.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


class PatchEmbed(nn.Module):
    """Convert a 2-D image into a sequence of flattened patch embeddings."""

    def __init__(
        self,
        img_size: int = 96,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 6*6 = 36
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor.

        Returns:
            (B, num_patches, embed_dim) patch embeddings.
        """
        # (B, embed_dim, H/P, W/P) -> (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    """Two-layer MLP used inside each Transformer block."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # each (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: LN -> MHSA -> residual -> LN -> MLP -> residual."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=dropout,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal encoding for a scalar continuous time value.

    Maps a scalar ``t`` (e.g. seconds since experiment start) to a
    ``dim``-dimensional vector using log-spaced sinusoidal frequency bands,
    similar to the positional encoding of Vaswani et al. (2017) but applied
    to a single continuous value rather than integer positions.
    """

    def __init__(self, dim: int, max_period: float = 3600.0) -> None:
        super().__init__()
        half = dim // 2
        freqs = torch.exp(
            torch.arange(0, half, dtype=torch.float32)
            * (-math.log(max_period) / half)
        )
        self.register_buffer("freqs", freqs)  # (half,)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) scalar time values (seconds).

        Returns:
            (B, dim) sinusoidal embedding.
        """
        args = t.unsqueeze(-1) * self.freqs  # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ViTSmall(nn.Module):
    """Vision Transformer – Small variant for 96x96 single-channel microscopy.

    Architecture
    ------------
    * Patch size 16 → 6x6 = 36 patches
    * 1 prepended CLS token → sequence length 37
    * 12 Transformer blocks, embed_dim 384, 6 heads
    * Stochastic depth with linearly increasing drop rate (max 0.1)
    * Optional time conditioning: sinusoidal time embedding added to all tokens

    Returns
    -------
    ``forward(x)`` → CLS token representation  (B, 384)
    ``forward_features(x)`` → all token representations (B, 37, 384)
    """

    def __init__(
        self,
        img_size: int = 96,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.1,
        time_conditioned: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2  # 36
        self.time_conditioned = time_conditioned

        # ---------- patch embedding ----------
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # ---------- special tokens & positional embeddings ----------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.num_patches, embed_dim)  # (1, 37, 384)
        )
        self.pos_drop = nn.Dropout(p=dropout)

        # ---------- optional time conditioning ----------
        if time_conditioned:
            self.time_sinusoidal = SinusoidalTimeEmbedding(embed_dim)
            self.time_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )

        # ---------- transformer blocks ----------
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # linearly increasing drop path
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    attn_drop=attn_drop,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # ---------- weight initialisation ----------
        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module)

    @staticmethod
    def _init_module(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            # Fan-out initialisation (like ResNet) for the patch-embed conv.
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    def _interpolate_pos_embed(self, x: torch.Tensor, num_patches: int) -> torch.Tensor:
        """Interpolate positional embeddings for inputs with different patch counts."""
        if num_patches == self.num_patches:
            return self.pos_embed
        cls_pos = self.pos_embed[:, :1]  # (1, 1, D)
        patch_pos = self.pos_embed[:, 1:]  # (1, N, D)
        dim = patch_pos.shape[-1]
        orig_size = int(self.num_patches ** 0.5)
        new_size = int(num_patches ** 0.5)
        patch_pos = patch_pos.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(new_size, new_size), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def _embed(
        self, x: torch.Tensor, time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Patch-embed, prepend CLS, add positional + time embeddings."""
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, D)
        num_patches = x.shape[1]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)
        pos_embed = self._interpolate_pos_embed(x, num_patches)
        x = x + pos_embed

        # Add time conditioning: broadcast to all tokens
        if time is not None and self.time_conditioned:
            time_emb = self.time_proj(self.time_sinusoidal(time))  # (B, D)
            x = x + time_emb.unsqueeze(1)  # (B, 1, D) broadcast

        x = self.pos_drop(x)
        return x

    def forward_features(
        self, x: torch.Tensor, time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return all token representations including CLS.

        Args:
            x: (B, 1, 96, 96) grayscale microscopy image.
            time: (B,) optional time in seconds since experiment start.

        Returns:
            (B, 37, 384) – CLS at index 0 followed by 36 patch tokens.
        """
        x = self._embed(x, time=time)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(
        self, x: torch.Tensor, time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the CLS token representation.

        Args:
            x: (B, 1, 96, 96) grayscale microscopy image.
            time: (B,) optional time in seconds since experiment start.

        Returns:
            (B, 384) CLS feature vector.
        """
        return self.forward_features(x, time=time)[:, 0]
