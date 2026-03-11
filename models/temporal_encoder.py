"""Population-level temporal modelling: bin encoding + temporal transformer."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_seq_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to *x*.  Shape: (B, T, D)."""
        return x + self.pe[:, : x.size(1), :]


class ContinuousTimeEncoding(nn.Module):
    """Sinusoidal encoding based on continuous time values (seconds).

    Instead of using integer sequence positions, this encodes the actual
    timestamp of each bin, allowing the model to understand irregular
    temporal spacing.
    """

    def __init__(self, d_model: int, max_period: float = 3600.0) -> None:
        super().__init__()
        self.d_model = d_model
        # Precompute frequency bands
        half = d_model // 2
        freqs = torch.exp(
            torch.arange(0, half, dtype=torch.float32)
            * (-math.log(max_period) / half)
        )
        self.register_buffer("freqs", freqs)  # (half,)

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        """Encode time values into sinusoidal features.

        Args:
            times: (B, T) time values in seconds.

        Returns:
            (B, T, d_model) time encoding.
        """
        # (B, T, 1) * (1, 1, half) → (B, T, half)
        angles = times.unsqueeze(-1) * self.freqs.unsqueeze(0).unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


# ---------------------------------------------------------------------------
# Population bin encoder
# ---------------------------------------------------------------------------

class AttentionBinEncoder(nn.Module):
    """Attention-based bin encoder that pools over crops using learned attention.

    Instead of computing hand-crafted statistics (mean, std, skewness,
    kurtosis), this learns to attend to relevant crops and aggregate
    their features via cross-attention with a learned query.

    Pipeline
    --------
    1. Project crop features: Linear(feat_dim → hidden_dim) + LN + GELU.
    2. Cross-attention: learned query attends to projected crops → (B, hidden_dim).
    3. Append normalised crop count → (B, hidden_dim + 1).
    4. Output projection → (B, hidden_dim).
    """

    def __init__(
        self,
        feat_dim: int = 384,
        hidden_dim: int = 256,
        num_heads: int = 4,
        max_count_normalizer: float = 256.0,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.max_count_normalizer = max_count_normalizer

        # Project each crop feature to hidden_dim
        self.crop_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Learned query seed for pooling (1 query → 1 output per bin)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Cross-attention: query attends to crop features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Final projection incorporating count
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        crop_features: torch.Tensor,
        crop_mask: torch.Tensor | None = None,
        crop_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a batch of bins via attention pooling.

        Args:
            crop_features: (B, N, feat_dim) features for crops in each bin.
            crop_mask:     (B, N) boolean mask for valid crops.
            crop_counts:   (B,) number of crops per bin.

        Returns:
            (B, hidden_dim) bin embedding.
        """
        B = crop_features.shape[0]

        # Project crops
        h = self.crop_proj(crop_features)  # (B, N, hidden_dim)

        # Expand query for batch
        query = self.query.expand(B, -1, -1)  # (B, 1, hidden_dim)

        # Key padding mask (True = ignore for nn.MultiheadAttention)
        key_padding_mask = ~crop_mask if crop_mask is not None else None

        # Cross-attention pooling
        pooled, _ = self.cross_attn(
            query, h, h,
            key_padding_mask=key_padding_mask,
        )  # (B, 1, hidden_dim)
        pooled = self.attn_norm(pooled.squeeze(1))  # (B, hidden_dim)

        # Append normalised count
        if crop_counts is not None:
            norm_count = (crop_counts / self.max_count_normalizer).unsqueeze(-1)
        elif crop_mask is not None:
            norm_count = (crop_mask.float().sum(dim=1) / self.max_count_normalizer).unsqueeze(-1)
        else:
            norm_count = torch.ones(B, 1, device=crop_features.device)

        combined = torch.cat([pooled, norm_count], dim=-1)  # (B, hidden_dim + 1)
        return self.out_proj(combined)  # (B, hidden_dim)


class PopulationBinEncoder(nn.Module):
    """Encode a population of bacteria features within a single time bin.

    For each bin, computes population statistics (mean, std, skewness,
    kurtosis) over the crop features, concatenates with a normalised
    crop count, and projects to a fixed-size bin embedding.

    Pipeline
    --------
    1. Compute masked mean, std, skewness, kurtosis → (B, 4 * feat_dim).
    2. Append normalised crop count → (B, 4 * feat_dim + 1).
    3. Project via MLP → (B, hidden_dim).
    """

    def __init__(
        self,
        feat_dim: int = 384,
        hidden_dim: int = 256,
        max_count_normalizer: float = 256.0,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.max_count_normalizer = max_count_normalizer
        self.proj = nn.Sequential(
            nn.Linear(feat_dim * 4 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def _masked_stats(
        h: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute masked mean, std, skewness, kurtosis.

        Args:
            h:    (B, N, D) crop features within a bin.
            mask: (B, N) boolean mask — True for valid crops.

        Returns:
            mean, std, skewness, kurtosis — each (B, D).
        """
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            count = m.sum(dim=1).clamp(min=1.0)
        else:
            m = torch.ones(
                h.shape[0], h.shape[1], 1, device=h.device, dtype=h.dtype
            )
            count = torch.full(
                (h.shape[0], 1), h.shape[1], device=h.device, dtype=h.dtype
            )

        mean = (h * m).sum(dim=1) / count
        diff = (h - mean.unsqueeze(1)) * m
        var = (diff ** 2).sum(dim=1) / count.clamp(min=1.0)
        std = (var + 1e-8).sqrt()
        m3 = (diff ** 3).sum(dim=1) / count.clamp(min=1.0)
        skewness = m3 / (std ** 3 + 1e-8)
        m4 = (diff ** 4).sum(dim=1) / count.clamp(min=1.0)
        kurtosis = m4 / (std ** 4 + 1e-8) - 3.0

        return mean, std, skewness, kurtosis

    def compute_stats(
        self,
        crop_features: torch.Tensor,
        crop_mask: torch.Tensor | None = None,
        crop_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute raw population statistics without projection.

        Args:
            crop_features: (B, N, feat_dim) features for crops in each bin.
            crop_mask:     (B, N) boolean mask for valid crops.
            crop_counts:   (B,) number of crops per bin (for count feature).

        Returns:
            (B, 4*feat_dim + 1) raw concatenated stats.
        """
        mean, std, skewness, kurtosis = self._masked_stats(crop_features, crop_mask)
        stats = torch.cat([mean, std, skewness, kurtosis], dim=-1)  # (B, 4*D)

        if crop_counts is not None:
            norm_count = (crop_counts / self.max_count_normalizer).unsqueeze(-1)
        else:
            if crop_mask is not None:
                norm_count = (crop_mask.float().sum(dim=1) / self.max_count_normalizer).unsqueeze(-1)
            else:
                norm_count = torch.ones(stats.shape[0], 1, device=stats.device)

        return torch.cat([stats, norm_count], dim=-1)  # (B, 4*D + 1)

    def forward(
        self,
        crop_features: torch.Tensor,
        crop_mask: torch.Tensor | None = None,
        crop_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a batch of bins.

        Args:
            crop_features: (B, N, feat_dim) features for crops in each bin.
            crop_mask:     (B, N) boolean mask for valid crops.
            crop_counts:   (B,) number of crops per bin (for count feature).

        Returns:
            (B, hidden_dim) bin embedding.
        """
        combined = self.compute_stats(crop_features, crop_mask, crop_counts)
        return self.proj(combined)  # (B, hidden_dim)


# ---------------------------------------------------------------------------
# Population temporal encoder
# ---------------------------------------------------------------------------

class PopulationTemporalEncoder(nn.Module):
    """Transformer-based temporal encoder over population bin embeddings.

    Models how the population-level feature distribution changes over
    the course of an experiment.

    Pipeline
    --------
    1. Add continuous-time positional encoding (based on bin center times).
    2. Pass through a 4-layer Transformer encoder.
    3. Mean-pool over valid bins → (B, hidden_dim).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        max_period: float = 3600.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.time_enc = ContinuousTimeEncoding(hidden_dim, max_period=max_period)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(
        self,
        bin_embeddings: torch.Tensor,
        bin_times: torch.Tensor,
        bin_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            bin_embeddings: (B, T, hidden_dim) per-bin population embeddings.
            bin_times:      (B, T) bin center times in seconds.
            bin_mask:       (B, T) boolean mask — True for valid bins.

        Returns:
            (B, hidden_dim) pooled temporal representation.
        """
        # Add continuous time encoding
        time_features = self.time_enc(bin_times)  # (B, T, hidden_dim)
        h = bin_embeddings + self.time_proj(time_features)

        # Build key-padding mask (True = ignore)
        if bin_mask is not None:
            src_key_padding_mask = ~bin_mask
        else:
            src_key_padding_mask = None

        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)

        # Mean-pool over valid bins
        if bin_mask is not None:
            mask_expanded = bin_mask.unsqueeze(-1).float()
            lengths = mask_expanded.sum(dim=1).clamp(min=1.0)
            pooled = (h * mask_expanded).sum(dim=1) / lengths
        else:
            pooled = h.mean(dim=1)

        return pooled
