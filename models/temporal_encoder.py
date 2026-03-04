"""Per-bacterium temporal modelling: delta features + Transformer encoder."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Delta feature computation
# ---------------------------------------------------------------------------

class DeltaFeatureComputer(nn.Module):
    """Compute multi-scale temporal differences and project back to the
    original feature dimension.

    For each scale *k* in ``{1, 5, 25, 125}``, the delta is

        delta_k[t] = feature[t] - feature[t - k]

    The first *k* positions of each scale are zero-padded.  The four delta
    tensors are concatenated along the feature axis (giving ``4 * feat_dim``)
    and then linearly projected back to *feat_dim*.
    """

    SCALES = (1, 5, 25, 125)

    def __init__(self, feat_dim: int = 384) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.proj = nn.Linear(feat_dim * len(self.SCALES), feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, feat_dim) – per-bacterium feature sequence.

        Returns:
            (B, T, feat_dim) – projected multi-scale delta features.
        """
        B, T, D = x.shape
        deltas: list[torch.Tensor] = []

        for k in self.SCALES:
            if k >= T:
                # If the scale is larger than the sequence, the entire delta
                # is zero (no valid look-back positions).
                deltas.append(torch.zeros_like(x))
            else:
                # Compute difference; zero-pad the first k frames.
                diff = x[:, k:, :] - x[:, :-k, :]  # (B, T-k, D)
                pad = torch.zeros(B, k, D, device=x.device, dtype=x.dtype)
                deltas.append(torch.cat([pad, diff], dim=1))  # (B, T, D)

        # Concatenate along feature dim → (B, T, 4*D), then project → (B, T, D).
        out = torch.cat(deltas, dim=-1)
        out = self.proj(out)
        return out


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_seq_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)  # (max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer so it moves with the model but is not a parameter.
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to *x*.

        Args:
            x: (B, T, D).

        Returns:
            (B, T, D) with positional encoding added.
        """
        return x + self.pe[:, : x.size(1), :]


# ---------------------------------------------------------------------------
# Bacterium temporal encoder
# ---------------------------------------------------------------------------

class BacteriumTemporalEncoder(nn.Module):
    """Transformer-based temporal encoder for a single bacterium track.

    Pipeline
    --------
    1. Project raw features (384 → 256) and delta features (384 → 256).
    2. Sum the two projections element-wise (keeps dim 256).
    3. Add sinusoidal positional encoding.
    4. Pass through a 4-layer Transformer encoder.
    5. Mean-pool over valid (non-masked) timesteps → (B, 256).
    """

    def __init__(
        self,
        feat_dim: int = 384,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Feature projections.
        self.feat_proj = nn.Linear(feat_dim, hidden_dim)
        self.delta_proj = nn.Linear(feat_dim, hidden_dim)

        # Positional encoding.
        self.pos_enc = SinusoidalPositionalEncoding(hidden_dim, max_seq_len)

        # Transformer encoder.
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
        features: torch.Tensor,
        delta_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features:       (B, T, 384) backbone feature sequence.
            delta_features: (B, T, 384) multi-scale delta features.
            mask:           (B, T) boolean mask – ``True`` for valid timesteps,
                            ``False`` for padding.

        Returns:
            (B, hidden_dim) mean-pooled temporal representation.
        """
        # Project and sum.
        h = self.feat_proj(features) + self.delta_proj(delta_features)  # (B, T, 256)

        # Add positional encoding.
        h = self.pos_enc(h)

        # Build the key-padding mask expected by nn.TransformerEncoder:
        # True means *ignore* that position (opposite of our convention).
        if mask is not None:
            src_key_padding_mask = ~mask  # (B, T)
        else:
            src_key_padding_mask = None

        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)  # (B, T, 256)

        # Mean-pool over valid timesteps.
        if mask is not None:
            # Expand mask for broadcasting: (B, T, 1).
            mask_expanded = mask.unsqueeze(-1).float()
            lengths = mask_expanded.sum(dim=1).clamp(min=1.0)  # (B, 1)
            pooled = (h * mask_expanded).sum(dim=1) / lengths   # (B, 256)
        else:
            pooled = h.mean(dim=1)  # (B, 256)

        return pooled
