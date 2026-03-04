"""Multi-instance learning aggregation: gated attention and population features."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gated Attention MIL (Ilse et al., 2018)
# ---------------------------------------------------------------------------

class GatedAttentionMIL(nn.Module):
    """Gated attention-based multi-instance learning pooling.

    For each instance representation *h_i*::

        V_i = tanh(W_1 @ h_i)
        U_i = sigmoid(W_2 @ h_i)
        a_i = W_3 @ (V_i * U_i)          # attention logit

    After masking invalid instances (setting their logits to ``-inf``), the
    attention weights are computed via softmax and used to form a
    weighted-sum bag representation.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.W_1 = nn.Linear(input_dim, hidden_dim)
        self.W_2 = nn.Linear(input_dim, hidden_dim)
        self.W_3 = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h:    (B, N, input_dim) instance representations.
            mask: (B, N) boolean mask – ``True`` for valid instances,
                  ``False`` for padding.

        Returns:
            bag_repr:          (B, input_dim) attention-weighted bag
                               representation.
            attention_weights: (B, N) normalised attention weights.
        """
        V = torch.tanh(self.W_1(h))       # (B, N, hidden_dim)
        U = torch.sigmoid(self.W_2(h))    # (B, N, hidden_dim)
        logits = self.W_3(V * U).squeeze(-1)  # (B, N)

        # Mask invalid instances before softmax.
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))

        attention_weights = F.softmax(logits, dim=-1)  # (B, N)

        # Handle edge case where all instances are masked (all -inf → NaN).
        attention_weights = attention_weights.nan_to_num(0.0)

        # Weighted sum over instances.
        bag_repr = torch.bmm(
            attention_weights.unsqueeze(1), h
        ).squeeze(1)  # (B, input_dim)

        return bag_repr, attention_weights


# ---------------------------------------------------------------------------
# Population feature extractor
# ---------------------------------------------------------------------------

class PopulationFeatureExtractor(nn.Module):
    """Compute masked population-level statistics over instance features.

    Statistics: mean, std, skewness, kurtosis – each (B, D).
    Concatenated to (B, 4*D) then projected through an MLP to (B, 64).
    """

    def __init__(
        self,
        input_dim: int = 256,
        output_dim: int = 64,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 4, 256),
            nn.GELU(),
            nn.Linear(256, output_dim),
        )

    @staticmethod
    def _masked_stats(
        h: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute masked mean, std, skewness, and kurtosis.

        Args:
            h:    (B, N, D) instance features.
            mask: (B, N) boolean mask or ``None``.

        Returns:
            mean, std, skewness, kurtosis – each (B, D).
        """
        if mask is not None:
            # (B, N, 1) float mask for broadcasting.
            m = mask.unsqueeze(-1).float()
            count = m.sum(dim=1).clamp(min=1.0)  # (B, 1)
        else:
            m = torch.ones(
                h.shape[0], h.shape[1], 1, device=h.device, dtype=h.dtype
            )
            count = torch.full(
                (h.shape[0], 1), h.shape[1], device=h.device, dtype=h.dtype
            )

        # Mean.
        mean = (h * m).sum(dim=1) / count  # (B, D)

        # Centred values.
        diff = (h - mean.unsqueeze(1)) * m  # (B, N, D)

        # Variance / std.
        var = (diff ** 2).sum(dim=1) / count.clamp(min=1.0)  # (B, D)
        std = (var + 1e-8).sqrt()

        # Skewness: E[(x - mu)^3] / sigma^3.
        m3 = (diff ** 3).sum(dim=1) / count.clamp(min=1.0)
        skewness = m3 / (std ** 3 + 1e-8)

        # Kurtosis (excess): E[(x - mu)^4] / sigma^4 - 3.
        m4 = (diff ** 4).sum(dim=1) / count.clamp(min=1.0)
        kurtosis = m4 / (std ** 4 + 1e-8) - 3.0

        return mean, std, skewness, kurtosis

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            h:    (B, N, input_dim) instance features.
            mask: (B, N) boolean mask – ``True`` for valid instances.

        Returns:
            (B, output_dim) population-level feature vector.
        """
        mean, std, skewness, kurtosis = self._masked_stats(h, mask)
        # Concatenate statistics → (B, 4 * input_dim).
        stats = torch.cat([mean, std, skewness, kurtosis], dim=-1)
        return self.mlp(stats)
