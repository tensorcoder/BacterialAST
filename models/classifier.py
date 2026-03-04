"""Full Temporal MIL Classifier for antimicrobial susceptibility testing."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .temporal_encoder import BacteriumTemporalEncoder, DeltaFeatureComputer
from .mil_aggregator import GatedAttentionMIL, PopulationFeatureExtractor


class ClassifierHead(nn.Module):
    """Two-hidden-layer classification head.

    Architecture::

        Linear(in_dim, 128) → LayerNorm → GELU → Dropout(0.1)
        → Linear(128, 128) → LayerNorm → GELU → Dropout(0.1)
        → Linear(128, num_classes)
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class TemporalMILClassifier(nn.Module):
    """End-to-end Temporal Multi-Instance Learning classifier.

    Expected input
    --------------
    A ``batch_dict`` with the following keys:

    * ``track_features``: (B, N, T, 384) – backbone features per track.
    * ``track_mask``:     (B, N)          – ``True`` for valid tracks.
    * ``seq_mask``:       (B, N, T)       – ``True`` for valid timesteps.
    * ``time_fraction``:  (B,)            – normalised elapsed time in [0, 1].

    Outputs
    -------
    A dict with:

    * ``logits``:            (B, 2) – classification logits.
    * ``attention_weights``: (B, N) – per-track attention weights.
    """

    def __init__(
        self,
        feature_dim: int = 384,
        temporal_hidden_dim: int = 256,
        temporal_num_layers: int = 4,
        temporal_num_heads: int = 4,
        temporal_ffn_dim: int = 512,
        mil_hidden_dim: int = 128,
        population_feat_dim: int = 64,
        classifier_hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1,
        delta_scales: list[int] | None = None,
        micro_batch_size: int = 256,
        # Legacy aliases
        feat_dim: int | None = None,
        pop_output_dim: int | None = None,
        temporal_dropout: float | None = None,
    ) -> None:
        super().__init__()
        # Support legacy parameter names
        feature_dim = feat_dim or feature_dim
        population_feat_dim = pop_output_dim or population_feat_dim
        dropout = temporal_dropout or dropout

        self.micro_batch_size = micro_batch_size

        # Per-track temporal modelling.
        self.delta_computer = DeltaFeatureComputer(feat_dim=feature_dim)
        self.temporal_encoder = BacteriumTemporalEncoder(
            feat_dim=feature_dim,
            hidden_dim=temporal_hidden_dim,
            num_layers=temporal_num_layers,
            num_heads=temporal_num_heads,
            ffn_dim=temporal_ffn_dim,
            dropout=dropout,
        )

        # Multi-instance learning aggregation.
        self.mil_aggregator = GatedAttentionMIL(
            input_dim=temporal_hidden_dim,
            hidden_dim=mil_hidden_dim,
        )
        self.pop_features = PopulationFeatureExtractor(
            input_dim=temporal_hidden_dim,
            output_dim=population_feat_dim,
        )

        # Classification head.
        # Input: bag_repr (256) + pop_features (64) + time_fraction (1) = 321
        self.classifier = ClassifierHead(
            in_dim=temporal_hidden_dim + population_feat_dim + 1,
            num_classes=num_classes,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    # Temporal encoding with micro-batching
    # ------------------------------------------------------------------

    def _encode_tracks(
        self,
        track_features: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode all tracks through the temporal encoder with micro-batching.

        Args:
            track_features: (B, N, T, D) backbone features.
            seq_mask:       (B, N, T) boolean mask for valid timesteps.

        Returns:
            (B, N, hidden_dim) per-track temporal representations.
        """
        B, N, T, D = track_features.shape
        hidden_dim = self.temporal_encoder.hidden_dim

        # Flatten batch and track dims → (B*N, T, D).
        flat_feats = track_features.reshape(B * N, T, D)
        flat_mask = seq_mask.reshape(B * N, T)

        # Compute deltas.
        flat_deltas = self.delta_computer(flat_feats)  # (B*N, T, D)

        # Micro-batch through temporal encoder to manage GPU memory.
        total = B * N
        outputs: list[torch.Tensor] = []
        for start in range(0, total, self.micro_batch_size):
            end = min(start + self.micro_batch_size, total)
            chunk_feats = flat_feats[start:end]
            chunk_deltas = flat_deltas[start:end]
            chunk_mask = flat_mask[start:end]
            encoded = self.temporal_encoder(
                chunk_feats, chunk_deltas, mask=chunk_mask
            )  # (chunk, hidden_dim)
            outputs.append(encoded)

        # Reassemble → (B, N, hidden_dim).
        flat_encoded = torch.cat(outputs, dim=0)  # (B*N, hidden_dim)
        return flat_encoded.reshape(B, N, hidden_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch_dict: dictionary with keys ``track_features``, ``track_mask``,
                ``seq_mask``, and ``time_fraction``.

        Returns:
            Dict with ``logits`` (B, 2) and ``attention_weights`` (B, N).
        """
        track_features: torch.Tensor = batch_dict["track_features"]  # (B, N, T, 384)
        track_mask: torch.Tensor = batch_dict["track_mask"]          # (B, N)
        seq_mask: torch.Tensor = batch_dict["seq_mask"]              # (B, N, T)
        time_fraction: torch.Tensor = batch_dict["time_fraction"]    # (B,)

        # 1. Temporal encoding per track.
        track_repr = self._encode_tracks(track_features, seq_mask)  # (B, N, 256)

        # 2. MIL aggregation.
        bag_repr, attn_weights = self.mil_aggregator(
            track_repr, mask=track_mask
        )  # (B, 256), (B, N)

        # 3. Population statistics.
        pop_feat = self.pop_features(track_repr, mask=track_mask)  # (B, 64)

        # 4. Concatenate bag repr, pop features, and time fraction.
        combined = torch.cat(
            [bag_repr, pop_feat, time_fraction.unsqueeze(-1)],
            dim=-1,
        )  # (B, 256 + 64 + 1) = (B, 321)

        # 5. Classify.
        logits = self.classifier(combined)  # (B, 2)

        return {
            "logits": logits,
            "attention_weights": attn_weights,
        }
