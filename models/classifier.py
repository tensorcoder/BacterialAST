"""Population Temporal Classifier for antimicrobial susceptibility testing."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .temporal_encoder import AttentionBinEncoder, PopulationBinEncoder, PopulationTemporalEncoder
from .mil_aggregator import GatedAttentionMIL


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


class PopulationTemporalClassifier(nn.Module):
    """End-to-end Population Temporal classifier.

    Replaces the per-track temporal MIL approach with population-level
    analysis. Instead of tracking individual bacteria, this model:

    1. Encodes each time bin's population via statistics (mean, std,
       skewness, kurtosis) of the crop features within that bin.
    2. Models the temporal evolution of these population snapshots
       via a Transformer encoder.
    3. Uses gated attention to pool over time bins.
    4. Classifies the experiment as resistant or susceptible.

    Expected input
    --------------
    A ``batch_dict`` with keys:

    * ``bin_features``:  (B, T, N, 384) — crop features per time bin.
    * ``bin_mask``:      (B, T)         — True for valid time bins.
    * ``crop_mask``:     (B, T, N)      — True for valid crops within bins.
    * ``bin_times``:     (B, T)         — bin center times in seconds.
    * ``bin_counts``:    (B, T)         — number of crops per bin.
    * ``time_fraction``: (B,)           — normalised elapsed time [0, 1].

    Outputs
    -------
    A dict with:

    * ``logits``:            (B, 2) — classification logits.
    * ``attention_weights``: (B, T) — per-bin attention weights.
    """

    def __init__(
        self,
        feature_dim: int = 384,
        temporal_hidden_dim: int = 256,
        temporal_num_layers: int = 4,
        temporal_num_heads: int = 4,
        temporal_ffn_dim: int = 512,
        classifier_hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1,
        max_count_normalizer: float = 256.0,
        use_delta_features: bool = False,
        bin_encoder_type: str = "stats",
        bin_attn_heads: int = 4,
        include_count: bool = True,
    ) -> None:
        super().__init__()
        self.use_delta_features = use_delta_features
        self.bin_encoder_type = bin_encoder_type

        # Per-bin population encoding
        if bin_encoder_type == "attention":
            self.bin_encoder = AttentionBinEncoder(
                feat_dim=feature_dim,
                hidden_dim=temporal_hidden_dim,
                num_heads=bin_attn_heads,
                max_count_normalizer=max_count_normalizer,
            )
        else:
            self.bin_encoder = PopulationBinEncoder(
                feat_dim=feature_dim,
                hidden_dim=temporal_hidden_dim,
                max_count_normalizer=max_count_normalizer,
                include_count=include_count,
            )

        # Temporal evolution across bins
        self.temporal_encoder = PopulationTemporalEncoder(
            hidden_dim=temporal_hidden_dim,
            num_layers=temporal_num_layers,
            num_heads=temporal_num_heads,
            ffn_dim=temporal_ffn_dim,
            dropout=dropout,
        )

        # Attention pooling over time bins
        self.bin_attention = GatedAttentionMIL(
            input_dim=temporal_hidden_dim,
            hidden_dim=temporal_hidden_dim // 2,
        )

        # Classification head
        # Input: bin_attention_repr (256) + time_fraction (1) = 257
        self.classifier = ClassifierHead(
            in_dim=temporal_hidden_dim + 1,
            num_classes=num_classes,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )

        # Time-conditioned auxiliary bin classifier
        # Predicts R/S per bin given (bin_embedding, normalized_time)
        self.aux_classifier = nn.Sequential(
            nn.Linear(temporal_hidden_dim + 1, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )

    def _encode_bins(
        self,
        bin_features: torch.Tensor,
        crop_mask: torch.Tensor,
        bin_counts: torch.Tensor,
        bin_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode each time bin into a population embedding.

        When ``use_delta_features`` is True and the stats encoder is used,
        raw population statistics are computed first, then the first valid
        bin's statistics are subtracted before projection.

        Args:
            bin_features: (B, T, N, D) crop features per bin.
            crop_mask:    (B, T, N) boolean mask for valid crops.
            bin_counts:   (B, T) number of crops per bin.
            bin_mask:     (B, T) boolean mask for valid bins.

        Returns:
            (B, T, hidden_dim) per-bin population embeddings.
        """
        B, T, N, D = bin_features.shape

        # Flatten batch and time dims
        flat_features = bin_features.reshape(B * T, N, D)
        flat_crop_mask = crop_mask.reshape(B * T, N)
        flat_counts = bin_counts.reshape(B * T)

        if (self.use_delta_features
                and isinstance(self.bin_encoder, PopulationBinEncoder)):
            # Stats encoder: delta in raw stats space before projection
            flat_stats = self.bin_encoder.compute_stats(
                flat_features, flat_crop_mask, flat_counts
            )  # (B*T, 4*D+1)
            stats = flat_stats.reshape(B, T, -1)  # (B, T, 4*D+1)

            if bin_mask is not None:
                first_idx = bin_mask.float().argmax(dim=1)  # (B,)
                baseline = stats[
                    torch.arange(B, device=stats.device), first_idx
                ]  # (B, 4*D+1)
            else:
                baseline = stats[:, 0]  # (B, 4*D+1)

            stats = stats - baseline.unsqueeze(1)  # (B, T, 4*D+1)
            flat_embeddings = self.bin_encoder.proj(stats.reshape(B * T, -1))
        else:
            flat_embeddings = self.bin_encoder(
                flat_features, flat_crop_mask, flat_counts
            )  # (B*T, hidden_dim)

        hidden_dim = flat_embeddings.shape[-1]
        return flat_embeddings.reshape(B, T, hidden_dim)

    def forward(self, batch_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        bin_features = batch_dict["bin_features"]    # (B, T, N, 384)
        bin_mask = batch_dict["bin_mask"]             # (B, T)
        crop_mask = batch_dict["crop_mask"]           # (B, T, N)
        bin_times = batch_dict["bin_times"]           # (B, T)
        bin_counts = batch_dict["bin_counts"]         # (B, T)
        time_fraction = batch_dict["time_fraction"]   # (B,)

        # 1. Encode each bin's population
        bin_embeddings = self._encode_bins(
            bin_features, crop_mask, bin_counts, bin_mask
        )  # (B, T, hidden_dim)

        # 2. Per-bin auxiliary predictions (time-conditioned)
        norm_times = (bin_times / 3600.0).unsqueeze(-1)  # (B, T, 1)
        aux_input = torch.cat([bin_embeddings, norm_times], dim=-1)
        bin_logits = self.aux_classifier(aux_input)  # (B, T, 2)

        # 3. Temporal encoding across bins
        time_features = self.temporal_encoder.time_enc(bin_times)
        h = bin_embeddings + self.temporal_encoder.time_proj(time_features)
        src_key_padding_mask = ~bin_mask if bin_mask is not None else None
        h_contextualized = self.temporal_encoder.transformer(
            h, src_key_padding_mask=src_key_padding_mask
        )  # (B, T, hidden_dim)

        # 4. Attention pooling over time bins
        experiment_repr, attn_weights = self.bin_attention(
            h_contextualized, mask=bin_mask
        )  # (B, hidden_dim), (B, T)

        # 5. Classify
        combined = torch.cat(
            [experiment_repr, time_fraction.unsqueeze(-1)],
            dim=-1,
        )  # (B, hidden_dim + 1)

        logits = self.classifier(combined)  # (B, 2)

        return {
            "logits": logits,
            "attention_weights": attn_weights,
            "bin_logits": bin_logits,
        }
