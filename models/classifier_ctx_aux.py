"""Classifier variant with time-weighted auxiliary loss on contextualized bins.

Key differences from classifier.py:
1. Aux classifier runs on POST-temporal-encoder embeddings (contextualized),
   so each bin's R/S prediction is informed by the full temporal sequence.
2. Returns bin_times in output so training can apply time-weighted aux loss
   (low weight for early bins where R/S look identical, high for late bins).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .temporal_encoder import PopulationBinEncoder, PopulationTemporalEncoder
from .mil_aggregator import GatedAttentionMIL


class ClassifierHead(nn.Module):
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


class ContextualAuxClassifier(nn.Module):
    """Classifier with aux predictions on contextualized (post-transformer) bins.

    Architecture is identical to PopulationTemporalClassifier except:
    - Aux classifier runs AFTER temporal encoder, not before
    - Returns bin_times for time-weighted aux loss computation
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
    ) -> None:
        super().__init__()
        self.use_delta_features = use_delta_features

        # Per-bin population encoding (stats only — attention encoder failed)
        self.bin_encoder = PopulationBinEncoder(
            feat_dim=feature_dim,
            hidden_dim=temporal_hidden_dim,
            max_count_normalizer=max_count_normalizer,
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

        # Classification head: bin_attention_repr + time_fraction
        self.classifier = ClassifierHead(
            in_dim=temporal_hidden_dim + 1,
            num_classes=num_classes,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )

        # Aux classifier on CONTEXTUALIZED bin embeddings + normalized time
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
        B, T, N, D = bin_features.shape
        flat_features = bin_features.reshape(B * T, N, D)
        flat_crop_mask = crop_mask.reshape(B * T, N)
        flat_counts = bin_counts.reshape(B * T)

        if self.use_delta_features:
            flat_stats = self.bin_encoder.compute_stats(
                flat_features, flat_crop_mask, flat_counts
            )
            stats = flat_stats.reshape(B, T, -1)

            if bin_mask is not None:
                first_idx = bin_mask.float().argmax(dim=1)
                baseline = stats[
                    torch.arange(B, device=stats.device), first_idx
                ]
            else:
                baseline = stats[:, 0]

            stats = stats - baseline.unsqueeze(1)
            flat_embeddings = self.bin_encoder.proj(stats.reshape(B * T, -1))
        else:
            flat_embeddings = self.bin_encoder(
                flat_features, flat_crop_mask, flat_counts
            )

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

        # 2. Temporal encoding across bins
        time_features = self.temporal_encoder.time_enc(bin_times)
        h = bin_embeddings + self.temporal_encoder.time_proj(time_features)
        src_key_padding_mask = ~bin_mask if bin_mask is not None else None
        h_contextualized = self.temporal_encoder.transformer(
            h, src_key_padding_mask=src_key_padding_mask
        )  # (B, T, hidden_dim)

        # 3. Aux predictions on CONTEXTUALIZED embeddings (temporal context)
        norm_times = (bin_times / 3600.0).unsqueeze(-1)  # (B, T, 1)
        aux_input = torch.cat([h_contextualized, norm_times], dim=-1)
        bin_logits = self.aux_classifier(aux_input)  # (B, T, 2)

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
            "bin_times": bin_times,  # for time-weighted aux loss
        }
