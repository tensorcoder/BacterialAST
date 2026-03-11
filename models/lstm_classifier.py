"""BiLSTM-based temporal classifier with per-timestep predictions.

Instead of a Transformer over bin embeddings, uses a BiLSTM that naturally
encodes temporal dynamics. Classifies at every timestep, giving dense
supervision and natural early-exit support.

Architecture:
    1. PopulationBinEncoder: per-bin population stats → bin embedding
    2. BiLSTM: processes bin embeddings in temporal order
    3. Per-timestep classifier: predicts R/S at each time bin
    4. Experiment-level prediction from last valid timestep
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .temporal_encoder import PopulationBinEncoder


class LSTMTemporalClassifier(nn.Module):
    """BiLSTM temporal classifier with per-timestep R/S predictions.

    Expected input: same ``batch_dict`` as PopulationTemporalClassifier.

    Outputs:
        * ``logits``:      (B, 2) — experiment-level prediction (last valid bin).
        * ``step_logits``: (B, T, 2) — per-timestep predictions.
    """

    def __init__(
        self,
        feature_dim: int = 384,
        bin_hidden_dim: int = 128,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        classifier_hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.2,
        max_count_normalizer: float = 256.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Per-bin population encoding (same stats encoder)
        self.bin_encoder = PopulationBinEncoder(
            feat_dim=feature_dim,
            hidden_dim=bin_hidden_dim,
            max_count_normalizer=max_count_normalizer,
        )

        # BiLSTM over temporal sequence of bin embeddings
        self.lstm = nn.LSTM(
            input_size=bin_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )

        # Per-timestep classifier: BiLSTM output (2*H) + normalized time (1)
        self.step_classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden_dim + 1, classifier_hidden_dim),
            nn.LayerNorm(classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def _encode_bins(
        self,
        bin_features: torch.Tensor,
        crop_mask: torch.Tensor,
        bin_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Encode each time bin via population statistics.

        Args:
            bin_features: (B, T, N, D) crop features per bin.
            crop_mask:    (B, T, N) boolean mask for valid crops.
            bin_counts:   (B, T) number of crops per bin.

        Returns:
            (B, T, bin_hidden_dim) per-bin embeddings.
        """
        B, T, N, D = bin_features.shape
        flat_features = bin_features.reshape(B * T, N, D)
        flat_crop_mask = crop_mask.reshape(B * T, N)
        flat_counts = bin_counts.reshape(B * T)
        flat_embeddings = self.bin_encoder(flat_features, flat_crop_mask, flat_counts)
        return flat_embeddings.reshape(B, T, -1)

    def forward(self, batch_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        bin_features = batch_dict["bin_features"]   # (B, T, N, 384)
        bin_mask = batch_dict["bin_mask"]            # (B, T)
        crop_mask = batch_dict["crop_mask"]          # (B, T, N)
        bin_times = batch_dict["bin_times"]          # (B, T)
        bin_counts = batch_dict["bin_counts"]        # (B, T)

        B, T = bin_mask.shape
        device = bin_mask.device

        # 1. Encode each bin's population
        bin_embeddings = self._encode_bins(bin_features, crop_mask, bin_counts)

        # 2. Run BiLSTM (pack to handle variable-length sequences)
        lengths = bin_mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            bin_embeddings, lengths, batch_first=True, enforce_sorted=False,
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=T,
        )  # (B, T, 2 * lstm_hidden_dim)

        # 3. Per-timestep classification with time conditioning
        norm_times = (bin_times / 3600.0).unsqueeze(-1)  # (B, T, 1)
        step_input = torch.cat([lstm_out, norm_times], dim=-1)
        step_logits = self.step_classifier(step_input)  # (B, T, num_classes)

        # 4. Experiment-level prediction = last valid timestep
        last_idx = (lengths - 1).long().to(device)  # (B,)
        final_logits = step_logits[torch.arange(B, device=device), last_idx]

        return {
            "logits": final_logits,       # (B, 2)
            "step_logits": step_logits,    # (B, T, 2)
        }
