"""Early exit mechanisms for efficient antimicrobial susceptibility prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS

from .classifier import TemporalMILClassifier


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EarlyExitResult:
    """Container for a single early-exit prediction."""

    prediction: int
    """Predicted class index (0 or 1)."""

    confidence: float
    """Softmax confidence of the predicted class at the exit point."""

    exit_time_sec: float
    """Wall-clock experiment time (in seconds) at which the prediction was made."""

    prediction_history: List[Dict[str, Any]] = field(default_factory=list)
    """Per-evaluation-step records: each dict contains at least
    ``time_sec``, ``prediction``, ``confidence``."""


# ---------------------------------------------------------------------------
# Rule-based early exit policy
# ---------------------------------------------------------------------------

class EarlyExitPolicy:
    """Rule-based early exit policy using confidence and patience.

    At every *eval_interval* seconds the classifier is queried.  If the model
    produces a prediction whose softmax confidence exceeds
    *confidence_threshold* for *patience* consecutive evaluations (and we are
    past *min_time*), the policy halts and returns that prediction.

    If *max_time* is reached without a confident halt, the last prediction is
    returned.
    """

    def __init__(
        self,
        model: TemporalMILClassifier,
        eval_interval: float = 30.0,
        patience: int = 3,
        confidence_threshold: float = 0.85,
        min_time: float = 60.0,
        max_time: float = 3600.0,
        temperature_scaler: Optional["TemperatureScaler"] = None,
    ) -> None:
        self.model = model
        self.eval_interval = eval_interval
        self.patience = patience
        self.confidence_threshold = confidence_threshold
        self.min_time = min_time
        self.max_time = max_time
        self.temperature_scaler = temperature_scaler

    @torch.no_grad()
    def predict_with_early_exit(
        self,
        experiment_features_dict: Dict[str, Any],
    ) -> EarlyExitResult:
        """Run the early-exit evaluation loop over an experiment.

        ``experiment_features_dict`` must contain:

        * ``track_features``:  (1, N, T_max, 384) – full experiment features.
        * ``track_mask``:      (1, N) – valid-track mask.
        * ``seq_mask``:        (1, N, T_max) – per-frame validity mask.
        * ``timestamps_sec``:  1-D array/list of timestamps (in seconds)
          corresponding to each frame index along the T dimension.

        The method evaluates the classifier at every ``eval_interval`` seconds
        from ``timestamps_sec[0]`` to ``max_time`` (or the last timestamp),
        progressively revealing more frames.

        Returns:
            An :class:`EarlyExitResult` with the final prediction.
        """
        self.model.eval()

        track_features: torch.Tensor = experiment_features_dict["track_features"]
        track_mask: torch.Tensor = experiment_features_dict["track_mask"]
        seq_mask_full: torch.Tensor = experiment_features_dict["seq_mask"]
        timestamps_sec: list[float] = list(experiment_features_dict["timestamps_sec"])

        total_experiment_time = min(timestamps_sec[-1], self.max_time)
        device = track_features.device

        prediction_history: List[Dict[str, Any]] = []
        consecutive_confident = 0
        last_confident_pred: Optional[int] = None

        # Evaluation times.
        eval_time = self.eval_interval
        while eval_time <= total_experiment_time:
            # Determine how many frames are available up to eval_time.
            num_frames = 0
            for ts in timestamps_sec:
                if ts <= eval_time:
                    num_frames += 1
                else:
                    break

            if num_frames == 0:
                eval_time += self.eval_interval
                continue

            # Slice features and masks up to the current time.
            feats_slice = track_features[:, :, :num_frames, :]  # (1, N, t, 384)
            seq_mask_slice = seq_mask_full[:, :, :num_frames]   # (1, N, t)
            time_frac = torch.tensor(
                [eval_time / self.max_time], device=device, dtype=track_features.dtype
            )

            batch_dict = {
                "track_features": feats_slice,
                "track_mask": track_mask,
                "seq_mask": seq_mask_slice,
                "time_fraction": time_frac,
            }

            output = self.model(batch_dict)
            logits = output["logits"]  # (1, 2)

            # Optionally calibrate.
            if self.temperature_scaler is not None:
                logits = self.temperature_scaler(logits)

            probs = F.softmax(logits, dim=-1)  # (1, 2)
            confidence, pred = probs.max(dim=-1)
            confidence_val = confidence.item()
            pred_val = int(pred.item())

            step_record = {
                "time_sec": eval_time,
                "prediction": pred_val,
                "confidence": confidence_val,
                "num_frames": num_frames,
            }
            prediction_history.append(step_record)

            # Check halting criterion.
            if confidence_val >= self.confidence_threshold:
                if last_confident_pred == pred_val:
                    consecutive_confident += 1
                else:
                    consecutive_confident = 1
                    last_confident_pred = pred_val

                if (
                    consecutive_confident >= self.patience
                    and eval_time >= self.min_time
                ):
                    return EarlyExitResult(
                        prediction=pred_val,
                        confidence=confidence_val,
                        exit_time_sec=eval_time,
                        prediction_history=prediction_history,
                    )
            else:
                consecutive_confident = 0
                last_confident_pred = None

            eval_time += self.eval_interval

        # Reached max_time without confident halt – return last prediction.
        if prediction_history:
            last = prediction_history[-1]
            return EarlyExitResult(
                prediction=last["prediction"],
                confidence=last["confidence"],
                exit_time_sec=last["time_sec"],
                prediction_history=prediction_history,
            )

        # Edge case: no evaluations were performed (e.g. experiment too short).
        return EarlyExitResult(
            prediction=-1,
            confidence=0.0,
            exit_time_sec=0.0,
            prediction_history=[],
        )


# ---------------------------------------------------------------------------
# Temperature scaling (post-hoc calibration)
# ---------------------------------------------------------------------------

class TemperatureScaler(nn.Module):
    """Platt-style temperature scaling for post-hoc calibration.

    A single learnable temperature parameter is optimised on a held-out
    validation set to minimise negative log-likelihood.
    """

    def __init__(self, init_temperature: float = 1.5) -> None:
        super().__init__()
        self.temperature = nn.Parameter(
            torch.tensor(init_temperature, dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by the learned temperature.

        Args:
            logits: (*, C) raw logits from the classifier.

        Returns:
            (*, C) temperature-scaled logits.
        """
        return logits / self.temperature

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """Optimise temperature to minimise NLL on validation data.

        Args:
            val_logits: (N, C) logits predicted by the classifier on the
                validation set (detached, on the same device as this module).
            val_labels: (N,) ground-truth class indices.
            lr: Learning rate for L-BFGS.
            max_iter: Maximum number of L-BFGS iterations.

        Returns:
            The final NLL loss value.
        """
        val_logits = val_logits.detach()
        val_labels = val_labels.detach().long()

        # Reset temperature before fitting.
        self.temperature.data.fill_(1.5)

        optimizer = LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        final_loss = torch.tensor(0.0)

        def closure() -> torch.Tensor:
            nonlocal final_loss
            optimizer.zero_grad()
            scaled = self.forward(val_logits)
            loss = F.cross_entropy(scaled, val_labels)
            loss.backward()
            final_loss = loss.detach()
            return loss

        optimizer.step(closure)
        return final_loss.item()


# ---------------------------------------------------------------------------
# Learned halting policy (LSTM-based)
# ---------------------------------------------------------------------------

class LearnedHaltingPolicy(nn.Module):
    """LSTM-based learned halting policy.

    At each evaluation timestep the network receives a feature vector

        [logit_0, logit_1, confidence, time_frac, delta_conf]

    and produces a halt probability via an LSTM followed by a sigmoid head.
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.halt_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features_seq: torch.Tensor,
        hx: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            features_seq: (B, T_eval, 5) sequence of per-timestep features:
                ``[logit_0, logit_1, confidence, time_frac, delta_conf]``.
            hx: optional initial LSTM hidden state ``(h_0, c_0)``.

        Returns:
            halt_probs: (B, T_eval) halt probability at each evaluation step.
            hx:         final LSTM hidden state (for incremental inference).
        """
        lstm_out, hx = self.lstm(features_seq, hx)  # (B, T_eval, hidden_dim)
        halt_probs = self.halt_head(lstm_out).squeeze(-1)  # (B, T_eval)
        return halt_probs, hx

    @staticmethod
    def build_features(
        logits_seq: torch.Tensor,
        time_fracs: torch.Tensor,
    ) -> torch.Tensor:
        """Build the 5-D input features from raw classifier outputs.

        Args:
            logits_seq: (B, T_eval, 2) classifier logits at each eval step.
            time_fracs: (B, T_eval) normalised time fraction at each step.

        Returns:
            (B, T_eval, 5) feature tensor.
        """
        probs = F.softmax(logits_seq, dim=-1)            # (B, T, 2)
        confidence, _ = probs.max(dim=-1)                 # (B, T)

        # Delta confidence: conf[t] - conf[t-1], zero-padded at t=0.
        delta_conf = torch.zeros_like(confidence)
        delta_conf[:, 1:] = confidence[:, 1:] - confidence[:, :-1]

        # Stack: [logit_0, logit_1, confidence, time_frac, delta_conf]
        features = torch.stack(
            [
                logits_seq[:, :, 0],
                logits_seq[:, :, 1],
                confidence,
                time_fracs,
                delta_conf,
            ],
            dim=-1,
        )  # (B, T, 5)
        return features
