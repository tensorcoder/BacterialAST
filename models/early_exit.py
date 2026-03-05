"""Early exit mechanisms for efficient antimicrobial susceptibility prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS


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
    """Per-evaluation-step records."""


# ---------------------------------------------------------------------------
# Rule-based early exit policy
# ---------------------------------------------------------------------------

class EarlyExitPolicy:
    """Rule-based early exit policy using confidence and patience.

    At every *eval_interval* seconds the classifier is queried.  If the model
    produces a prediction whose softmax confidence exceeds
    *confidence_threshold* for *patience* consecutive evaluations (and we are
    past *min_time*), the policy halts and returns that prediction.
    """

    def __init__(
        self,
        model: nn.Module,
        eval_interval: float = 30.0,
        patience: int = 3,
        confidence_threshold: float = 0.85,
        min_time: float = 60.0,
        max_time: float = 3600.0,
        temperature_scaler: Optional[TemperatureScaler] = None,
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
        experiment_data: Dict[str, Any],
    ) -> EarlyExitResult:
        """Run the early-exit evaluation loop over an experiment.

        ``experiment_data`` must contain all keys expected by the
        PopulationTemporalClassifier's forward method, plus
        ``max_experiment_time_sec`` indicating the total duration.

        The method progressively increases the time window and evaluates
        the classifier at each ``eval_interval``.
        """
        self.model.eval()

        max_experiment_time = min(
            experiment_data.get("max_experiment_time_sec", self.max_time),
            self.max_time,
        )

        prediction_history: List[Dict[str, Any]] = []
        consecutive_confident = 0
        last_confident_pred: Optional[int] = None

        eval_time = self.eval_interval
        while eval_time <= max_experiment_time:
            # The caller is responsible for constructing the batch_dict
            # for the current eval_time. This policy just manages the
            # halting logic.
            batch_dict = experiment_data.get("batch_dict_fn", lambda t: {})(eval_time)
            if not batch_dict:
                eval_time += self.eval_interval
                continue

            output = self.model(batch_dict)
            logits = output["logits"]

            if self.temperature_scaler is not None:
                logits = self.temperature_scaler(logits)

            probs = F.softmax(logits, dim=-1)
            confidence, pred = probs.max(dim=-1)
            confidence_val = confidence.item()
            pred_val = int(pred.item())

            step_record = {
                "time_sec": eval_time,
                "prediction": pred_val,
                "confidence": confidence_val,
            }
            prediction_history.append(step_record)

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

        if prediction_history:
            last = prediction_history[-1]
            return EarlyExitResult(
                prediction=last["prediction"],
                confidence=last["confidence"],
                exit_time_sec=last["time_sec"],
                prediction_history=prediction_history,
            )

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
    """Platt-style temperature scaling for post-hoc calibration."""

    def __init__(self, init_temperature: float = 1.5) -> None:
        super().__init__()
        self.temperature = nn.Parameter(
            torch.tensor(init_temperature, dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        val_logits = val_logits.detach()
        val_labels = val_labels.detach().long()
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
    """LSTM-based learned halting policy."""

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
        lstm_out, hx = self.lstm(features_seq, hx)
        halt_probs = self.halt_head(lstm_out).squeeze(-1)
        return halt_probs, hx

    @staticmethod
    def build_features(
        logits_seq: torch.Tensor,
        time_fracs: torch.Tensor,
    ) -> torch.Tensor:
        probs = F.softmax(logits_seq, dim=-1)
        confidence, _ = probs.max(dim=-1)
        delta_conf = torch.zeros_like(confidence)
        delta_conf[:, 1:] = confidence[:, 1:] - confidence[:, :-1]
        features = torch.stack(
            [
                logits_seq[:, :, 0],
                logits_seq[:, :, 1],
                confidence,
                time_fracs,
                delta_conf,
            ],
            dim=-1,
        )
        return features
