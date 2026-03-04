"""Evaluation metrics for antibiotic susceptibility testing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)

from ..models.early_exit import EarlyExitResult


@dataclass
class ClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    auroc: float
    auprc: float
    sensitivity: float
    specificity: float
    f1: float
    mcc: float


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> ClassificationMetrics:
    """Compute full suite of classification metrics."""
    try:
        auroc = roc_auc_score(labels, probabilities)
    except ValueError:
        auroc = 0.5

    try:
        auprc = average_precision_score(labels, probabilities)
    except ValueError:
        auprc = 0.5

    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    return ClassificationMetrics(
        accuracy=accuracy_score(labels, predictions),
        balanced_accuracy=balanced_accuracy_score(labels, predictions),
        auroc=auroc,
        auprc=auprc,
        sensitivity=sensitivity,
        specificity=specificity,
        f1=f1_score(labels, predictions, zero_division=0),
        mcc=matthews_corrcoef(labels, predictions),
    )


@dataclass
class TimeToPredictionMetrics:
    mean_exit_time_sec: float
    median_exit_time_sec: float
    std_exit_time_sec: float
    time_for_90_acc: float | None
    time_for_95_acc: float | None
    time_for_99_acc: float | None
    accuracy_at_5min: float | None
    accuracy_at_10min: float | None
    accuracy_at_15min: float | None
    accuracy_at_30min: float | None


def time_to_prediction_analysis(
    exit_results: list[EarlyExitResult],
    accuracy_vs_time: dict[float, float] | None = None,
) -> TimeToPredictionMetrics:
    """Analyze time-to-prediction characteristics."""
    exit_times = np.array([r.exit_time_sec for r in exit_results])

    # Find time to reach accuracy thresholds
    def find_time_for_accuracy(target: float) -> float | None:
        if accuracy_vs_time is None:
            return None
        for t in sorted(accuracy_vs_time.keys()):
            if accuracy_vs_time[t] >= target:
                return t
        return None

    # Accuracy at fixed time points
    def accuracy_at_time(target_time: float) -> float | None:
        if accuracy_vs_time is None:
            return None
        # Find closest time point
        times = sorted(accuracy_vs_time.keys())
        closest = min(times, key=lambda t: abs(t - target_time))
        if abs(closest - target_time) > 60:  # within 1 minute
            return None
        return accuracy_vs_time[closest]

    return TimeToPredictionMetrics(
        mean_exit_time_sec=float(np.mean(exit_times)),
        median_exit_time_sec=float(np.median(exit_times)),
        std_exit_time_sec=float(np.std(exit_times)),
        time_for_90_acc=find_time_for_accuracy(0.90),
        time_for_95_acc=find_time_for_accuracy(0.95),
        time_for_99_acc=find_time_for_accuracy(0.99),
        accuracy_at_5min=accuracy_at_time(300),
        accuracy_at_10min=accuracy_at_time(600),
        accuracy_at_15min=accuracy_at_time(900),
        accuracy_at_30min=accuracy_at_time(1800),
    )


def per_antibiotic_analysis(
    exit_results: list[EarlyExitResult],
    experiment_ids: list[str],
    labels: np.ndarray,
) -> dict[str, dict]:
    """Breakdown of accuracy and exit time by antibiotic type.

    Assumes experiment_id format: BacteriaID_AntibioticID_Dosage_ExpID
    """
    # Parse antibiotic IDs
    abx_map: dict[str, list[int]] = {}
    for i, exp_id in enumerate(experiment_ids):
        parts = exp_id.split("_")
        abx_id = parts[1] if len(parts) > 1 else "unknown"
        abx_map.setdefault(abx_id, []).append(i)

    results = {}
    for abx_id, indices in abx_map.items():
        sub_results = [exit_results[i] for i in indices]
        sub_labels = labels[indices]
        sub_preds = np.array([r.prediction for r in sub_results])
        sub_times = np.array([r.exit_time_sec for r in sub_results])

        acc = (sub_preds == sub_labels).mean()
        results[abx_id] = {
            "n_experiments": len(indices),
            "accuracy": float(acc),
            "mean_exit_time": float(np.mean(sub_times)),
            "median_exit_time": float(np.median(sub_times)),
        }

    return results
