"""Stage 5: Early-exit calibration on validation set."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import FullConfig
from ..data.dataset import PopulationTemporalDataset, build_experiment_list, create_splits, population_temporal_collate
from ..models.classifier import PopulationTemporalClassifier
from ..models.early_exit import TemperatureScaler

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    accuracy_vs_time: dict[float, float]
    pareto_points: list[tuple[float, float]]  # [(accuracy, mean_exit_time), ...]
    optimal_configs: dict[float, dict]  # {target_acc: {patience, threshold, mean_time}}
    per_window_predictions: dict[float, dict]  # {time: {labels, probs, preds}}


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move batch tensors to device."""
    return {
        "bin_features": batch["bin_features"].to(device, non_blocking=True),
        "bin_mask": batch["bin_mask"].to(device, non_blocking=True),
        "crop_mask": batch["crop_mask"].to(device, non_blocking=True),
        "bin_times": batch["bin_times"].to(device, non_blocking=True),
        "bin_counts": batch["bin_counts"].to(device, non_blocking=True),
        "time_fraction": batch["time_fraction"].to(device, non_blocking=True),
    }


def evaluate_at_fixed_times(
    model: PopulationTemporalClassifier,
    experiments: list,
    feature_dir: Path,
    time_windows: list[float],
    time_bin_width_sec: float,
    max_crops_per_bin: int,
    feature_dim: int,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 4,
) -> dict[float, dict]:
    """Evaluate model at each fixed time window, returning predictions per window."""
    model.eval()
    results = {}

    for window_sec in tqdm(time_windows, desc="Evaluating time windows"):
        dataset = PopulationTemporalDataset(
            feature_dir=feature_dir,
            experiments=experiments,
            time_bin_width_sec=time_bin_width_sec,
            time_windows_sec=[window_sec],
            max_crops_per_bin=max_crops_per_bin,
            feature_dim=feature_dim,
            random_window=False,
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            collate_fn=population_temporal_collate,
        )

        all_probs = []
        all_labels = []
        all_logits = []

        with torch.no_grad():
            for batch in loader:
                batch_gpu = _batch_to_device(batch, device)
                with torch.amp.autocast("cuda"):
                    output = model(batch_gpu)

                logits = output["logits"].float()
                probs = F.softmax(logits, dim=-1)[:, 1]
                all_logits.append(logits.cpu())
                all_probs.append(probs.cpu().numpy())
                all_labels.append(batch["label"].numpy())

        results[window_sec] = {
            "probs": np.concatenate(all_probs),
            "labels": np.concatenate(all_labels),
            "logits": torch.cat(all_logits),
            "preds": (np.concatenate(all_probs) > 0.5).astype(int),
        }

    return results


def find_pareto_optimal(
    per_window: dict[float, dict],
    patience_range: list[int],
    threshold_range: list[float],
    time_windows: list[float],
) -> tuple[list[tuple[float, float]], dict[float, dict]]:
    """Sweep patience/threshold and find Pareto-optimal operating points."""
    n_experiments = len(list(per_window.values())[0]["labels"])

    configs = []
    for patience in patience_range:
        for threshold in threshold_range:
            exit_times = []
            correct = []

            for exp_idx in range(n_experiments):
                label = None
                consecutive = 0
                exited = False

                for t in sorted(time_windows):
                    data = per_window[t]
                    prob = data["probs"][exp_idx]
                    label = data["labels"][exp_idx]
                    pred = int(prob > 0.5)
                    conf = max(prob, 1 - prob)

                    if conf >= threshold:
                        consecutive += 1
                    else:
                        consecutive = 0

                    if consecutive >= patience:
                        exit_times.append(t)
                        correct.append(int(pred == label))
                        exited = True
                        break

                if not exited:
                    last_t = max(time_windows)
                    last_data = per_window[last_t]
                    pred = int(last_data["probs"][exp_idx] > 0.5)
                    exit_times.append(last_t)
                    correct.append(int(pred == label))

            acc = np.mean(correct)
            mean_time = np.mean(exit_times)
            median_time = np.median(exit_times)

            configs.append({
                "patience": patience,
                "threshold": threshold,
                "accuracy": acc,
                "mean_exit_time": mean_time,
                "median_exit_time": median_time,
            })

    # Find Pareto front
    pareto = []
    for c in configs:
        dominated = False
        for other in configs:
            if (other["accuracy"] >= c["accuracy"] and
                other["mean_exit_time"] <= c["mean_exit_time"] and
                (other["accuracy"] > c["accuracy"] or
                 other["mean_exit_time"] < c["mean_exit_time"])):
                dominated = True
                break
        if not dominated:
            pareto.append((c["accuracy"], c["mean_exit_time"]))

    pareto.sort(key=lambda x: x[0])

    # Find optimal configs for target accuracy levels
    optimal = {}
    for target_acc in [0.90, 0.95, 0.99]:
        valid = [c for c in configs if c["accuracy"] >= target_acc]
        if valid:
            best = min(valid, key=lambda x: x["mean_exit_time"])
            optimal[target_acc] = best
        else:
            best = max(configs, key=lambda x: x["accuracy"])
            optimal[target_acc] = best

    return pareto, optimal


def calibrate_early_exit(config: FullConfig) -> CalibrationResult:
    """Run full early-exit calibration on validation set."""
    cfg = config.early_exit
    clf_cfg = config.classifier
    device = torch.device(config.device)

    # Load trained classifier
    ckpt_path = Path(config.paths.checkpoints_dir) / "classifier" / "best_classifier.pt"
    model = PopulationTemporalClassifier(
        feature_dim=clf_cfg.feature_dim,
        temporal_hidden_dim=clf_cfg.temporal_hidden_dim,
        temporal_num_layers=clf_cfg.temporal_num_layers,
        temporal_num_heads=clf_cfg.temporal_num_heads,
        temporal_ffn_dim=clf_cfg.temporal_ffn_dim,
        classifier_hidden_dim=clf_cfg.classifier_hidden_dim,
        num_classes=clf_cfg.num_classes,
        dropout=clf_cfg.dropout,
        use_delta_features=clf_cfg.use_delta_features,
        bin_encoder_type=clf_cfg.bin_encoder_type,
        bin_attn_heads=clf_cfg.bin_attn_heads,
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Get validation experiments
    experiments = build_experiment_list(
        config.paths.data_root, config.paths.features_dir
    )
    _, val_exps, _ = create_splits(
        experiments,
        data_root=config.paths.data_root,
        seed=config.data_split.random_seed,
    )

    # Finer time windows for calibration (every 30 seconds)
    eval_times = list(range(
        cfg.min_time_sec,
        cfg.max_time_sec + 1,
        cfg.eval_interval_sec,
    ))

    logger.info(f"Evaluating at {len(eval_times)} time points on {len(val_exps)} experiments")

    # Evaluate at all time windows
    per_window = evaluate_at_fixed_times(
        model=model,
        experiments=val_exps,
        feature_dir=config.paths.features_dir,
        time_windows=eval_times,
        time_bin_width_sec=clf_cfg.time_bin_width_sec,
        max_crops_per_bin=clf_cfg.max_crops_per_bin,
        feature_dim=clf_cfg.feature_dim,
        device=device,
    )

    # Temperature scaling
    logger.info("Fitting temperature scaling...")
    temp_scaler = TemperatureScaler().to(device)
    last_t = max(eval_times)
    temp_scaler.fit(
        per_window[last_t]["logits"].to(device),
        torch.from_numpy(per_window[last_t]["labels"]).to(device),
    )
    logger.info(f"Optimal temperature: {temp_scaler.temperature.item():.3f}")

    # Apply temperature scaling to all windows
    for t in eval_times:
        with torch.no_grad():
            scaled_logits = temp_scaler(per_window[t]["logits"].to(device))
            per_window[t]["probs"] = F.softmax(scaled_logits, dim=-1)[:, 1].cpu().numpy()
            per_window[t]["preds"] = (per_window[t]["probs"] > 0.5).astype(int)

    # Accuracy vs time
    accuracy_vs_time = {}
    for t in eval_times:
        acc = (per_window[t]["preds"] == per_window[t]["labels"]).mean()
        accuracy_vs_time[t] = acc

    # Find Pareto-optimal operating points
    pareto, optimal = find_pareto_optimal(
        per_window, cfg.patience_range, cfg.threshold_range, eval_times
    )

    # Log results
    logger.info("\nAccuracy vs Time:")
    for t in [60, 120, 300, 600, 900, 1800, 3600]:
        if t in accuracy_vs_time:
            logger.info(f"  {t:5d}s ({t/60:.0f}min): {accuracy_vs_time[t]:.4f}")

    logger.info("\nOptimal configs:")
    for target_acc, cfg_dict in optimal.items():
        logger.info(
            f"  Target {target_acc:.0%}: patience={cfg_dict['patience']}, "
            f"threshold={cfg_dict['threshold']:.2f}, "
            f"mean_time={cfg_dict['mean_exit_time']:.0f}s, "
            f"actual_acc={cfg_dict['accuracy']:.4f}"
        )

    # Save temperature scaler
    temp_path = Path(config.paths.checkpoints_dir) / "classifier" / "temperature_scaler.pt"
    torch.save(temp_scaler.state_dict(), temp_path)

    # Save calibration results
    result = CalibrationResult(
        accuracy_vs_time=accuracy_vs_time,
        pareto_points=pareto,
        optimal_configs=optimal,
        per_window_predictions=per_window,
    )

    import pickle
    cal_path = Path(config.paths.checkpoints_dir) / "classifier" / "calibration_result.pkl"
    with open(cal_path, "wb") as f:
        pickle.dump(result, f)

    logger.info(f"Calibration complete. Results saved to {cal_path}")
    return result
