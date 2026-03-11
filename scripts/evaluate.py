"""CLI entry point for evaluation with early-exit analysis.

Usage:
    python -m ast_classifier.scripts.evaluate \
        --data-root /path/to/MainFolder \
        --features-dir /path/to/features \
        --checkpoints-dir /path/to/checkpoints \
        --output-dir /path/to/results
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..config import FullConfig
from ..data.dataset import build_experiment_list, create_splits
from ..models.classifier import PopulationTemporalClassifier
from ..models.early_exit import EarlyExitPolicy, EarlyExitResult, TemperatureScaler
from ..training.calibrate_exit import evaluate_at_fixed_times
from ..utils.metrics import (
    compute_metrics,
    time_to_prediction_analysis,
)
from ..utils.visualization import (
    plot_accuracy_vs_time,
    plot_exit_time_distribution,
    plot_pareto_front,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate(config: FullConfig, output_dir: Path) -> None:
    """Run full evaluation on test set."""
    clf_cfg = config.classifier
    exit_cfg = config.early_exit
    device = torch.device(config.device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
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

    # Load temperature scaler if available
    temp_scaler = None
    temp_path = Path(config.paths.checkpoints_dir) / "classifier" / "temperature_scaler.pt"
    if temp_path.exists():
        temp_scaler = TemperatureScaler().to(device)
        temp_scaler.load_state_dict(
            torch.load(temp_path, map_location=device, weights_only=True)
        )

    # Get test experiments
    experiments = build_experiment_list(
        config.paths.data_root, config.paths.features_dir
    )
    _, _, test_exps = create_splits(
        experiments,
        data_root=config.paths.data_root,
        seed=config.data_split.random_seed,
    )
    logger.info(f"Evaluating on {len(test_exps)} test experiments")

    # Evaluate at all time windows
    eval_times = list(range(
        exit_cfg.min_time_sec,
        exit_cfg.max_time_sec + 1,
        exit_cfg.eval_interval_sec,
    ))

    per_window = evaluate_at_fixed_times(
        model=model,
        experiments=test_exps,
        feature_dir=config.paths.features_dir,
        time_windows=eval_times,
        time_bin_width_sec=clf_cfg.time_bin_width_sec,
        max_crops_per_bin=clf_cfg.max_crops_per_bin,
        feature_dim=clf_cfg.feature_dim,
        device=device,
    )

    # Apply temperature scaling
    if temp_scaler is not None:
        import torch.nn.functional as F
        for t in eval_times:
            with torch.no_grad():
                scaled = temp_scaler(per_window[t]["logits"].to(device))
                per_window[t]["probs"] = F.softmax(scaled, dim=-1)[:, 1].cpu().numpy()
                per_window[t]["preds"] = (per_window[t]["probs"] > 0.5).astype(int)

    # 1. Accuracy vs time
    accuracy_vs_time = {}
    for t in eval_times:
        acc = (per_window[t]["preds"] == per_window[t]["labels"]).mean()
        accuracy_vs_time[t] = float(acc)

    plot_accuracy_vs_time(accuracy_vs_time, output_dir / "accuracy_vs_time.png")

    # 2. Full metrics at final time point
    last_t = max(eval_times)
    full_metrics = compute_metrics(
        labels=per_window[last_t]["labels"],
        predictions=per_window[last_t]["preds"],
        probabilities=per_window[last_t]["probs"],
    )

    # 3. Simulate early exit on test set
    cal_path = Path(config.paths.checkpoints_dir) / "classifier" / "calibration_result.pkl"
    if cal_path.exists():
        with open(cal_path, "rb") as f:
            cal_result = pickle.load(f)
        if 0.95 in cal_result.optimal_configs:
            opt = cal_result.optimal_configs[0.95]
            patience = opt["patience"]
            threshold = opt["threshold"]
        else:
            patience = exit_cfg.patience
            threshold = exit_cfg.confidence_threshold
    else:
        patience = exit_cfg.patience
        threshold = exit_cfg.confidence_threshold

    # Simulate early exit
    exit_results = []
    for exp_idx in range(len(test_exps)):
        consecutive = 0
        last_pred = -1

        for t in sorted(eval_times):
            prob = per_window[t]["probs"][exp_idx]
            label = per_window[t]["labels"][exp_idx]
            pred = int(prob > 0.5)
            conf = max(prob, 1 - prob)

            if conf >= threshold and pred == last_pred:
                consecutive += 1
            elif conf >= threshold:
                consecutive = 1
                last_pred = pred
            else:
                consecutive = 0
                last_pred = -1

            if consecutive >= patience and t >= exit_cfg.min_time_sec:
                exit_results.append(EarlyExitResult(
                    prediction=pred,
                    confidence=float(conf),
                    exit_time_sec=float(t),
                    prediction_history=[],
                ))
                break
        else:
            prob = per_window[last_t]["probs"][exp_idx]
            exit_results.append(EarlyExitResult(
                prediction=int(prob > 0.5),
                confidence=float(max(prob, 1 - prob)),
                exit_time_sec=float(last_t),
                prediction_history=[],
            ))

    # 4. Time-to-prediction analysis
    ttp_metrics = time_to_prediction_analysis(exit_results, accuracy_vs_time)

    # 5. Early exit metrics
    exit_preds = np.array([r.prediction for r in exit_results])
    exit_labels = per_window[last_t]["labels"]
    exit_probs = np.array([r.confidence for r in exit_results])
    early_exit_metrics = compute_metrics(exit_labels, exit_preds, exit_probs)

    # 6. Plots
    plot_exit_time_distribution(exit_results, output_dir / "exit_time_distribution.png")

    if cal_path.exists():
        plot_pareto_front(
            cal_result.pareto_points,
            save_path=output_dir / "pareto_front.png",
        )

    # 7. Save results
    results = {
        "full_metrics_at_60min": asdict(full_metrics),
        "early_exit_metrics": asdict(early_exit_metrics),
        "time_to_prediction": asdict(ttp_metrics),
        "accuracy_vs_time": accuracy_vs_time,
        "early_exit_config": {
            "patience": patience,
            "threshold": threshold,
        },
        "n_test_experiments": len(test_exps),
    }

    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"\nFull metrics at 60 min:")
    for k, v in asdict(full_metrics).items():
        logger.info(f"  {k}: {v:.4f}")

    logger.info(f"\nEarly-exit metrics (patience={patience}, threshold={threshold:.2f}):")
    for k, v in asdict(early_exit_metrics).items():
        logger.info(f"  {k}: {v:.4f}")

    logger.info(f"\nTime-to-prediction:")
    logger.info(f"  Mean exit time: {ttp_metrics.mean_exit_time_sec:.0f}s ({ttp_metrics.mean_exit_time_sec/60:.1f} min)")
    logger.info(f"  Median exit time: {ttp_metrics.median_exit_time_sec:.0f}s ({ttp_metrics.median_exit_time_sec/60:.1f} min)")
    if ttp_metrics.time_for_95_acc:
        logger.info(f"  Time to 95% acc: {ttp_metrics.time_for_95_acc:.0f}s ({ttp_metrics.time_for_95_acc/60:.1f} min)")

    logger.info(f"\nAccuracy at key time points:")
    for t in [300, 600, 900, 1800, 3600]:
        if t in accuracy_vs_time:
            logger.info(f"  {t//60:3d} min: {accuracy_vs_time[t]:.4f}")

    logger.info(f"\nResults saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate AST classifier with early-exit analysis"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--checkpoints-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    config = FullConfig()
    config.paths.data_root = args.data_root
    config.paths.features_dir = args.features_dir
    config.paths.checkpoints_dir = args.checkpoints_dir
    config.device = args.device

    evaluate(config, args.output_dir)


if __name__ == "__main__":
    main()
