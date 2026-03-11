"""Generate per-strain result plots for all strain-holdout experiments.

Re-evaluates each fold's checkpoint per-experiment to get per-strain
predictions at each time bucket, then generates publication-quality plots.

Usage:
    PYTHONPATH=/path/to/parent python3 -m ast_classifier.scripts.generate_result_plots \
        --device cuda:0
"""

from __future__ import annotations

import json
import logging
import re
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from torch.utils.data import DataLoader

from ..config import FullConfig
from ..data.dataset import (
    ExperimentMeta,
    PopulationTemporalDataset,
    population_temporal_collate,
)
from ..models.classifier import PopulationTemporalClassifier
from ..models.lstm_classifier import LSTMTemporalClassifier
from ..models.classifier_ctx_aux import ContextualAuxClassifier
from .strain_holdout_eval import build_strain_grouped_experiments, generate_folds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _extract_ec(experiment_id: str) -> str:
    m = re.match(r"^(EC\d+)", experiment_id, re.IGNORECASE)
    return m.group(1).upper() if m else experiment_id


# ---------------------------------------------------------------------------
# Per-experiment evaluation
# ---------------------------------------------------------------------------

def evaluate_per_experiment(
    model: torch.nn.Module,
    test_exps: list[ExperimentMeta],
    config: FullConfig,
    device: torch.device,
    eval_times: list[int],
) -> dict[str, dict[int, float]]:
    """Evaluate model on each test experiment individually.

    Returns {experiment_id: {time_sec: prob_resistant}}.
    """
    cfg = config.classifier
    model.eval()
    results: dict[str, dict[int, float]] = {}

    for exp in test_exps:
        results[exp.experiment_id] = {}
        for window_sec in eval_times:
            dataset = PopulationTemporalDataset(
                feature_dir=config.paths.features_dir,
                experiments=[exp],
                time_bin_width_sec=cfg.time_bin_width_sec,
                time_windows_sec=[window_sec],
                max_crops_per_bin=cfg.max_crops_per_bin,
                feature_dim=cfg.feature_dim,
                random_window=False,
            )
            loader = DataLoader(
                dataset, batch_size=1, shuffle=False,
                num_workers=0, collate_fn=population_temporal_collate,
            )
            with torch.no_grad():
                for batch in loader:
                    batch_gpu = {
                        k: batch[k].to(device, non_blocking=True)
                        for k in [
                            "bin_features", "bin_mask", "crop_mask",
                            "bin_times", "bin_counts", "time_fraction",
                        ]
                    }
                    with torch.amp.autocast("cuda"):
                        output = model(batch_gpu)
                    prob = F.softmax(output["logits"].float(), dim=-1)[0, 1].item()
                    results[exp.experiment_id][window_sec] = prob

    return results


def load_model_for_variant(
    variant_dir: str,
    ckpt_path: Path,
    config: FullConfig,
    device: torch.device,
) -> torch.nn.Module:
    """Load the appropriate model class for a given experiment variant."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "lstm" in variant_dir:
        model = LSTMTemporalClassifier(
            feature_dim=384,
            bin_hidden_dim=128,
            lstm_hidden_dim=128,
            lstm_num_layers=2,
            classifier_hidden_dim=64,
            num_classes=2,
            dropout=0.2,
            max_count_normalizer=256.0,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        return model

    if "ctx_aux" in variant_dir:
        cfg = ckpt.get("config", config.classifier)
        model = ContextualAuxClassifier(
            feature_dim=getattr(cfg, "feature_dim", 384),
            temporal_hidden_dim=getattr(cfg, "temporal_hidden_dim", 256),
            temporal_num_layers=getattr(cfg, "temporal_num_layers", 4),
            temporal_num_heads=getattr(cfg, "temporal_num_heads", 4),
            temporal_ffn_dim=getattr(cfg, "temporal_ffn_dim", 512),
            classifier_hidden_dim=getattr(cfg, "classifier_hidden_dim", 128),
            num_classes=2,
            dropout=getattr(cfg, "dropout", 0.1),
            max_count_normalizer=float(getattr(cfg, "max_crops_per_bin", 256)),
            use_delta_features=getattr(cfg, "use_delta_features", False),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        return model

    # Default: PopulationTemporalClassifier
    cfg = ckpt.get("config", config.classifier)
    model = PopulationTemporalClassifier(
        feature_dim=cfg.feature_dim,
        temporal_hidden_dim=cfg.temporal_hidden_dim,
        temporal_num_layers=cfg.temporal_num_layers,
        temporal_num_heads=cfg.temporal_num_heads,
        temporal_ffn_dim=cfg.temporal_ffn_dim,
        classifier_hidden_dim=cfg.classifier_hidden_dim,
        num_classes=cfg.num_classes,
        dropout=cfg.dropout,
        max_count_normalizer=float(cfg.max_crops_per_bin),
        use_delta_features=cfg.use_delta_features,
        bin_encoder_type=cfg.bin_encoder_type,
        bin_attn_heads=cfg.bin_attn_heads,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    return model


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Color palette for strains (15 strains total)
STRAIN_COLORS = {
    "EC33": "#1f77b4", "EC36": "#ff7f0e", "EC39": "#2ca02c",
    "EC42": "#d62728", "EC67": "#9467bd", "EC79": "#8c564b",
    "EC89": "#e377c2", "EC126": "#7f7f7f",
    "EC35": "#bcbd22", "EC40": "#17becf", "EC48": "#aec7e8",
    "EC58": "#ffbb78", "EC60": "#98df8a", "EC65": "#ff9896",
    "EC87": "#c5b0d5",
}

EXPERIMENT_NAMES = {
    "results_strain_holdout": "Baseline (Transformer + Stats)",
    "results_strain_holdout_delta": "Delta Features",
    "results_strain_holdout_attention": "Attention Bin Encoder",
    "results_strain_holdout_attn_aux": "Attention + Auxiliary Loss",
    "results_strain_holdout_stats_aux": "Stats + Pre-Transformer Aux Loss",
    "results_strain_holdout_lstm": "BiLSTM Temporal Classifier",
    "results_strain_holdout_ctx_aux": "Contextualized Auxiliary Classifier",
}


def plot_experiment_variant(
    variant_name: str,
    display_name: str,
    folds: list[dict],
    per_fold_predictions: list[dict[str, dict[int, float]]],
    eval_times: list[int],
    output_path: Path,
    aggregate_auroc: float,
    aggregate_auroc_std: float,
) -> None:
    """Generate a multi-panel plot for one experiment variant.

    One row per fold + one summary row.
    Each fold panel: per-experiment P(R) over time, colored by strain.
    Summary panel: mean accuracy over time with std band.
    """
    n_folds = len(folds)
    fig, axes = plt.subplots(
        n_folds + 1, 1, figsize=(12, 4 * (n_folds + 1)),
        sharex=True, gridspec_kw={"hspace": 0.3},
    )

    time_minutes = [t / 60 for t in eval_times]

    for fold_idx in range(n_folds):
        ax = axes[fold_idx]
        fold = folds[fold_idx]
        preds = per_fold_predictions[fold_idx]

        holdout_r = fold["holdout_r"]
        holdout_s = fold["holdout_s"]
        test_exps = fold["test"]

        # Compute fold AUROC at 60min
        probs_60 = []
        labels_60 = []
        for exp in test_exps:
            if exp.experiment_id in preds and 3600 in preds[exp.experiment_id]:
                probs_60.append(preds[exp.experiment_id][3600])
                labels_60.append(exp.label)
        try:
            from sklearn.metrics import roc_auc_score
            fold_auroc = roc_auc_score(labels_60, probs_60)
        except (ValueError, ImportError):
            fold_auroc = 0.5

        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, linewidth=1)
        ax.axhspan(0.5, 1.0, alpha=0.03, color="red")
        ax.axhspan(0.0, 0.5, alpha=0.03, color="blue")

        # Plot each test experiment
        for exp in test_exps:
            if exp.experiment_id not in preds:
                continue
            ec = _extract_ec(exp.experiment_id)
            color = STRAIN_COLORS.get(ec, "#333333")
            is_resistant = exp.label == 1
            linestyle = "-" if is_resistant else "--"
            marker = "^" if is_resistant else "v"

            probs = [preds[exp.experiment_id].get(t, float("nan")) for t in eval_times]
            ax.plot(
                time_minutes, probs,
                color=color, linestyle=linestyle, marker=marker,
                markersize=5, linewidth=1.5, alpha=0.8,
                label=f"{ec} ({'R' if is_resistant else 'S'})",
            )

        ax.set_ylabel("P(Resistant)", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

        # Title with fold info
        ax.set_title(
            f"Fold {fold_idx + 1}: Holdout R=[{', '.join(holdout_r)}], "
            f"S=[{', '.join(holdout_s)}]  |  AUROC@60min = {fold_auroc:.3f}",
            fontsize=11, fontweight="bold",
        )

        # Legend (deduplicate by strain)
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        unique_h, unique_l = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                unique_h.append(h)
                unique_l.append(l)
        ax.legend(
            unique_h, unique_l,
            loc="upper left", fontsize=8, ncol=2,
            framealpha=0.9, borderpad=0.3, handlelength=2.5,
        )

        ax.grid(True, alpha=0.2)

    # Summary panel: aggregate accuracy over time
    ax_summary = axes[n_folds]

    # Compute per-fold accuracy at each time point
    fold_accs = {t: [] for t in eval_times}
    for fold_idx in range(n_folds):
        fold = folds[fold_idx]
        preds = per_fold_predictions[fold_idx]
        for t in eval_times:
            correct = 0
            total = 0
            for exp in fold["test"]:
                if exp.experiment_id in preds and t in preds[exp.experiment_id]:
                    prob = preds[exp.experiment_id][t]
                    pred_label = 1 if prob > 0.5 else 0
                    if pred_label == exp.label:
                        correct += 1
                    total += 1
            if total > 0:
                fold_accs[t].append(correct / total)

    mean_acc = [np.mean(fold_accs[t]) if fold_accs[t] else 0 for t in eval_times]
    std_acc = [np.std(fold_accs[t]) if fold_accs[t] else 0 for t in eval_times]

    ax_summary.plot(
        time_minutes, mean_acc,
        color="#1f77b4", linewidth=2.5, marker="o", markersize=6,
        label=f"Mean Accuracy (AUROC = {aggregate_auroc:.3f} +/- {aggregate_auroc_std:.3f})",
    )
    ax_summary.fill_between(
        time_minutes,
        [m - s for m, s in zip(mean_acc, std_acc)],
        [m + s for m, s in zip(mean_acc, std_acc)],
        alpha=0.2, color="#1f77b4",
    )
    ax_summary.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, linewidth=1)

    ax_summary.set_xlabel("Time (minutes)", fontsize=12)
    ax_summary.set_ylabel("Accuracy", fontsize=10)
    ax_summary.set_ylim(-0.05, 1.05)
    ax_summary.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_summary.set_title("Aggregate: Mean Accuracy Across Folds", fontsize=11, fontweight="bold")
    ax_summary.legend(fontsize=10, loc="lower right")
    ax_summary.grid(True, alpha=0.2)

    # X-axis formatting
    ax_summary.set_xticks(time_minutes)
    ax_summary.set_xticklabels([f"{int(t)}" for t in time_minutes])

    # Overall title
    fig.suptitle(
        display_name,
        fontsize=14, fontweight="bold", y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved plot to {output_path}")


def plot_comparison_summary(
    all_results: dict[str, dict],
    eval_times: list[int],
    output_path: Path,
) -> None:
    """Generate a single comparison plot across all experiment variants."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    time_minutes = [t / 60 for t in eval_times]

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    # Left panel: accuracy over time
    for i, (variant, data) in enumerate(all_results.items()):
        display = EXPERIMENT_NAMES.get(variant, variant)
        mean_acc = data["mean_acc"]
        std_acc = data["std_acc"]
        auroc = data["auroc"]
        auroc_std = data["auroc_std"]

        ax1.plot(
            time_minutes, mean_acc,
            color=colors[i], linewidth=2, marker="o", markersize=5,
            label=f"{display} (AUROC={auroc:.2f})",
        )
        ax1.fill_between(
            time_minutes,
            [m - s for m, s in zip(mean_acc, std_acc)],
            [m + s for m, s in zip(mean_acc, std_acc)],
            alpha=0.1, color=colors[i],
        )

    ax1.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel("Time (minutes)", fontsize=12)
    ax1.set_ylabel("Mean Accuracy", fontsize=12)
    ax1.set_title("Accuracy Over Time (Strain-Holdout CV)", fontsize=13, fontweight="bold")
    ax1.set_ylim(0.3, 0.85)
    ax1.set_xticks(time_minutes)
    ax1.set_xticklabels([f"{int(t)}" for t in time_minutes])
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, alpha=0.2)

    # Right panel: AUROC bar chart
    names = []
    aurocs = []
    auroc_stds = []
    for variant, data in all_results.items():
        names.append(EXPERIMENT_NAMES.get(variant, variant).replace(" ", "\n"))
        aurocs.append(data["auroc"])
        auroc_stds.append(data["auroc_std"])

    x = np.arange(len(names))
    bars = ax2.bar(x, aurocs, yerr=auroc_stds, capsize=5, color=colors[:len(names)], alpha=0.8)
    ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax2.set_ylabel("AUROC @ 60 min", fontsize=12)
    ax2.set_title("AUROC Comparison (Strain-Holdout CV)", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=8, ha="center")
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.2, axis="y")

    # Add value labels on bars
    for bar, val, std in zip(bars, aurocs, auroc_stds):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.02,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved comparison plot to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-strain result plots")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("./plots"))
    args = parser.parse_args()

    config = FullConfig()
    config.device = args.device
    device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_times = [60, 120, 180, 300, 600, 900, 1800, 3600]

    # Build strain-grouped experiments and folds (same seed as training)
    groups = build_strain_grouped_experiments(
        config.paths.features_dir, config.paths.data_root
    )
    folds = generate_folds(groups, n_holdout_per_class=2, n_folds=5, seed=42)

    # Define experiment variants to evaluate
    variants = [
        "results_strain_holdout",
        "results_strain_holdout_delta",
        "results_strain_holdout_attention",
        "results_strain_holdout_attn_aux",
        "results_strain_holdout_stats_aux",
        "results_strain_holdout_lstm",
        "results_strain_holdout_ctx_aux",
    ]

    all_comparison_data = {}

    for variant in variants:
        variant_dir = Path(variant)
        ckpt_dir = variant_dir / "checkpoints"
        results_json = variant_dir / "strain_holdout_results.json"

        if not ckpt_dir.exists():
            logger.warning(f"Skipping {variant}: no checkpoints directory")
            continue

        # Check all fold checkpoints exist
        fold_ckpts = [ckpt_dir / f"fold{i}_best.pt" for i in range(5)]
        if not all(p.exists() for p in fold_ckpts):
            logger.warning(f"Skipping {variant}: missing fold checkpoints")
            continue

        display_name = EXPERIMENT_NAMES.get(variant, variant)
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {display_name}")
        logger.info(f"{'='*60}")

        # Load aggregate results if available (computed from predictions later if missing)
        if results_json.exists():
            with open(results_json) as f:
                agg = json.load(f)
            aggregate_auroc = agg["mean_auroc_60min"]
            aggregate_auroc_std = agg["std_auroc_60min"]
        else:
            aggregate_auroc = None  # will compute from predictions
            aggregate_auroc_std = None

        per_fold_predictions = []

        for fold_idx in range(5):
            fold = folds[fold_idx]
            ckpt_path = fold_ckpts[fold_idx]

            logger.info(
                f"  Fold {fold_idx}: loading {ckpt_path.name}, "
                f"evaluating {len(fold['test'])} test experiments"
            )

            model = load_model_for_variant(variant, ckpt_path, config, device)
            preds = evaluate_per_experiment(
                model, fold["test"], config, device, eval_times
            )
            per_fold_predictions.append(preds)

            # Clean up GPU memory
            del model
            torch.cuda.empty_cache()

        # Compute AUROC from predictions if not available from JSON
        if aggregate_auroc is None:
            from sklearn.metrics import roc_auc_score as _roc_auc
            fold_aurocs = []
            for fi in range(5):
                probs_60 = []
                labels_60 = []
                for exp in folds[fi]["test"]:
                    eid = exp.experiment_id
                    if eid in per_fold_predictions[fi] and 3600 in per_fold_predictions[fi][eid]:
                        probs_60.append(per_fold_predictions[fi][eid][3600])
                        labels_60.append(exp.label)
                try:
                    fold_aurocs.append(_roc_auc(labels_60, probs_60))
                except ValueError:
                    fold_aurocs.append(0.5)
            aggregate_auroc = float(np.mean(fold_aurocs))
            aggregate_auroc_std = float(np.std(fold_aurocs))

        # Generate per-variant plot
        plot_path = output_dir / f"{variant}.png"
        plot_experiment_variant(
            variant_name=variant,
            display_name=display_name,
            folds=folds,
            per_fold_predictions=per_fold_predictions,
            eval_times=eval_times,
            output_path=plot_path,
            aggregate_auroc=aggregate_auroc,
            aggregate_auroc_std=aggregate_auroc_std,
        )

        # Save per-experiment predictions as JSON for future reference
        predictions_json = output_dir / f"{variant}_predictions.json"
        serializable = []
        for fold_idx in range(5):
            fold = folds[fold_idx]
            fold_data = {
                "fold": fold_idx,
                "holdout_r": fold["holdout_r"],
                "holdout_s": fold["holdout_s"],
                "experiments": {},
            }
            for exp in fold["test"]:
                ec = _extract_ec(exp.experiment_id)
                preds = per_fold_predictions[fold_idx].get(exp.experiment_id, {})
                fold_data["experiments"][exp.experiment_id] = {
                    "strain": ec,
                    "label": exp.label,
                    "label_name": "Resistant" if exp.label == 1 else "Susceptible",
                    "predictions": {str(t): round(p, 4) for t, p in preds.items()},
                }
            serializable.append(fold_data)

        with open(predictions_json, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"  Saved predictions to {predictions_json}")

        # Collect data for comparison plot
        fold_accs = {t: [] for t in eval_times}
        for fold_idx in range(5):
            fold = folds[fold_idx]
            preds = per_fold_predictions[fold_idx]
            for t in eval_times:
                correct = total = 0
                for exp in fold["test"]:
                    if exp.experiment_id in preds and t in preds[exp.experiment_id]:
                        pred_label = 1 if preds[exp.experiment_id][t] > 0.5 else 0
                        if pred_label == exp.label:
                            correct += 1
                        total += 1
                if total > 0:
                    fold_accs[t].append(correct / total)

        all_comparison_data[variant] = {
            "mean_acc": [float(np.mean(fold_accs[t])) for t in eval_times],
            "std_acc": [float(np.std(fold_accs[t])) for t in eval_times],
            "auroc": aggregate_auroc,
            "auroc_std": aggregate_auroc_std,
        }

    # Generate comparison plot
    if all_comparison_data:
        plot_comparison_summary(
            all_comparison_data, eval_times,
            output_dir / "comparison_summary.png",
        )

    logger.info(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
