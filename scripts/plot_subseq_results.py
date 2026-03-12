"""Generate plots for sub-sequence sampling strain-holdout results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_all_results() -> dict[str, dict]:
    """Load results from all strain-holdout experiment directories."""
    base = Path("/home/mkedz/code/ast_classifier")
    variants = {
        "Stats baseline": base / "results_strain_holdout",
        "Delta features": base / "results_strain_holdout_delta",
        "Attention bin": base / "results_strain_holdout_attention",
        "Attention + aux": base / "results_strain_holdout_attn_aux",
        "Stats + aux": base / "results_strain_holdout_stats_aux",
        "LSTM": base / "results_strain_holdout_lstm",
        "Ctx aux": base / "results_strain_holdout_ctx_aux",
        "Crop MLP": base / "results_crop_mlp",
        "Sub-seq sampling": base / "results_strain_holdout_subseq",
    }
    results = {}
    for name, path in variants.items():
        f = path / "strain_holdout_results.json"
        if f.exists():
            with open(f) as fh:
                results[name] = json.load(fh)
    return results


def _get_auroc(r: dict) -> tuple[float, float]:
    """Extract mean/std AUROC from either result format."""
    if "mean_auroc_60min" in r:
        return r["mean_auroc_60min"], r["std_auroc_60min"]
    return r.get("mean_experiment_auroc", 0.5), r.get("std_experiment_auroc", 0.0)


def _get_acc_vs_time(r: dict) -> tuple[dict, dict]:
    """Extract accuracy vs time from either result format."""
    if "mean_accuracy_vs_time" in r and "std_accuracy_vs_time" in r:
        return r["mean_accuracy_vs_time"], r["std_accuracy_vs_time"]
    # Crop MLP has mean_accuracy_vs_time but no std
    acc = r.get("mean_accuracy_vs_time", {})
    std = {k: 0.0 for k in acc}
    return acc, std


def plot_auroc_comparison(results: dict, output_dir: Path) -> None:
    """Bar chart of AUROC across all variants."""
    fig, ax = plt.subplots(figsize=(12, 5))

    names = list(results.keys())
    means = [_get_auroc(r)[0] for r in results.values()]
    stds = [_get_auroc(r)[1] for r in results.values()]

    colors = []
    for name in names:
        if name == "Sub-seq sampling":
            colors.append("#2196F3")
        elif name == "Stats baseline":
            colors.append("#4CAF50")
        else:
            colors.append("#9E9E9E")

    bars = ax.bar(range(len(names)), means, yerr=stds, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("AUROC @ 60 min")
    ax.set_title("Strain-Holdout AUROC: All Classifier Variants")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.02, f"{m:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "auroc_comparison_all_variants.png", dpi=150)
    plt.close(fig)


def plot_accuracy_vs_time_comparison(results: dict, output_dir: Path) -> None:
    """Accuracy vs time for baseline, sub-sequence, and crop MLP."""
    fig, ax = plt.subplots(figsize=(10, 6))

    highlight = {
        "Stats baseline": ("#4CAF50", "-", "o"),
        "Sub-seq sampling": ("#2196F3", "-", "s"),
        "Crop MLP": ("#FF9800", "--", "^"),
        "Delta features": ("#9C27B0", "--", "d"),
    }

    for name, r in results.items():
        if name not in highlight:
            continue
        color, ls, marker = highlight[name]
        acc, std = _get_acc_vs_time(r)
        times = sorted(int(t) for t in acc.keys())
        means = [acc[str(t)] for t in times]
        stds = [std[str(t)] for t in times]
        time_min = [t / 60 for t in times]

        ax.errorbar(time_min, means, yerr=stds, label=name,
                     color=color, linestyle=ls, marker=marker,
                     capsize=3, markersize=5, linewidth=1.5)

    ax.axhline(y=0.5, color="red", linestyle=":", alpha=0.4, label="Random")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Experiment-Level Accuracy vs Observation Time")
    ax.legend(fontsize=9)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlim(0, 65)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_time_comparison.png", dpi=150)
    plt.close(fig)


def plot_per_fold_auroc(subseq_results: dict, output_dir: Path) -> None:
    """Per-fold AUROC for sub-sequence sampling."""
    fig, ax = plt.subplots(figsize=(10, 5))

    folds = subseq_results["folds"]
    fold_labels = []
    aurocs = []
    colors = []

    for f in folds:
        label = f"F{f['fold']}: R={','.join(f['holdout_r'])}\nS={','.join(f['holdout_s'])}"
        fold_labels.append(label)
        aurocs.append(f["auroc_60min"])
        colors.append("#F44336" if f["auroc_60min"] < 0.5 else "#2196F3")

    bars = ax.bar(range(len(fold_labels)), aurocs, color=colors,
                  edgecolor="black", linewidth=0.5, alpha=0.85)

    ax.set_xticks(range(len(fold_labels)))
    ax.set_xticklabels(fold_labels, fontsize=8)
    ax.set_ylabel("Test AUROC @ 60 min")
    ax.set_title("Sub-Sequence Sampling: Per-Fold Test AUROC")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random")
    ax.axhline(y=subseq_results["mean_auroc_60min"], color="blue",
               linestyle=":", alpha=0.7,
               label=f"Mean = {subseq_results['mean_auroc_60min']:.3f}")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)

    for i, v in enumerate(aurocs):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_dir / "per_fold_auroc_subseq.png", dpi=150)
    plt.close(fig)


def plot_per_fold_accuracy_vs_time(subseq_results: dict, output_dir: Path) -> None:
    """Per-fold accuracy vs time for sub-sequence sampling."""
    fig, ax = plt.subplots(figsize=(10, 6))

    fold_colors = ["#2196F3", "#4CAF50", "#F44336", "#FF9800", "#9C27B0"]

    for f in subseq_results["folds"]:
        acc = f["accuracy_vs_time"]
        times = sorted(int(t) for t in acc.keys())
        vals = [acc[str(t)] for t in times]
        time_min = [t / 60 for t in times]
        label = f"F{f['fold']}: R={','.join(f['holdout_r'])}, S={','.join(f['holdout_s'])}"
        ax.plot(time_min, vals, marker="o", markersize=4, linewidth=1.5,
                color=fold_colors[f["fold"]], label=label)

    # Mean line
    mean_acc = subseq_results["mean_accuracy_vs_time"]
    times = sorted(int(t) for t in mean_acc.keys())
    means = [mean_acc[str(t)] for t in times]
    time_min = [t / 60 for t in times]
    ax.plot(time_min, means, "k-", linewidth=2.5, marker="s", markersize=5,
            label="Mean", zorder=10)

    ax.axhline(y=0.5, color="red", linestyle=":", alpha=0.4)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Sub-Sequence Sampling: Accuracy vs Time (Per Fold)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_ylim(0.2, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "per_fold_accuracy_vs_time_subseq.png", dpi=150)
    plt.close(fig)


def main() -> None:
    output_dir = Path("/home/mkedz/code/ast_classifier/results_strain_holdout_subseq/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_all_results()
    subseq = results["Sub-seq sampling"]

    plot_auroc_comparison(results, output_dir)
    plot_accuracy_vs_time_comparison(results, output_dir)
    plot_per_fold_auroc(subseq, output_dir)
    plot_per_fold_accuracy_vs_time(subseq, output_dir)

    print(f"Plots saved to {output_dir}")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
