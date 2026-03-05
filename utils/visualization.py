"""Visualization utilities for analysis and publication figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

from ..models.early_exit import EarlyExitResult


def plot_accuracy_vs_time(
    accuracy_vs_time: dict[float, float],
    save_path: Path,
    title: str = "Classification Accuracy vs. Observation Time",
) -> None:
    """Line plot: accuracy at each evaluation time point."""
    times = sorted(accuracy_vs_time.keys())
    accs = [accuracy_vs_time[t] for t in times]
    times_min = [t / 60 for t in times]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times_min, accs, "b-o", markersize=4, linewidth=2)
    ax.axhline(y=0.90, color="orange", linestyle="--", alpha=0.7, label="90% threshold")
    ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.7, label="95% threshold")
    ax.set_xlabel("Observation Time (minutes)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_exit_time_distribution(
    exit_results: list[EarlyExitResult],
    save_path: Path,
    title: str = "Early-Exit Time Distribution",
) -> None:
    """Histogram of exit times."""
    times = np.array([r.exit_time_sec for r in exit_results]) / 60

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(times, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    ax.set_xlabel("Exit Time (minutes)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.axvline(np.median(times), color="red", linestyle="--", label=f"Median: {np.median(times):.1f} min")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    bin_times_min: np.ndarray | None = None,
    save_path: Path | None = None,
    title: str = "Time Bin Attention Weights",
) -> None:
    """Visualizes which time bins the model attends to most."""
    fig, ax = plt.subplots(figsize=(12, 3))

    order = np.argsort(attention_weights)[::-1]
    sorted_weights = attention_weights[order]

    if bin_times_min is not None:
        labels = [f"{bin_times_min[i]:.0f}m" for i in order]
    else:
        labels = [str(i) for i in order]
    n_show = min(30, len(sorted_weights))

    ax.bar(range(n_show), sorted_weights[:n_show], color="steelblue")
    ax.set_xticks(range(n_show))
    ax.set_xticklabels(labels[:n_show], rotation=45, fontsize=8)
    ax.set_xlabel("Time Bin", fontsize=12)
    ax.set_ylabel("Attention Weight", fontsize=12)
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tsne_embeddings(
    features: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
    perplexity: int = 30,
    title: str = "t-SNE of DINO Embeddings",
    max_points: int = 5000,
) -> None:
    """t-SNE of DINO embeddings colored by resistance label."""
    if len(features) > max_points:
        idx = np.random.choice(len(features), max_points, replace=False)
        features = features[idx]
        labels = labels[idx]

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings = tsne.fit_transform(features.astype(np.float32))

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {0: "dodgerblue", 1: "crimson"}
    names = {0: "Susceptible", 1: "Resistant"}

    for label_val in [0, 1]:
        mask = labels == label_val
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=colors[label_val],
            label=names[label_val],
            alpha=0.5,
            s=10,
        )

    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12, markerscale=3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_population_heterogeneity(
    features_over_time: dict[float, np.ndarray],
    labels: dict[float, int],
    save_path: Path,
    title: str = "Population Feature Heterogeneity Over Time",
) -> None:
    """Feature variance over time: susceptible populations should show
    increasing heterogeneity after antibiotic exposure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for label_val, color, name in [(0, "dodgerblue", "Susceptible"), (1, "crimson", "Resistant")]:
        times = sorted([t for t, l in labels.items() if l == label_val])
        if not times:
            continue
        variances = []
        for t in times:
            if t in features_over_time:
                var = np.var(features_over_time[t], axis=0).mean()
                variances.append(var)
        if variances:
            ax.plot(
                [t / 60 for t in times[:len(variances)]],
                variances,
                color=color,
                label=name,
                linewidth=2,
            )

    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_ylabel("Mean Feature Variance", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pareto_front(
    pareto_points: list[tuple[float, float]],
    all_configs: list[dict] | None = None,
    save_path: Path | None = None,
    title: str = "Accuracy vs. Time-to-Prediction Pareto Front",
) -> None:
    """Plot Pareto-optimal operating points."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if all_configs:
        accs = [c["accuracy"] for c in all_configs]
        times = [c["mean_exit_time"] / 60 for c in all_configs]
        ax.scatter(times, accs, c="lightgray", s=20, alpha=0.5, label="All configs")

    if pareto_points:
        p_accs = [p[0] for p in pareto_points]
        p_times = [p[1] / 60 for p in pareto_points]
        ax.plot(p_times, p_accs, "ro-", markersize=8, linewidth=2, label="Pareto front")

    ax.set_xlabel("Mean Exit Time (minutes)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
