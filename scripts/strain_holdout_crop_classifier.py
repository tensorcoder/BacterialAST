"""Per-crop MLP classifier with strain-holdout cross-validation.

Instead of population-level temporal analysis, this classifier operates on
individual bacterium DINO embeddings. It trains only on late-time crops
(t > 40 min) where R/S differences are most pronounced, then evaluates
across all time points to show how the signal evolves.

Output plots show per-experiment time-series of R vs S classification
in 5-minute bins, revealing when the morphological signal emerges.

Usage:
    PYTHONPATH=/path/to/parent python3 -m ast_classifier.scripts.strain_holdout_crop_classifier \
        --output-dir ./results_crop_mlp \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from ..config import FullConfig
from ..data.dataset import ExperimentMeta

# Reuse fold generation from existing strain-holdout script
from .strain_holdout_eval import build_strain_grouped_experiments, generate_folds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CropMLP(nn.Module):
    """Small MLP for per-crop R/S classification.

    384 -> 128 -> 64 -> 2  with LayerNorm + GELU + Dropout.
    """

    def __init__(self, in_dim: int = 384, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_experiment_crops(
    exp: ExperimentMeta,
    features_dir: Path,
    min_time_sec: float | None = None,
    max_crops: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load features and relative timestamps for one experiment.

    Args:
        min_time_sec: If set, only return crops with relative_ts >= this value.
        max_crops: If set, subsample to this many crops.

    Returns:
        (features, rel_timestamps) or None if no crops match.
    """
    npz_path = features_dir / f"{exp.experiment_id}.npz"
    if not npz_path.exists():
        npz_path = exp.features_path
    if not npz_path.exists():
        return None

    data = np.load(npz_path)
    features = data["features"].astype(np.float32)  # (N, 384)
    timestamps = data["timestamps"].astype(np.float64)
    rel_ts = (timestamps - timestamps.min()).astype(np.float32)

    if min_time_sec is not None:
        mask = rel_ts >= min_time_sec
        features = features[mask]
        rel_ts = rel_ts[mask]

    if len(features) == 0:
        return None

    if max_crops is not None and len(features) > max_crops:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(features), max_crops, replace=False)
        idx.sort()
        features = features[idx]
        rel_ts = rel_ts[idx]

    return features, rel_ts


def load_crops_for_training(
    experiments: list[ExperimentMeta],
    features_dir: Path,
    min_time_sec: float = 2400.0,
    max_per_exp: int = 10000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Load late-time crops from multiple experiments for training.

    Returns:
        (all_features, all_labels) concatenated arrays.
    """
    all_features = []
    all_labels = []

    for exp in experiments:
        result = load_experiment_crops(
            exp, features_dir, min_time_sec=min_time_sec,
            max_crops=max_per_exp, seed=seed,
        )
        if result is None:
            logger.debug(
                f"  Skipping {exp.experiment_id} (no crops after {min_time_sec}s)"
            )
            continue

        features, _ = result
        all_features.append(features)
        all_labels.append(np.full(len(features), exp.label, dtype=np.int64))

    return np.concatenate(all_features), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_fold(
    fold_idx: int,
    fold: dict,
    features_dir: Path,
    ckpt_dir: Path,
    device: torch.device,
    min_time_sec: float = 2400.0,
    max_per_exp: int = 10000,
    seed: int = 42,
) -> Path:
    """Train CropMLP for one fold. Returns path to best checkpoint."""
    train_exps = fold["train"]
    val_exps = fold["val"]

    logger.info(
        f"Fold {fold_idx}: {len(train_exps)} train, {len(val_exps)} val, "
        f"{len(fold['test'])} test | Holdout R: {fold['holdout_r']}, "
        f"S: {fold['holdout_s']}"
    )

    # Load training and validation crops
    train_X, train_y = load_crops_for_training(
        train_exps, features_dir, min_time_sec=min_time_sec,
        max_per_exp=max_per_exp, seed=seed,
    )
    val_X, val_y = load_crops_for_training(
        val_exps, features_dir, min_time_sec=min_time_sec,
        max_per_exp=max_per_exp, seed=seed,
    )

    logger.info(
        f"  Train: {len(train_X)} crops "
        f"(R={np.sum(train_y == 1)}, S={np.sum(train_y == 0)})"
    )
    logger.info(
        f"  Val:   {len(val_X)} crops "
        f"(R={np.sum(val_y == 1)}, S={np.sum(val_y == 0)})"
    )

    # Class weights for imbalanced data
    n_per_class = np.bincount(train_y, minlength=2).astype(np.float32)
    class_weights = len(train_y) / (2 * n_per_class)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    logger.info(f"  Class weights: S={class_weights[0]:.2f}, R={class_weights[1]:.2f}")

    # Dataloaders
    train_ds = TensorDataset(
        torch.from_numpy(train_X), torch.from_numpy(train_y)
    )
    val_ds = TensorDataset(
        torch.from_numpy(val_X), torch.from_numpy(val_y)
    )
    train_loader = DataLoader(
        train_ds, batch_size=2048, shuffle=True, pin_memory=True, num_workers=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=4096, shuffle=False, pin_memory=True, num_workers=2,
    )

    # Model
    model = CropMLP(in_dim=train_X.shape[1]).to(device)
    if fold_idx == 0:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  CropMLP parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    total_steps = len(train_loader) * 100
    warmup_steps = len(train_loader) * 5

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.5 * 0.99 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_auc = 0.0
    patience_counter = 0
    patience = 15
    save_path = ckpt_dir / f"fold{fold_idx}_best.pt"

    for epoch in range(100):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(features)
            loss = F.cross_entropy(logits, labels, weight=class_weights)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validate
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device, non_blocking=True)
                logits = model(features)
                probs = F.softmax(logits.float(), dim=-1)[:, 1]
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.numpy())

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            val_auc = 0.5

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_auc": val_auc,
            }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        if (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(
                f"  Fold {fold_idx} Epoch {epoch+1} - "
                f"Loss: {avg_loss:.4f} Val AUC: {val_auc:.4f}"
            )

    logger.info(f"  Fold {fold_idx} best val AUC: {best_val_auc:.4f}")
    return save_path


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_fold(
    fold_idx: int,
    fold: dict,
    ckpt_path: Path,
    features_dir: Path,
    device: torch.device,
    bin_width_sec: float = 300.0,
) -> dict:
    """Evaluate on held-out strains: per-crop predictions across all time points."""
    model = CropMLP().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_exps = fold["test"]
    per_experiment = {}

    for exp in test_exps:
        result = load_experiment_crops(exp, features_dir)
        if result is None:
            logger.warning(f"  No crops for {exp.experiment_id}")
            continue

        features, rel_ts = result
        max_time = float(rel_ts.max())
        n_bins = max(1, int(np.ceil(max_time / bin_width_sec)))

        # Run inference on all crops
        features_t = torch.from_numpy(features).to(device)
        all_probs = []
        with torch.no_grad():
            # Process in chunks to avoid OOM
            for start in range(0, len(features_t), 8192):
                chunk = features_t[start:start + 8192]
                logits = model(chunk)
                probs = F.softmax(logits.float(), dim=-1)[:, 1]
                all_probs.append(probs.cpu().numpy())
        crop_probs = np.concatenate(all_probs)  # P(Resistant) per crop

        # Bin by 5-minute intervals
        bin_data = []
        for b in range(n_bins):
            t_lo = b * bin_width_sec
            t_hi = (b + 1) * bin_width_sec
            in_bin = (rel_ts >= t_lo) & (rel_ts < t_hi)

            if np.sum(in_bin) == 0:
                bin_data.append({
                    "bin_center_min": (t_lo + t_hi) / 2 / 60,
                    "n_crops": 0,
                    "frac_resistant": None,
                    "mean_prob_r": None,
                })
                continue

            bin_probs = crop_probs[in_bin]
            bin_preds = (bin_probs > 0.5).astype(int)

            bin_data.append({
                "bin_center_min": (t_lo + t_hi) / 2 / 60,
                "n_crops": int(np.sum(in_bin)),
                "frac_resistant": float(np.mean(bin_preds)),
                "mean_prob_r": float(np.mean(bin_probs)),
            })

        # Experiment-level prediction: majority vote at 60 min (all crops)
        exp_prob_r = float(np.mean(crop_probs))
        exp_pred = int(exp_prob_r > 0.5)

        # Also compute cumulative predictions at various time points
        cumulative_preds = {}
        for t_sec in [300, 600, 900, 1200, 1500, 1800, 2400, 3000, 3600]:
            mask = rel_ts <= t_sec
            if np.sum(mask) == 0:
                continue
            cum_prob = float(np.mean(crop_probs[mask]))
            cumulative_preds[t_sec] = {
                "prob_r": cum_prob,
                "pred": int(cum_prob > 0.5),
                "correct": int((cum_prob > 0.5) == exp.label),
                "n_crops": int(np.sum(mask)),
            }

        per_experiment[exp.experiment_id] = {
            "label": exp.label,
            "label_name": "R" if exp.label == 1 else "S",
            "exp_prob_r": exp_prob_r,
            "exp_pred": exp_pred,
            "correct": int(exp_pred == exp.label),
            "n_total_crops": len(features),
            "max_time_min": max_time / 60,
            "bin_timeseries": bin_data,
            "cumulative_preds": {
                str(k): v for k, v in cumulative_preds.items()
            },
        }

    # Aggregate experiment-level metrics
    labels = np.array([v["label"] for v in per_experiment.values()])
    probs = np.array([v["exp_prob_r"] for v in per_experiment.values()])
    preds = (probs > 0.5).astype(int)
    acc = float(np.mean(preds == labels))
    try:
        auroc = float(roc_auc_score(labels, probs))
    except ValueError:
        auroc = 0.5

    # Accuracy at cumulative time points
    accuracy_vs_time = {}
    for t_sec in [300, 600, 900, 1200, 1500, 1800, 2400, 3000, 3600]:
        correct = []
        for v in per_experiment.values():
            if str(t_sec) in v["cumulative_preds"]:
                correct.append(v["cumulative_preds"][str(t_sec)]["correct"])
        if correct:
            accuracy_vs_time[t_sec] = float(np.mean(correct))

    return {
        "fold": fold_idx,
        "holdout_r": fold["holdout_r"],
        "holdout_s": fold["holdout_s"],
        "n_train": len(fold["train"]),
        "n_val": len(fold["val"]),
        "n_test": len(fold["test"]),
        "experiment_accuracy": acc,
        "experiment_auroc": auroc,
        "accuracy_vs_time": accuracy_vs_time,
        "per_experiment": per_experiment,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Color palette for strains
STRAIN_COLORS = {
    "EC35": "#e41a1c", "EC40": "#377eb8", "EC48": "#4daf4a",
    "EC58": "#984ea3", "EC60": "#ff7f00", "EC65": "#a65628",
    "EC87": "#f781bf", "EC126": "#1b9e77", "EC33": "#d95f02",
    "EC36": "#7570b3", "EC39": "#e7298a", "EC42": "#66a61e",
    "EC67": "#e6ab02", "EC79": "#a6761d", "EC89": "#666666",
}


def _extract_ec(experiment_id: str) -> str:
    import re
    m = re.match(r"^(EC\d+)", experiment_id, re.IGNORECASE)
    return m.group(1).upper() if m else experiment_id[:6]


def plot_fold_timeseries(
    fold_idx: int,
    fold_result: dict,
    output_dir: Path,
) -> None:
    """Plot per-experiment time-series of P(Resistant) for one fold."""
    per_exp = fold_result["per_experiment"]
    if not per_exp:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for exp_id, exp_data in sorted(per_exp.items()):
        bins = exp_data["bin_timeseries"]
        times = [b["bin_center_min"] for b in bins if b["mean_prob_r"] is not None]
        probs = [b["mean_prob_r"] for b in bins if b["mean_prob_r"] is not None]

        if not times:
            continue

        ec = _extract_ec(exp_id)
        color = STRAIN_COLORS.get(ec, "#333333")
        is_resistant = exp_data["label"] == 1
        linestyle = "-" if is_resistant else "--"
        marker = "o" if is_resistant else "s"

        ax.plot(
            times, probs,
            color=color, linestyle=linestyle, marker=marker,
            markersize=3, linewidth=1.5, alpha=0.8,
            label=f"{ec} ({'R' if is_resistant else 'S'})",
        )

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.axhspan(0.5, 1.0, alpha=0.05, color="red")
    ax.axhspan(0.0, 0.5, alpha=0.05, color="blue")

    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_ylabel("Mean P(Resistant)", fontsize=12)
    ax.set_title(
        f"Fold {fold_idx} | Holdout R: {fold_result['holdout_r']}, "
        f"S: {fold_result['holdout_s']} | "
        f"AUROC: {fold_result['experiment_auroc']:.3f}",
        fontsize=11,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 65)

    # De-duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(), by_label.keys(),
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=8, framealpha=0.9,
    )

    fig.tight_layout()
    fig.savefig(output_dir / f"fold{fold_idx}_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fold_crop_counts(
    fold_idx: int,
    fold_result: dict,
    output_dir: Path,
) -> None:
    """Plot per-experiment crop counts classified as R vs S per 5-min bin."""
    per_exp = fold_result["per_experiment"]
    if not per_exp:
        return

    # Separate resistant and susceptible experiments
    r_exps = {k: v for k, v in per_exp.items() if v["label"] == 1}
    s_exps = {k: v for k, v in per_exp.items() if v["label"] == 0}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    for ax, exps, title in [
        (axes[0], r_exps, "True Resistant"),
        (axes[1], s_exps, "True Susceptible"),
    ]:
        for exp_id, exp_data in sorted(exps.items()):
            bins = exp_data["bin_timeseries"]
            times = [b["bin_center_min"] for b in bins if b["n_crops"] > 0]
            n_r = [
                int(b["n_crops"] * b["frac_resistant"])
                for b in bins if b["n_crops"] > 0
            ]
            n_s = [
                b["n_crops"] - int(b["n_crops"] * b["frac_resistant"])
                for b in bins if b["n_crops"] > 0
            ]

            if not times:
                continue

            ec = _extract_ec(exp_id)
            color = STRAIN_COLORS.get(ec, "#333333")

            # Plot fraction resistant as line
            frac_r = [
                b["frac_resistant"] for b in bins if b["n_crops"] > 0
            ]
            ax.plot(
                times, frac_r,
                color=color, marker="o", markersize=3, linewidth=1.5,
                alpha=0.8, label=ec,
            )

        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_xlabel("Time (minutes)", fontsize=11)
        ax.set_ylabel("Fraction classified as Resistant", fontsize=11)
        ax.set_title(f"{title} experiments", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, 65)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="best")

    fig.suptitle(
        f"Fold {fold_idx} | Crop Classification Over Time",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / f"fold{fold_idx}_crop_fractions.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)


def plot_aggregate_accuracy(
    all_fold_results: list[dict],
    output_dir: Path,
) -> None:
    """Plot aggregate experiment-level accuracy vs cumulative time."""
    time_points = [300, 600, 900, 1200, 1500, 1800, 2400, 3000, 3600]
    time_min = [t / 60 for t in time_points]

    # Collect accuracies per time point across folds
    acc_per_time = {t: [] for t in time_points}
    aurocs = []

    for result in all_fold_results:
        aurocs.append(result["experiment_auroc"])
        for t in time_points:
            if t in result["accuracy_vs_time"]:
                acc_per_time[t].append(result["accuracy_vs_time"][t])

    fig, ax = plt.subplots(figsize=(10, 6))

    # Mean and std bands
    means = []
    stds = []
    valid_times = []
    for t, t_m in zip(time_points, time_min):
        vals = acc_per_time[t]
        if vals:
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            valid_times.append(t_m)

    means = np.array(means)
    stds = np.array(stds)
    valid_times = np.array(valid_times)

    ax.plot(valid_times, means, "o-", color="#2c7bb6", linewidth=2, markersize=6)
    ax.fill_between(
        valid_times, means - stds, means + stds,
        alpha=0.2, color="#2c7bb6",
    )

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="Chance")
    ax.set_xlabel("Cumulative time (minutes)", fontsize=12)
    ax.set_ylabel("Experiment-level accuracy", fontsize=12)
    ax.set_title(
        f"Per-Crop MLP: Experiment Accuracy vs Time\n"
        f"(mean +/- std across {len(all_fold_results)} folds, "
        f"AUROC@60min: {np.mean(aurocs):.3f} +/- {np.std(aurocs):.3f})",
        fontsize=11,
    )
    ax.set_ylim(0.3, 1.05)
    ax.set_xlim(0, 65)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "aggregate_accuracy_vs_time.png", dpi=150)
    plt.close(fig)


def plot_aggregate_timeseries(
    all_fold_results: list[dict],
    output_dir: Path,
) -> None:
    """Plot aggregate mean P(R) over time, separated by true label."""
    # Collect all per-experiment timeseries across folds
    r_probs_by_bin = {}  # bin_center_min -> list of mean_prob_r
    s_probs_by_bin = {}

    for result in all_fold_results:
        for exp_data in result["per_experiment"].values():
            target = r_probs_by_bin if exp_data["label"] == 1 else s_probs_by_bin
            for b in exp_data["bin_timeseries"]:
                if b["mean_prob_r"] is not None:
                    t = b["bin_center_min"]
                    target.setdefault(t, []).append(b["mean_prob_r"])

    fig, ax = plt.subplots(figsize=(10, 6))

    for probs_by_bin, label, color in [
        (r_probs_by_bin, "True Resistant", "#d73027"),
        (s_probs_by_bin, "True Susceptible", "#4575b4"),
    ]:
        times = sorted(probs_by_bin.keys())
        means = [np.mean(probs_by_bin[t]) for t in times]
        stds = [np.std(probs_by_bin[t]) for t in times]
        means = np.array(means)
        stds = np.array(stds)

        ax.plot(times, means, "o-", color=color, linewidth=2, markersize=4, label=label)
        ax.fill_between(times, means - stds, means + stds, alpha=0.15, color=color)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_ylabel("Mean P(Resistant)", fontsize=12)
    ax.set_title(
        "Per-Crop MLP: Mean Prediction Over Time by True Label\n"
        "(aggregated across all folds)",
        fontsize=11,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 65)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "aggregate_timeseries_by_label.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-crop MLP classifier with strain-holdout CV"
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--features-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-time-sec", type=float, default=2400.0,
        help="Only train on crops after this time (default: 2400 = 40 min)",
    )
    parser.add_argument(
        "--max-crops-per-exp", type=int, default=10000,
        help="Max training crops per experiment (default: 10000)",
    )
    parser.add_argument(
        "--bin-width-sec", type=float, default=300.0,
        help="Evaluation bin width in seconds (default: 300 = 5 min)",
    )
    args = parser.parse_args()

    config = FullConfig()
    config.device = args.device
    config.seed = args.seed
    if args.data_root:
        config.paths.data_root = args.data_root
    if args.features_dir:
        config.paths.features_dir = args.features_dir

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Build strain-grouped experiments
    groups = build_strain_grouped_experiments(
        config.paths.features_dir, config.paths.data_root,
    )
    for label, strain_dict in groups.items():
        label_name = "Resistant" if label == 1 else "Susceptible"
        for ec, exps in sorted(strain_dict.items()):
            logger.info(f"  {label_name} {ec}: {len(exps)} experiments")

    # Generate folds
    folds = generate_folds(
        groups, n_holdout_per_class=2, n_folds=args.n_folds, seed=args.seed,
    )

    device = torch.device(config.device)
    all_fold_results = []

    for i, fold in enumerate(folds):
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {i+1}/{len(folds)}")
        logger.info(f"{'='*60}")

        ckpt_path = train_fold(
            i, fold,
            features_dir=config.paths.features_dir,
            ckpt_dir=ckpt_dir,
            device=device,
            min_time_sec=args.min_time_sec,
            max_per_exp=args.max_crops_per_exp,
            seed=args.seed,
        )

        result = evaluate_fold(
            i, fold, ckpt_path,
            features_dir=config.paths.features_dir,
            device=device,
            bin_width_sec=args.bin_width_sec,
        )
        all_fold_results.append(result)

        logger.info(
            f"  Fold {i} test: Accuracy={result['experiment_accuracy']:.4f}, "
            f"AUROC={result['experiment_auroc']:.4f}"
        )
        for t_sec, acc in sorted(result["accuracy_vs_time"].items()):
            logger.info(f"    @{t_sec//60:2d}min: {acc:.4f}")

        # Per-fold plots
        plot_fold_timeseries(i, result, plots_dir)
        plot_fold_crop_counts(i, result, plots_dir)

    # Aggregate plots
    plot_aggregate_accuracy(all_fold_results, plots_dir)
    plot_aggregate_timeseries(all_fold_results, plots_dir)

    # Aggregate results
    logger.info(f"\n{'='*60}")
    logger.info("AGGREGATE RESULTS (Per-Crop MLP)")
    logger.info(f"{'='*60}")

    agg_aurocs = [r["experiment_auroc"] for r in all_fold_results]
    agg_accs = [r["experiment_accuracy"] for r in all_fold_results]

    logger.info(
        f"\nExperiment AUROC: {np.mean(agg_aurocs):.4f} +/- {np.std(agg_aurocs):.4f}"
    )
    logger.info(
        f"Experiment Accuracy: {np.mean(agg_accs):.4f} +/- {np.std(agg_accs):.4f}"
    )

    time_points = [300, 600, 900, 1200, 1500, 1800, 2400, 3000, 3600]
    logger.info("\nAccuracy vs cumulative time:")
    for t in time_points:
        vals = [r["accuracy_vs_time"].get(t) for r in all_fold_results]
        vals = [v for v in vals if v is not None]
        if vals:
            logger.info(
                f"  {t//60:3d} min: {np.mean(vals):.4f} +/- {np.std(vals):.4f}"
            )

    # Save results JSON (strip per-experiment details for compact JSON,
    # keep them in a separate file)
    summary_results = []
    for r in all_fold_results:
        sr = {k: v for k, v in r.items() if k != "per_experiment"}
        sr["accuracy_vs_time"] = {
            str(k): v for k, v in sr["accuracy_vs_time"].items()
        }
        summary_results.append(sr)

    summary = {
        "model": "CropMLP",
        "training_time_threshold_sec": args.min_time_sec,
        "max_crops_per_experiment": args.max_crops_per_exp,
        "bin_width_sec": args.bin_width_sec,
        "n_folds": len(folds),
        "mean_experiment_auroc": float(np.mean(agg_aurocs)),
        "std_experiment_auroc": float(np.std(agg_aurocs)),
        "mean_experiment_accuracy": float(np.mean(agg_accs)),
        "std_experiment_accuracy": float(np.std(agg_accs)),
        "mean_accuracy_vs_time": {
            str(t): float(np.mean([
                r["accuracy_vs_time"].get(t, np.nan) for r in all_fold_results
                if t in r["accuracy_vs_time"]
            ]))
            for t in time_points
        },
        "folds": summary_results,
    }

    with open(output_dir / "strain_holdout_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save detailed per-experiment data separately
    detailed = {}
    for r in all_fold_results:
        detailed[f"fold{r['fold']}"] = r["per_experiment"]

    with open(output_dir / "per_experiment_details.json", "w") as f:
        json.dump(detailed, f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"  Summary:  {output_dir / 'strain_holdout_results.json'}")
    logger.info(f"  Details:  {output_dir / 'per_experiment_details.json'}")
    logger.info(f"  Plots:    {plots_dir}")


if __name__ == "__main__":
    main()
