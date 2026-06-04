"""Generate per-fold time-series and per-bin plots for the
Baseline (Transformer + Stats) model, matching the style of
``results_crop_mlp/plots/``.

For each test experiment:
- Per-bin (non-overlapping 5-min slices) inference -> P(Resistant) per slice
- Cumulative inference at multiple time windows -> experiment accuracy vs time

Reuses the fold split from strain_holdout_eval (seed=42), same as the
checkpoints in ``results_strain_holdout/checkpoints/``.

Usage::

    PYTHONPATH=/home/mkedz/code python3 -m ast_classifier.scripts.plot_baseline_per_bin \
        --results-dir results_strain_holdout \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from ..config import FullConfig
from ..data.dataset import ExperimentMeta
from ..models.classifier import PopulationTemporalClassifier
from .strain_holdout_eval import build_strain_grouped_experiments, generate_folds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _build_inputs(
    features: np.ndarray,
    rel_ts: np.ndarray,
    start_sec: float,
    end_sec: float,
    time_bin_width_sec: float,
    max_crops_per_bin: int,
    feature_dim: int,
    max_experiment_sec: float,
    device: torch.device,
    rng: np.random.RandomState,
) -> tuple[dict | None, int]:
    """Construct a single-batch model input for crops in [start_sec, end_sec].

    Mirrors ``PopulationTemporalDataset.__getitem__`` but supports an
    arbitrary [start, end] time slice (not only cumulative windows from 0).

    Returns:
        (batch_dict_on_device, n_crops_in_slice).  Returns (None, 0) if no
        crops fall in the slice.
    """
    sel = (rel_ts >= start_sec) & (rel_ts <= end_sec)
    sel_features = features[sel].astype(np.float32)
    sel_ts = rel_ts[sel]
    if len(sel_features) == 0:
        return None, 0

    duration = max(end_sec - start_sec, time_bin_width_sec)
    n_bins = max(1, int(np.ceil(duration / time_bin_width_sec)))

    bin_features = torch.zeros(1, n_bins, max_crops_per_bin, feature_dim)
    bin_mask = torch.zeros(1, n_bins, dtype=torch.bool)
    crop_mask = torch.zeros(1, n_bins, max_crops_per_bin, dtype=torch.bool)
    bin_times = torch.zeros(1, n_bins, dtype=torch.float32)
    bin_counts = torch.zeros(1, n_bins, dtype=torch.float32)

    for b in range(n_bins):
        t_lo = start_sec + b * time_bin_width_sec
        t_hi = start_sec + (b + 1) * time_bin_width_sec
        bin_times[0, b] = (t_lo + t_hi) / 2.0
        in_bin = (sel_ts >= t_lo) & (sel_ts < t_hi)
        bin_feats = sel_features[in_bin]
        if len(bin_feats) == 0:
            continue
        bin_mask[0, b] = True
        bin_counts[0, b] = len(bin_feats)
        if len(bin_feats) > max_crops_per_bin:
            idx = rng.choice(len(bin_feats), max_crops_per_bin, replace=False)
            bin_feats = bin_feats[idx]
        n = len(bin_feats)
        bin_features[0, b, :n] = torch.from_numpy(bin_feats)
        crop_mask[0, b, :n] = True

    time_fraction = min(end_sec / max_experiment_sec, 1.0)

    batch = {
        "bin_features": bin_features.to(device, non_blocking=True),
        "bin_mask": bin_mask.to(device, non_blocking=True),
        "crop_mask": crop_mask.to(device, non_blocking=True),
        "bin_times": bin_times.to(device, non_blocking=True),
        "bin_counts": bin_counts.to(device, non_blocking=True),
        "time_fraction": torch.tensor(
            [time_fraction], dtype=torch.float32, device=device
        ),
    }
    return batch, int(len(sel_features))


def _model_prob_resistant(
    model: torch.nn.Module,
    batch: dict,
) -> float:
    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            out = model(batch)
        prob = F.softmax(out["logits"].float(), dim=-1)[0, 1].item()
    return float(prob)


def _load_features_for_experiment(
    exp: ExperimentMeta,
    features_dir: Path,
) -> tuple[np.ndarray, np.ndarray] | None:
    npz = features_dir / f"{exp.experiment_id}.npz"
    if not npz.exists():
        npz = exp.features_path
    if not npz.exists():
        return None
    data = np.load(npz)
    features = data["features"].astype(np.float32)
    timestamps = data["timestamps"].astype(np.float64)
    rel_ts = (timestamps - timestamps.min()).astype(np.float32)
    return features, rel_ts


def _load_baseline_model(
    ckpt_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, object]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
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
        include_count=getattr(cfg, "include_count", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Per-fold evaluation
# ---------------------------------------------------------------------------

def _smooth_series(values: list[float | None], width: int) -> list[float | None]:
    """Centered moving average over a list with possible None entries.

    None entries pass through. Width is the full window size (odd recommended).
    """
    if width <= 1:
        return values
    half = width // 2
    out: list[float | None] = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        valid = [v for v in values[lo:hi] if v is not None]
        if values[i] is None or not valid:
            out.append(values[i])
        else:
            out.append(float(np.mean(valid)))
    return out


def _mc_prob(
    model: torch.nn.Module,
    features: np.ndarray,
    rel_ts: np.ndarray,
    start_sec: float,
    end_sec: float,
    cfg,
    max_experiment_sec: float,
    device: torch.device,
    base_rng: np.random.RandomState,
    n_samples: int,
) -> tuple[float | None, int]:
    """Monte-Carlo P(R) average across n_samples random crop subsamples."""
    probs = []
    last_n = 0
    for _ in range(max(1, n_samples)):
        sub_rng = np.random.RandomState(base_rng.randint(0, 2**31 - 1))
        batch, n_crops = _build_inputs(
            features=features,
            rel_ts=rel_ts,
            start_sec=start_sec,
            end_sec=end_sec,
            time_bin_width_sec=cfg.time_bin_width_sec,
            max_crops_per_bin=cfg.max_crops_per_bin,
            feature_dim=cfg.feature_dim,
            max_experiment_sec=max_experiment_sec,
            device=device,
            rng=sub_rng,
        )
        if batch is None:
            return None, 0
        last_n = n_crops
        probs.append(_model_prob_resistant(model, batch))
    return float(np.mean(probs)), last_n


def evaluate_fold(
    fold_idx: int,
    fold: dict,
    ckpt_path: Path,
    features_dir: Path,
    device: torch.device,
    window_sec: float = 300.0,
    stride_sec: float | None = None,
    smoothing_bins: int = 1,
    mc_samples: int = 1,
    cumulative_times_sec: tuple[int, ...] = (
        300, 600, 900, 1200, 1500, 1800, 2400, 3000, 3600,
    ),
    max_experiment_sec: float = 3600.0,
    seed: int = 42,
) -> dict:
    """Per-bin (sliding) and cumulative inference for one fold.

    If ``stride_sec`` equals ``window_sec`` (or is None), bins are
    non-overlapping (legacy behaviour). Smaller stride values produce
    overlapping sliding windows, which smooths the per-bin time-series.
    """
    model, cfg = _load_baseline_model(ckpt_path, device)
    rng = np.random.RandomState(seed)

    if stride_sec is None or stride_sec <= 0:
        stride_sec = window_sec

    test_exps = fold["test"]
    per_experiment: dict[str, dict] = {}

    for exp in test_exps:
        loaded = _load_features_for_experiment(exp, features_dir)
        if loaded is None:
            logger.warning(f"  No features for {exp.experiment_id}")
            continue
        features, rel_ts = loaded
        max_time = float(rel_ts.max())

        # Sliding windows of width window_sec stepping by stride_sec
        bin_data: list[dict] = []
        # Place bin centers at half-window inside the experiment so we don't
        # extrapolate beyond the data.  Last center is at max_time - window/2.
        first_center = window_sec / 2.0
        last_center = max(first_center, max_time - window_sec / 2.0)
        n_steps = max(1, int(np.floor((last_center - first_center) / stride_sec)) + 1)

        for s in range(n_steps):
            center = first_center + s * stride_sec
            t_lo = center - window_sec / 2.0
            t_hi = center + window_sec / 2.0
            prob_r, n_crops = _mc_prob(
                model=model,
                features=features,
                rel_ts=rel_ts,
                start_sec=t_lo,
                end_sec=t_hi,
                cfg=cfg,
                max_experiment_sec=max_experiment_sec,
                device=device,
                base_rng=rng,
                n_samples=mc_samples,
            )
            if prob_r is None:
                bin_data.append({
                    "bin_center_min": center / 60,
                    "n_crops": 0,
                    "frac_resistant": None,
                    "mean_prob_r": None,
                })
                continue
            bin_data.append({
                "bin_center_min": center / 60,
                "n_crops": n_crops,
                "frac_resistant": prob_r,
                "mean_prob_r": prob_r,
            })

        # Optional centered moving-average smoothing over the time series
        if smoothing_bins > 1 and len(bin_data) > 1:
            raw_probs = [b["mean_prob_r"] for b in bin_data]
            smoothed = _smooth_series(raw_probs, smoothing_bins)
            for b, s_val in zip(bin_data, smoothed):
                b["mean_prob_r"] = s_val
                b["frac_resistant"] = s_val

        # Cumulative windows from t=0
        cumulative_preds: dict[str, dict] = {}
        for t_sec in cumulative_times_sec:
            batch, n_crops = _build_inputs(
                features=features,
                rel_ts=rel_ts,
                start_sec=0.0,
                end_sec=float(t_sec),
                time_bin_width_sec=cfg.time_bin_width_sec,
                max_crops_per_bin=cfg.max_crops_per_bin,
                feature_dim=cfg.feature_dim,
                max_experiment_sec=max_experiment_sec,
                device=device,
                rng=rng,
            )
            if batch is None:
                continue
            prob_r = _model_prob_resistant(model, batch)
            cumulative_preds[str(t_sec)] = {
                "prob_r": prob_r,
                "pred": int(prob_r > 0.5),
                "correct": int((prob_r > 0.5) == exp.label),
                "n_crops": n_crops,
            }

        # Experiment-level prediction at the longest cumulative window with data
        last_key = max(cumulative_preds.keys(), key=int) if cumulative_preds else None
        exp_prob_r = (
            cumulative_preds[last_key]["prob_r"] if last_key is not None else 0.5
        )
        exp_pred = int(exp_prob_r > 0.5)

        per_experiment[exp.experiment_id] = {
            "label": int(exp.label),
            "label_name": "R" if exp.label == 1 else "S",
            "exp_prob_r": float(exp_prob_r),
            "exp_pred": exp_pred,
            "correct": int(exp_pred == exp.label),
            "n_total_crops": int(len(features)),
            "max_time_min": float(max_time / 60),
            "bin_timeseries": bin_data,
            "cumulative_preds": cumulative_preds,
        }

    # Aggregate experiment-level metrics
    labels = np.array([v["label"] for v in per_experiment.values()])
    probs = np.array([v["exp_prob_r"] for v in per_experiment.values()])
    preds = (probs > 0.5).astype(int)
    acc = float(np.mean(preds == labels)) if len(labels) else 0.0
    try:
        auroc = float(roc_auc_score(labels, probs))
    except ValueError:
        auroc = 0.5

    accuracy_vs_time: dict[int, float] = {}
    for t_sec in cumulative_times_sec:
        correct = []
        for v in per_experiment.values():
            cp = v["cumulative_preds"].get(str(t_sec))
            if cp is not None:
                correct.append(cp["correct"])
        if correct:
            accuracy_vs_time[t_sec] = float(np.mean(correct))

    # Free model memory before next fold
    del model
    torch.cuda.empty_cache()

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
# Plotting (mirrors strain_holdout_crop_classifier.py style)
# ---------------------------------------------------------------------------

# Same palette as crop_mlp's plots, so direct comparison is easy
STRAIN_COLORS = {
    "EC35": "#e41a1c", "EC40": "#377eb8", "EC48": "#4daf4a",
    "EC58": "#984ea3", "EC60": "#ff7f00", "EC65": "#a65628",
    "EC87": "#f781bf", "EC126": "#1b9e77", "EC33": "#d95f02",
    "EC36": "#7570b3", "EC39": "#e7298a", "EC42": "#66a61e",
    "EC67": "#e6ab02", "EC79": "#a6761d", "EC89": "#666666",
}


def _extract_ec(experiment_id: str) -> str:
    m = re.match(r"^(EC\d+)", experiment_id, re.IGNORECASE)
    return m.group(1).upper() if m else experiment_id[:6]


def plot_fold_timeseries(
    fold_idx: int,
    fold_result: dict,
    output_dir: Path,
) -> None:
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
    ax.set_ylabel("P(Resistant) per window", fontsize=12)
    ax.set_title(
        f"Fold {fold_idx} | Holdout R: {fold_result['holdout_r']}, "
        f"S: {fold_result['holdout_s']} | "
        f"AUROC: {fold_result['experiment_auroc']:.3f}",
        fontsize=11,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 65)

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
    """R/S split panel of per-bin P(Resistant)."""
    per_exp = fold_result["per_experiment"]
    if not per_exp:
        return

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
            frac_r = [b["frac_resistant"] for b in bins if b["n_crops"] > 0]
            if not times:
                continue

            ec = _extract_ec(exp_id)
            color = STRAIN_COLORS.get(ec, "#333333")
            ax.plot(
                times, frac_r,
                color=color, marker="o", markersize=3, linewidth=1.5,
                alpha=0.8, label=ec,
            )

        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_xlabel("Time (minutes)", fontsize=11)
        ax.set_ylabel("P(Resistant) per window", fontsize=11)
        ax.set_title(f"{title} experiments", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, 65)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="best")

    fig.suptitle(
        f"Fold {fold_idx} | Per-Slice P(Resistant) Over Time",
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
    cumulative_times_sec: tuple[int, ...],
) -> None:
    time_min = [t / 60 for t in cumulative_times_sec]
    acc_per_time: dict[int, list[float]] = {t: [] for t in cumulative_times_sec}
    aurocs = []
    for result in all_fold_results:
        aurocs.append(result["experiment_auroc"])
        for t in cumulative_times_sec:
            if t in result["accuracy_vs_time"]:
                acc_per_time[t].append(result["accuracy_vs_time"][t])

    fig, ax = plt.subplots(figsize=(10, 6))

    valid_times, means, stds = [], [], []
    for t, t_m in zip(cumulative_times_sec, time_min):
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
        f"Baseline (Transformer + Stats): Experiment Accuracy vs Time\n"
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
    r_probs_by_bin: dict[float, list[float]] = {}
    s_probs_by_bin: dict[float, list[float]] = {}

    for result in all_fold_results:
        for exp_data in result["per_experiment"].values():
            target = r_probs_by_bin if exp_data["label"] == 1 else s_probs_by_bin
            for b in exp_data["bin_timeseries"]:
                if b["mean_prob_r"] is not None:
                    target.setdefault(b["bin_center_min"], []).append(b["mean_prob_r"])

    fig, ax = plt.subplots(figsize=(10, 6))

    for probs_by_bin, label, color in [
        (r_probs_by_bin, "True Resistant", "#d73027"),
        (s_probs_by_bin, "True Susceptible", "#4575b4"),
    ]:
        times = sorted(probs_by_bin.keys())
        means = np.array([np.mean(probs_by_bin[t]) for t in times])
        stds = np.array([np.std(probs_by_bin[t]) for t in times])
        ax.plot(times, means, "o-", color=color, linewidth=2, markersize=4, label=label)
        ax.fill_between(times, means - stds, means + stds, alpha=0.15, color=color)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_ylabel("P(Resistant) per window", fontsize=12)
    ax.set_title(
        "Baseline (Transformer + Stats): Mean Prediction Over Time by True Label\n"
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
        description="Per-bin plots for Baseline (Transformer + Stats)"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results_strain_holdout"),
        help="Directory with checkpoints/ and where plots/ will be written",
    )
    parser.add_argument("--features-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--window-sec", type=float, default=300.0,
        help="Sliding window width (default: 300s = 5 min)",
    )
    parser.add_argument(
        "--stride-sec", type=float, default=None,
        help="Sliding window stride. Defaults to --window-sec (no overlap). "
             "Set lower than --window-sec for overlapping windows.",
    )
    parser.add_argument(
        "--output-subdir", type=str, default="plots",
        help="Subdirectory of --results-dir to write plots into (default: plots)",
    )
    parser.add_argument(
        "--smoothing-bins", type=int, default=1,
        help="Centered moving-average width over the per-window time-series. "
             "1 disables smoothing. With stride=60s, smoothing-bins=11 gives "
             "an effective ~10-min smooth on top of the per-window inference.",
    )
    parser.add_argument(
        "--mc-samples", type=int, default=1,
        help="Monte-Carlo samples per window: re-runs inference with different "
             "random crop subsamples and averages the P(R). 1 disables.",
    )
    args = parser.parse_args()

    config = FullConfig()
    config.device = args.device
    if args.features_dir:
        config.paths.features_dir = args.features_dir
    if args.data_root:
        config.paths.data_root = args.data_root

    results_dir = Path(args.results_dir)
    ckpt_dir = results_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoints not found at {ckpt_dir}")

    plots_dir = results_dir / args.output_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # Same fold split as training (seed=42)
    groups = build_strain_grouped_experiments(
        config.paths.features_dir, config.paths.data_root,
    )
    folds = generate_folds(
        groups, n_holdout_per_class=2, n_folds=args.n_folds, seed=args.seed,
    )

    cumulative_times = (300, 600, 900, 1200, 1500, 1800, 2400, 3000, 3600)

    all_fold_results: list[dict] = []
    for i, fold in enumerate(folds):
        ckpt_path = ckpt_dir / f"fold{i}_best.pt"
        if not ckpt_path.exists():
            logger.warning(f"Skipping fold {i}: missing {ckpt_path}")
            continue

        logger.info(f"\n{'='*60}\nFOLD {i+1}/{len(folds)}\n{'='*60}")
        logger.info(
            f"  Holdout R: {fold['holdout_r']}, S: {fold['holdout_s']} | "
            f"{len(fold['test'])} test experiments"
        )

        result = evaluate_fold(
            i, fold, ckpt_path,
            features_dir=config.paths.features_dir,
            device=device,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
            smoothing_bins=args.smoothing_bins,
            mc_samples=args.mc_samples,
            cumulative_times_sec=cumulative_times,
            seed=args.seed,
        )
        all_fold_results.append(result)

        logger.info(
            f"  Test: Accuracy={result['experiment_accuracy']:.4f}, "
            f"AUROC={result['experiment_auroc']:.4f}"
        )
        for t_sec in cumulative_times:
            acc = result["accuracy_vs_time"].get(t_sec)
            if acc is not None:
                logger.info(f"    @{t_sec//60:2d}min: {acc:.4f}")

        plot_fold_timeseries(i, result, plots_dir)
        plot_fold_crop_counts(i, result, plots_dir)

    # Aggregate plots
    plot_aggregate_accuracy(all_fold_results, plots_dir, cumulative_times)
    plot_aggregate_timeseries(all_fold_results, plots_dir)

    # Save per-experiment details
    detailed = {f"fold{r['fold']}": r["per_experiment"] for r in all_fold_results}
    with open(results_dir / "per_experiment_details.json", "w") as f:
        json.dump(detailed, f, indent=2, default=str)
    logger.info(f"\nDetails: {results_dir / 'per_experiment_details.json'}")

    # Update summary results with new times
    summary = {
        "model": "PopulationTemporalClassifier (Baseline Transformer + Stats)",
        "n_folds": len(all_fold_results),
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec if args.stride_sec is not None else args.window_sec,
        "smoothing_bins": args.smoothing_bins,
        "mc_samples": args.mc_samples,
        "mean_experiment_auroc": float(
            np.mean([r["experiment_auroc"] for r in all_fold_results])
        ),
        "std_experiment_auroc": float(
            np.std([r["experiment_auroc"] for r in all_fold_results])
        ),
        "mean_experiment_accuracy": float(
            np.mean([r["experiment_accuracy"] for r in all_fold_results])
        ),
        "std_experiment_accuracy": float(
            np.std([r["experiment_accuracy"] for r in all_fold_results])
        ),
        "mean_accuracy_vs_time": {
            str(t): float(np.mean([
                r["accuracy_vs_time"].get(t) for r in all_fold_results
                if t in r["accuracy_vs_time"]
            ]))
            for t in cumulative_times
            if any(t in r["accuracy_vs_time"] for r in all_fold_results)
        },
        "folds": [
            {k: v for k, v in r.items() if k != "per_experiment"}
            for r in all_fold_results
        ],
    }
    with open(results_dir / "per_bin_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Summary: {results_dir / 'per_bin_summary.json'}")
    logger.info(f"Plots:   {plots_dir}")


if __name__ == "__main__":
    main()
