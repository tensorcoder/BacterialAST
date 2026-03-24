"""Evaluate no-antibiotic control experiments through ALL trained classifiers.

Susceptible bacteria imaged WITHOUT ampicillin exposure.
Expected result if classifiers learn antibiotic-induced morphology:
  -> classified as RESISTANT (no morphological change from antibiotics)
If classifiers learn strain identity instead:
  -> classified as SUSCEPTIBLE (same strains as in Susceptible/ training set)

Runs the new experiments through all 5-fold checkpoints from every classifier
variant that has v1 features (both population-temporal and per-crop MLP).

Usage:
    PYTHONPATH=/path/to/parent python3 -m ast_classifier.scripts.eval_no_amp_control \
        --features-dir ./features \
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
from ..scripts.strain_holdout_crop_classifier import CropMLP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# All v1-feature classifier variants and their display names
VARIANTS = {
    "results_strain_holdout": "Baseline (Transformer + Stats)",
    "results_strain_holdout_no_count": "No Count (morphology only)",
    "results_strain_holdout_delta": "Delta Features",
    "results_strain_holdout_subseq": "Subsequence Sampling",
    "results_strain_holdout_attention": "Attention Bin Encoder",
    "results_strain_holdout_attn_aux": "Attention + Auxiliary Loss",
    "results_strain_holdout_stats_aux": "Stats + Pre-Transformer Aux Loss",
    "results_strain_holdout_lstm": "BiLSTM Temporal",
    "results_strain_holdout_ctx_aux": "Contextualized Auxiliary",
    "results_crop_mlp": "Per-Crop MLP",
}

STRAIN_COLORS = {
    "EC33": "#d95f02", "EC36": "#7570b3", "EC39": "#e7298a",
}

EVAL_TIMES = [60, 120, 180, 300, 600, 900, 1800, 3600]


def _extract_ec(experiment_id: str) -> str:
    m = re.match(r"^(EC\d+)", experiment_id, re.IGNORECASE)
    return m.group(1).upper() if m else experiment_id[:10]


# ---------------------------------------------------------------------------
# Experiment discovery
# ---------------------------------------------------------------------------

def discover_no_amp_experiments(
    features_dir: Path,
    data_root: Path,
) -> list[ExperimentMeta]:
    no_amp_dir = data_root / "Susceptible_no_amp"
    if not no_amp_dir.exists():
        raise FileNotFoundError(f"No Susceptible_no_amp directory at {no_amp_dir}")

    experiments = []
    for exp_dir in sorted(no_amp_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        feat_path = features_dir / f"{exp_dir.name}.npz"
        if not feat_path.exists():
            logger.warning(f"No features for {exp_dir.name}, skipping")
            continue
        experiments.append(ExperimentMeta(
            experiment_id=exp_dir.name,
            label=0,  # true label: susceptible strain
            features_path=feat_path,
        ))

    logger.info(f"Found {len(experiments)} no-amp control experiments")
    for exp in experiments:
        logger.info(f"  {exp.experiment_id} ({_extract_ec(exp.experiment_id)})")
    return experiments


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_for_variant(
    variant: str,
    ckpt_path: Path,
    config: FullConfig,
    device: torch.device,
) -> torch.nn.Module:
    """Load the appropriate model class for a variant."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "lstm" in variant:
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

    if "ctx_aux" in variant:
        cfg = config.classifier
        model = ContextualAuxClassifier(
            feature_dim=cfg.feature_dim,
            temporal_hidden_dim=cfg.temporal_hidden_dim,
            temporal_num_layers=cfg.temporal_num_layers,
            temporal_num_heads=cfg.temporal_num_heads,
            temporal_ffn_dim=cfg.temporal_ffn_dim,
            classifier_hidden_dim=cfg.classifier_hidden_dim,
            num_classes=2,
            dropout=cfg.dropout,
            max_count_normalizer=float(cfg.max_crops_per_bin),
            use_delta_features=cfg.use_delta_features,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        return model

    if "crop_mlp" in variant:
        model = CropMLP(in_dim=384).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        return model

    # PopulationTemporalClassifier variants
    cfg = config.classifier
    include_count = "no_count" not in variant
    use_delta = "delta" in variant
    bin_encoder = "attention" if "attention" in variant else "stats"

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
        use_delta_features=use_delta,
        bin_encoder_type=bin_encoder,
        bin_attn_heads=cfg.bin_attn_heads,
        include_count=include_count,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    return model


# ---------------------------------------------------------------------------
# Inference: population-temporal models
# ---------------------------------------------------------------------------

def run_temporal_inference(
    experiments: list[ExperimentMeta],
    model: torch.nn.Module,
    config: FullConfig,
    device: torch.device,
) -> dict[str, dict[int, float]]:
    """Run a population-temporal model at multiple time windows.

    Returns {experiment_id: {window_sec: prob_resistant}}.
    """
    cfg = config.classifier
    model.eval()
    results = {}

    for exp in experiments:
        exp_results = {}
        for window_sec in EVAL_TIMES:
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
                        for k in ["bin_features", "bin_mask", "crop_mask",
                                   "bin_times", "bin_counts", "time_fraction"]
                    }
                    with torch.amp.autocast("cuda"):
                        output = model(batch_gpu)
                    prob_r = F.softmax(output["logits"].float(), dim=-1)[0, 1].item()
            exp_results[window_sec] = prob_r
        results[exp.experiment_id] = exp_results

    return results


# ---------------------------------------------------------------------------
# Inference: per-crop MLP
# ---------------------------------------------------------------------------

def run_crop_mlp_inference(
    experiments: list[ExperimentMeta],
    model: torch.nn.Module,
    features_dir: Path,
    device: torch.device,
    bin_width_sec: float = 300.0,
) -> dict[str, dict[int, float]]:
    """Run per-crop MLP and aggregate to P(Resistant) at each time window.

    Returns {experiment_id: {window_sec: prob_resistant}}.
    """
    model.eval()
    results = {}

    for exp in experiments:
        npz_path = features_dir / f"{exp.experiment_id}.npz"
        if not npz_path.exists():
            continue

        data = np.load(npz_path)
        features = data["features"].astype(np.float32)
        timestamps = data["timestamps"].astype(np.float64)
        rel_ts = (timestamps - timestamps.min()).astype(np.float32)

        # Run inference on all crops
        features_t = torch.from_numpy(features).to(device)
        all_probs = []
        with torch.no_grad():
            for start in range(0, len(features_t), 8192):
                chunk = features_t[start:start + 8192]
                logits = model(chunk)
                probs = F.softmax(logits.float(), dim=-1)[:, 1]
                all_probs.append(probs.cpu().numpy())
        crop_probs = np.concatenate(all_probs)

        # Cumulative P(Resistant) at each time window
        exp_results = {}
        for window_sec in EVAL_TIMES:
            mask = rel_ts <= window_sec
            if np.sum(mask) == 0:
                exp_results[window_sec] = 0.5
                continue
            exp_results[window_sec] = float(np.mean(crop_probs[mask]))

        results[exp.experiment_id] = exp_results

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_variant_trajectories(
    variant: str,
    display_name: str,
    all_fold_results: dict[str, dict[str, dict[int, float]]],
    experiments: list[ExperimentMeta],
    output_dir: Path,
) -> None:
    """Plot P(Resistant) over time for one variant, all folds."""
    fig, ax = plt.subplots(figsize=(10, 6))

    exp_ids = sorted(set(
        eid for fold_res in all_fold_results.values() for eid in fold_res
    ))

    for exp_id in exp_ids:
        fold_trajs = [
            fold_res[exp_id]
            for fold_res in all_fold_results.values()
            if exp_id in fold_res
        ]
        if not fold_trajs:
            continue

        times_min = [t / 60 for t in EVAL_TIMES]
        mean_probs = []
        std_probs = []
        for t in EVAL_TIMES:
            probs = [ft[t] for ft in fold_trajs if t in ft]
            mean_probs.append(np.mean(probs) if probs else np.nan)
            std_probs.append(np.std(probs) if probs else 0)

        ec = _extract_ec(exp_id)
        color = STRAIN_COLORS.get(ec, "#333333")

        mean_arr = np.array(mean_probs)
        std_arr = np.array(std_probs)

        ax.plot(times_min, mean_arr, "o-", color=color, linewidth=2,
                markersize=5, alpha=0.8, label=exp_id)
        ax.fill_between(times_min, mean_arr - std_arr, mean_arr + std_arr,
                         alpha=0.1, color=color)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.axhspan(0.5, 1.0, alpha=0.04, color="red")
    ax.axhspan(0.0, 0.5, alpha=0.04, color="blue")
    ax.text(62, 0.75, "Resistant\n(expected if\nmorphology-based)",
            fontsize=8, ha="right", color="red", alpha=0.6)
    ax.text(62, 0.25, "Susceptible\n(expected if\nstrain-based)",
            fontsize=8, ha="right", color="blue", alpha=0.6)

    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_ylabel("P(Resistant)", fontsize=12)
    ax.set_title(f"No-Antibiotic Control — {display_name}\n(mean +/- std across folds)",
                 fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 65)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    safe_name = variant.replace("/", "_")
    fig.savefig(output_dir / f"no_amp_{safe_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all_variants_comparison(
    all_variant_results: dict[str, dict[str, dict[str, dict[int, float]]]],
    output_dir: Path,
) -> None:
    """Single comparison plot: mean P(R) over time for each variant."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    times_min = [t / 60 for t in EVAL_TIMES]

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_variant_results)))

    for i, (variant, fold_results) in enumerate(all_variant_results.items()):
        display = VARIANTS.get(variant, variant)

        # Mean across all folds and experiments at each time
        mean_per_time = []
        std_per_time = []
        for t in EVAL_TIMES:
            all_probs = []
            for fold_res in fold_results.values():
                for exp_res in fold_res.values():
                    if t in exp_res:
                        all_probs.append(exp_res[t])
            mean_per_time.append(np.mean(all_probs) if all_probs else np.nan)
            std_per_time.append(np.std(all_probs) if all_probs else 0)

        ax1.plot(times_min, mean_per_time, "o-", color=colors[i], linewidth=2,
                 markersize=5, label=display)
        ax1.fill_between(
            times_min,
            [m - s for m, s in zip(mean_per_time, std_per_time)],
            [m + s for m, s in zip(mean_per_time, std_per_time)],
            alpha=0.08, color=colors[i],
        )

    ax1.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax1.axhspan(0.5, 1.0, alpha=0.03, color="red")
    ax1.axhspan(0.0, 0.5, alpha=0.03, color="blue")
    ax1.set_xlabel("Time (minutes)", fontsize=12)
    ax1.set_ylabel("Mean P(Resistant)", fontsize=12)
    ax1.set_title("No-Antibiotic Control: All Classifiers\n"
                   "P(R) > 0.5 = morphology-based, P(R) < 0.5 = strain-based",
                   fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(0, 65)
    ax1.legend(fontsize=7, loc="best")
    ax1.grid(True, alpha=0.3)

    # Right: bar chart of mean P(R) @ 60 min per variant
    names = []
    means_60 = []
    stds_60 = []
    bar_colors = []
    for i, (variant, fold_results) in enumerate(all_variant_results.items()):
        probs_60 = []
        for fold_res in fold_results.values():
            for exp_res in fold_res.values():
                if 3600 in exp_res:
                    probs_60.append(exp_res[3600])
        names.append(VARIANTS.get(variant, variant).replace(" ", "\n"))
        means_60.append(np.mean(probs_60) if probs_60 else 0.5)
        stds_60.append(np.std(probs_60) if probs_60 else 0)
        bar_colors.append(colors[i])

    x = np.arange(len(names))
    bars = ax2.bar(x, means_60, yerr=stds_60, capsize=4,
                   color=bar_colors, alpha=0.8)
    ax2.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    for bar, val, std in zip(bars, means_60, stds_60):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax2.set_ylabel("Mean P(Resistant) @ 60 min", fontsize=12)
    ax2.set_title("All Classifiers: P(R) at 60 min\n(no-antibiotic controls)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=7, ha="center")
    ax2.set_ylim(0, 1.15)
    ax2.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "no_amp_all_classifiers_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate no-antibiotic control experiments through all classifiers"
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--features-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("./results_no_amp_control"))
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    config = FullConfig()
    config.device = args.device
    if args.data_root:
        config.paths.data_root = args.data_root
    if args.features_dir:
        config.paths.features_dir = args.features_dir

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    device = torch.device(args.device)

    # Discover no-amp experiments
    experiments = discover_no_amp_experiments(
        Path(config.paths.features_dir), Path(config.paths.data_root)
    )
    if not experiments:
        logger.error("No no-amp experiments found with features.")
        return

    all_variant_results: dict[str, dict] = {}

    for variant, display_name in VARIANTS.items():
        variant_dir = Path(variant)
        ckpt_dir = variant_dir / "checkpoints"

        if not ckpt_dir.exists():
            logger.warning(f"Skipping {variant}: no checkpoints directory")
            continue

        fold_ckpts = [ckpt_dir / f"fold{i}_best.pt" for i in range(5)]
        if not all(p.exists() for p in fold_ckpts):
            logger.warning(f"Skipping {variant}: missing fold checkpoints")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"VARIANT: {display_name} ({variant})")
        logger.info(f"{'='*60}")

        is_crop_mlp = "crop_mlp" in variant
        fold_results: dict[str, dict[str, dict[int, float]]] = {}

        for fold_idx in range(5):
            ckpt_path = fold_ckpts[fold_idx]
            model = load_model_for_variant(variant, ckpt_path, config, device)

            if is_crop_mlp:
                preds = run_crop_mlp_inference(
                    experiments, model, Path(config.paths.features_dir), device
                )
            else:
                preds = run_temporal_inference(
                    experiments, model, config, device
                )

            fold_results[f"fold{fold_idx}"] = preds

            for exp_id, exp_res in preds.items():
                prob_60 = exp_res.get(3600, -1)
                logger.info(f"  Fold {fold_idx} | {exp_id}: P(R)@60min = {prob_60:.3f}")

            del model
            torch.cuda.empty_cache()

        all_variant_results[variant] = fold_results

        # Per-variant plot
        plot_variant_trajectories(
            variant, display_name, fold_results, experiments, plots_dir
        )

    # Comparison plot
    if all_variant_results:
        plot_all_variants_comparison(all_variant_results, plots_dir)

    # Summary log
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY: No-Antibiotic Control Results")
    logger.info(f"{'='*60}")

    for variant, fold_results in all_variant_results.items():
        display = VARIANTS.get(variant, variant)
        all_probs_60 = []
        for fold_res in fold_results.values():
            for exp_res in fold_res.values():
                if 3600 in exp_res:
                    all_probs_60.append(exp_res[3600])
        if all_probs_60:
            mean_p = np.mean(all_probs_60)
            std_p = np.std(all_probs_60)
            frac_r = np.mean(np.array(all_probs_60) > 0.5)
            interpretation = "MORPHOLOGY-BASED" if mean_p > 0.5 else "STRAIN-BASED"
            logger.info(
                f"  {display:40s}  P(R)@60min: {mean_p:.3f} +/- {std_p:.3f}  "
                f"({frac_r:.0%} classified R)  -> {interpretation}"
            )

    # Save JSON
    save_data = {
        "description": (
            "No-antibiotic control: susceptible bacteria without ampicillin. "
            "If classifier is morphology-based, these should be classified as resistant "
            "(no drug-induced changes). If strain-based, classified as susceptible."
        ),
        "experiments": [
            {"id": e.experiment_id, "strain": _extract_ec(e.experiment_id),
             "true_label": "susceptible"}
            for e in experiments
        ],
        "variants": {
            variant: {
                "display_name": VARIANTS.get(variant, variant),
                "folds": {
                    fold_key: {
                        exp_id: {str(k): round(v, 4) for k, v in exp_res.items()}
                        for exp_id, exp_res in fold_res.items()
                    }
                    for fold_key, fold_res in fold_results.items()
                },
            }
            for variant, fold_results in all_variant_results.items()
        },
    }

    with open(output_dir / "no_amp_control_results.json", "w") as f:
        json.dump(save_data, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"  JSON:  {output_dir / 'no_amp_control_results.json'}")
    logger.info(f"  Plots: {plots_dir}")


if __name__ == "__main__":
    main()
