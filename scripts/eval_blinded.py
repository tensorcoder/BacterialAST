"""Evaluate blinded test experiments using trained CropMLP checkpoints.

Runs the full pipeline:
  1. YOLO preprocessing (if not already done)
  2. DINO feature extraction (if not already done)
  3. CropMLP inference with all 5-fold checkpoints
  4. Plot generation (same style as plots/ and plots_no_amp/)

Usage:
    PYTHONPATH=/home/mkedz/code python3 -m ast_classifier.scripts.eval_blinded \
        --blinded-dir /mnt/f/Data_second_protocol/BlindedTest \
        --output-dir ./results_crop_mlp/plots_blinded \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from ..config import FullConfig
from ..data.preprocessing import extract_experiment
from ..training.extract_features import extract_features_for_experiment
from ..models.backbone import ViTSmall
from .strain_holdout_crop_classifier import CropMLP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Colors for blinded experiments
BLINDED_COLORS = {
    "B1": "#e41a1c",
    "B2": "#377eb8",
    "B3": "#4daf4a",
    "B1_part2": "#984ea3",
}


def _extract_blinded_id(experiment_id: str) -> str:
    """Extract short blinded ID like B1, B2, B3."""
    if "part2" in experiment_id.lower():
        return "B1_part2"
    for prefix in ["B1", "B2", "B3"]:
        if experiment_id.startswith(prefix):
            return prefix
    return experiment_id[:4]


def discover_blinded_experiments(blinded_dir: Path) -> list[tuple[str, Path]]:
    """Find all blinded experiment directories.

    Returns list of (experiment_id, images_dir).
    """
    experiments = []
    for exp_dir in sorted(blinded_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        images_dir = exp_dir / "images"
        if not images_dir.exists():
            logger.warning(f"No images dir in {exp_dir}")
            continue
        experiments.append((exp_dir.name, images_dir))
    return experiments


def preprocess_blinded(
    experiments: list[tuple[str, Path]],
    preprocessed_dir: Path,
    config: FullConfig,
    device: str,
    crop_size: int = 128,
    border_mode: int = cv2.BORDER_REFLECT_101,
) -> list[Path]:
    """Run YOLO preprocessing on blinded experiments.

    Uses 128x128 crops with reflection padding to match DINO v1 training.
    Returns HDF5 paths.
    """
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    h5_paths = []

    for exp_id, images_dir in experiments:
        h5_path = preprocessed_dir / f"{exp_id}.h5"
        if h5_path.exists():
            logger.info(f"Skipping preprocessing for {exp_id} (already done)")
            h5_paths.append(h5_path)
            continue

        logger.info(f"Preprocessing {exp_id} (crop_size={crop_size}, reflection padding)...")
        try:
            h5_path = extract_experiment(
                image_dir=images_dir,
                output_dir=preprocessed_dir,
                model_path=str(config.paths.yolo_weights),
                batch_size=config.preprocessing.yolo_batch_size,
                crop_size=crop_size,
                conf_threshold=config.preprocessing.yolo_confidence,
                focused_class_name=config.preprocessing.focused_class_name,
                device=device,
                border_mode=border_mode,
            )
            h5_paths.append(h5_path)
        except Exception as e:
            logger.error(f"Failed to preprocess {exp_id}: {e}", exc_info=True)

    return h5_paths


def extract_blinded_features(
    h5_paths: list[Path],
    features_dir: Path,
    config: FullConfig,
    device: str,
) -> list[Path]:
    """Extract DINO features for blinded experiments. Returns .npz paths."""
    features_dir.mkdir(parents=True, exist_ok=True)

    # Check which experiments still need feature extraction
    to_extract = []
    npz_paths = []
    for h5_path in h5_paths:
        npz_path = features_dir / f"{h5_path.stem}.npz"
        npz_paths.append(npz_path)
        if not npz_path.exists():
            to_extract.append((h5_path, npz_path))

    if not to_extract:
        logger.info("All blinded features already extracted")
        return npz_paths

    # Load DINO backbone (v1: 128x128, no time conditioning)
    # These must match the checkpoint, not the current config defaults
    dino_cfg = config.dino
    device_torch = torch.device(device)
    V1_IMG_SIZE = 128  # from pos_embed shape (1, 65, 384) = 8x8 patches + cls

    backbone = ViTSmall(
        img_size=V1_IMG_SIZE,
        in_channels=1,
        patch_size=dino_cfg.patch_size,
        embed_dim=dino_cfg.embed_dim,
        depth=dino_cfg.depth,
        num_heads=dino_cfg.num_heads,
        time_conditioned=False,
        time_quantize_sec=0.0,
    ).to(device_torch)

    ckpt_path = config.paths.checkpoints_dir / "dino" / "best_backbone.pt"
    checkpoint = torch.load(ckpt_path, map_location=device_torch, weights_only=False)
    if "student_state_dict" in checkpoint:
        backbone.load_state_dict(checkpoint["student_state_dict"])
    else:
        backbone.load_state_dict(checkpoint)
    backbone.eval()
    logger.info(f"Loaded DINO backbone from {ckpt_path}")

    for h5_path, npz_path in to_extract:
        logger.info(f"Extracting features for {h5_path.stem}...")
        extract_features_for_experiment(
            backbone=backbone,
            h5_path=h5_path,
            output_path=npz_path,
            batch_size=512,
            num_workers=4,
            device=device_torch,
            mean=dino_cfg.dataset_mean,
            std=dino_cfg.dataset_std,
            img_size=V1_IMG_SIZE,
        )

    return npz_paths


def run_inference(
    npz_paths: list[Path],
    ckpt_dir: Path,
    device: str,
    bin_width_sec: float = 300.0,
) -> dict:
    """Run all 5-fold CropMLP checkpoints on blinded experiments.

    Returns dict with per-fold, per-experiment results.
    """
    device_torch = torch.device(device)
    all_fold_results = {}

    for fold_idx in range(5):
        ckpt_path = ckpt_dir / f"fold{fold_idx}_best.pt"
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found: {ckpt_path}")
            continue

        model = CropMLP().to(device_torch)
        ckpt = torch.load(ckpt_path, map_location=device_torch, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        logger.info(
            f"Loaded fold {fold_idx} checkpoint (val AUC: {ckpt.get('val_auc', 'N/A')})"
        )

        per_experiment = {}

        for npz_path in npz_paths:
            exp_id = npz_path.stem
            if not npz_path.exists():
                logger.warning(f"Features not found: {npz_path}")
                continue

            data = np.load(npz_path)
            features = data["features"].astype(np.float32)
            timestamps = data["timestamps"].astype(np.float64)
            rel_ts = (timestamps - timestamps.min()).astype(np.float32)

            if len(features) == 0:
                logger.warning(f"No features for {exp_id}")
                continue

            max_time = float(rel_ts.max())
            n_bins = max(1, int(np.ceil(max_time / bin_width_sec)))

            # Run inference
            features_t = torch.from_numpy(features).to(device_torch)
            all_probs = []
            with torch.no_grad():
                for start in range(0, len(features_t), 8192):
                    chunk = features_t[start:start + 8192]
                    logits = model(chunk)
                    probs = F.softmax(logits.float(), dim=-1)[:, 1]
                    all_probs.append(probs.cpu().numpy())
            crop_probs = np.concatenate(all_probs)

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

            # Experiment-level prediction
            exp_prob_r = float(np.mean(crop_probs))

            # Cumulative predictions at various time points
            cumulative_preds = {}
            for t_sec in [300, 600, 900, 1200, 1500, 1800, 2400, 3000, 3600]:
                mask = rel_ts <= t_sec
                if np.sum(mask) == 0:
                    continue
                cum_prob = float(np.mean(crop_probs[mask]))
                cumulative_preds[t_sec] = {
                    "prob_r": cum_prob,
                    "pred": int(cum_prob > 0.5),
                    "n_crops": int(np.sum(mask)),
                }

            per_experiment[exp_id] = {
                "exp_prob_r": exp_prob_r,
                "exp_pred": int(exp_prob_r > 0.5),
                "pred_label": "R" if exp_prob_r > 0.5 else "S",
                "n_total_crops": len(features),
                "max_time_min": max_time / 60,
                "bin_timeseries": bin_data,
                "cumulative_preds": {
                    str(k): v for k, v in cumulative_preds.items()
                },
            }

            logger.info(
                f"  Fold {fold_idx} | {exp_id}: P(R)={exp_prob_r:.3f} "
                f"→ {'Resistant' if exp_prob_r > 0.5 else 'Susceptible'} "
                f"({len(features)} crops, {max_time/60:.1f} min)"
            )

        all_fold_results[fold_idx] = per_experiment

    return all_fold_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_fold_timeseries(
    fold_idx: int,
    fold_result: dict,
    output_dir: Path,
) -> None:
    """Plot per-experiment time-series of P(Resistant) for one fold."""
    if not fold_result:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for exp_id, exp_data in sorted(fold_result.items()):
        bins = exp_data["bin_timeseries"]
        times = [b["bin_center_min"] for b in bins if b["mean_prob_r"] is not None]
        probs = [b["mean_prob_r"] for b in bins if b["mean_prob_r"] is not None]

        if not times:
            continue

        bid = _extract_blinded_id(exp_id)
        color = BLINDED_COLORS.get(bid, "#333333")

        ax.plot(
            times, probs,
            color=color, linestyle="-", marker="o",
            markersize=4, linewidth=2, alpha=0.8,
            label=f"{exp_id} (P(R)={exp_data['exp_prob_r']:.2f})",
        )

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.axhspan(0.5, 1.0, alpha=0.05, color="red", label="_nolegend_")
    ax.axhspan(0.0, 0.5, alpha=0.05, color="blue", label="_nolegend_")

    ax.text(62, 0.75, "Resistant", fontsize=10, color="red", alpha=0.5, ha="right")
    ax.text(62, 0.25, "Susceptible", fontsize=10, color="blue", alpha=0.5, ha="right")

    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_ylabel("Mean P(Resistant)", fontsize=12)
    ax.set_title(f"Blinded Test — Fold {fold_idx}", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 65)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(), by_label.keys(),
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=9, framealpha=0.9,
    )

    fig.tight_layout()
    fig.savefig(output_dir / f"fold{fold_idx}_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fold_crop_fractions(
    fold_idx: int,
    fold_result: dict,
    output_dir: Path,
) -> None:
    """Plot fraction of crops classified as R per 5-min bin for one fold."""
    if not fold_result:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for exp_id, exp_data in sorted(fold_result.items()):
        bins = exp_data["bin_timeseries"]
        times = [b["bin_center_min"] for b in bins if b["n_crops"] > 0]
        frac_r = [b["frac_resistant"] for b in bins if b["n_crops"] > 0]

        if not times:
            continue

        bid = _extract_blinded_id(exp_id)
        color = BLINDED_COLORS.get(bid, "#333333")

        ax.plot(
            times, frac_r,
            color=color, marker="o", markersize=4, linewidth=2,
            alpha=0.8, label=exp_id,
        )

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_ylabel("Fraction classified as Resistant", fontsize=12)
    ax.set_title(f"Blinded Test — Fold {fold_idx} — Crop Classification", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 65)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(), by_label.keys(),
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=9, framealpha=0.9,
    )

    fig.tight_layout()
    fig.savefig(
        output_dir / f"fold{fold_idx}_crop_fractions.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)


def plot_aggregate_timeseries(
    all_fold_results: dict,
    output_dir: Path,
) -> None:
    """Plot ensemble mean P(R) over time per experiment (averaged across folds)."""
    # Collect per-experiment, per-bin probabilities across all folds
    exp_bin_probs: dict[str, dict[float, list[float]]] = {}

    for fold_idx, fold_result in all_fold_results.items():
        for exp_id, exp_data in fold_result.items():
            if exp_id not in exp_bin_probs:
                exp_bin_probs[exp_id] = {}
            for b in exp_data["bin_timeseries"]:
                if b["mean_prob_r"] is not None:
                    t = b["bin_center_min"]
                    exp_bin_probs[exp_id].setdefault(t, []).append(b["mean_prob_r"])

    fig, ax = plt.subplots(figsize=(12, 6))

    for exp_id in sorted(exp_bin_probs.keys()):
        bid = _extract_blinded_id(exp_id)
        color = BLINDED_COLORS.get(bid, "#333333")
        bin_probs = exp_bin_probs[exp_id]

        times = sorted(bin_probs.keys())
        means = [np.mean(bin_probs[t]) for t in times]
        stds = [np.std(bin_probs[t]) for t in times]
        means = np.array(means)
        stds = np.array(stds)

        ax.plot(
            times, means, "o-",
            color=color, linewidth=2, markersize=4,
            label=exp_id,
        )
        ax.fill_between(times, means - stds, means + stds, alpha=0.15, color=color)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.axhspan(0.5, 1.0, alpha=0.05, color="red")
    ax.axhspan(0.0, 0.5, alpha=0.05, color="blue")

    ax.text(62, 0.75, "Resistant", fontsize=10, color="red", alpha=0.5, ha="right")
    ax.text(62, 0.25, "Susceptible", fontsize=10, color="blue", alpha=0.5, ha="right")

    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_ylabel("Ensemble Mean P(Resistant)", fontsize=12)
    ax.set_title(
        "Blinded Test — Ensemble Prediction Over Time\n"
        f"(mean ± std across {len(all_fold_results)} folds)",
        fontsize=11,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 65)
    ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=9, framealpha=0.9,
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        output_dir / "aggregate_timeseries_by_experiment.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)


def plot_aggregate_cumulative(
    all_fold_results: dict,
    output_dir: Path,
) -> None:
    """Plot cumulative P(R) over time per experiment (ensemble across folds)."""
    time_points = [300, 600, 900, 1200, 1500, 1800, 2400, 3000, 3600]
    time_min = [t / 60 for t in time_points]

    # Collect cumulative predictions per experiment across folds
    exp_cum: dict[str, dict[int, list[float]]] = {}

    for fold_idx, fold_result in all_fold_results.items():
        for exp_id, exp_data in fold_result.items():
            if exp_id not in exp_cum:
                exp_cum[exp_id] = {}
            for t_sec_str, pred in exp_data["cumulative_preds"].items():
                t_sec = int(t_sec_str)
                exp_cum[exp_id].setdefault(t_sec, []).append(pred["prob_r"])

    fig, ax = plt.subplots(figsize=(12, 6))

    for exp_id in sorted(exp_cum.keys()):
        bid = _extract_blinded_id(exp_id)
        color = BLINDED_COLORS.get(bid, "#333333")

        valid_times = []
        means = []
        stds = []
        for t_sec, t_m in zip(time_points, time_min):
            if t_sec in exp_cum[exp_id]:
                vals = exp_cum[exp_id][t_sec]
                valid_times.append(t_m)
                means.append(np.mean(vals))
                stds.append(np.std(vals))

        means = np.array(means)
        stds = np.array(stds)

        ax.plot(
            valid_times, means, "o-",
            color=color, linewidth=2, markersize=5,
            label=exp_id,
        )
        ax.fill_between(valid_times, means - stds, means + stds, alpha=0.15, color=color)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.axhspan(0.5, 1.0, alpha=0.05, color="red")
    ax.axhspan(0.0, 0.5, alpha=0.05, color="blue")

    ax.text(62, 0.75, "Resistant", fontsize=10, color="red", alpha=0.5, ha="right")
    ax.text(62, 0.25, "Susceptible", fontsize=10, color="blue", alpha=0.5, ha="right")

    ax.set_xlabel("Cumulative time (minutes)", fontsize=12)
    ax.set_ylabel("Ensemble P(Resistant)", fontsize=12)
    ax.set_title(
        "Blinded Test — Cumulative Ensemble Prediction\n"
        f"(mean ± std across {len(all_fold_results)} folds)",
        fontsize=11,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 65)
    ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=9, framealpha=0.9,
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        output_dir / "aggregate_cumulative_prediction.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate blinded test experiments with CropMLP"
    )
    parser.add_argument(
        "--blinded-dir", type=Path,
        default=Path("/mnt/f/Data_second_protocol/BlindedTest"),
        help="Path to BlindedTest folder",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("./results_crop_mlp/plots_blinded"),
        help="Output directory for plots and results",
    )
    parser.add_argument(
        "--ckpt-dir", type=Path,
        default=Path("./results_crop_mlp/checkpoints"),
        help="Directory containing fold*_best.pt checkpoints",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--skip-preprocess", action="store_true",
        help="Skip YOLO preprocessing (assume HDF5 files exist)",
    )
    parser.add_argument(
        "--skip-features", action="store_true",
        help="Skip DINO feature extraction (assume .npz files exist)",
    )
    parser.add_argument(
        "--bin-width-sec", type=float, default=300.0,
        help="Evaluation bin width in seconds (default: 300 = 5 min)",
    )
    args = parser.parse_args()

    config = FullConfig()
    config.device = args.device

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover blinded experiments
    experiments = discover_blinded_experiments(args.blinded_dir)
    logger.info(f"Found {len(experiments)} blinded experiments:")
    for exp_id, images_dir in experiments:
        logger.info(f"  {exp_id}: {images_dir}")

    # Stage 1: YOLO preprocessing
    if not args.skip_preprocess:
        logger.info("\n=== Stage 1: YOLO Preprocessing ===")
        h5_paths = preprocess_blinded(
            experiments, config.paths.preprocessed_dir, config, args.device,
        )
    else:
        h5_paths = [
            config.paths.preprocessed_dir / f"{exp_id}.h5"
            for exp_id, _ in experiments
        ]
        logger.info("Skipping preprocessing (--skip-preprocess)")

    # Stage 2: DINO feature extraction
    if not args.skip_features:
        logger.info("\n=== Stage 2: DINO Feature Extraction ===")
        npz_paths = extract_blinded_features(
            h5_paths, config.paths.features_dir, config, args.device,
        )
    else:
        npz_paths = [
            config.paths.features_dir / f"{exp_id}.npz"
            for exp_id, _ in experiments
        ]
        logger.info("Skipping feature extraction (--skip-features)")

    # Stage 3: CropMLP inference
    logger.info("\n=== Stage 3: CropMLP Inference ===")
    all_fold_results = run_inference(
        npz_paths, args.ckpt_dir, args.device,
        bin_width_sec=args.bin_width_sec,
    )

    # Stage 4: Plots
    logger.info("\n=== Stage 4: Generating Plots ===")
    for fold_idx, fold_result in all_fold_results.items():
        plot_fold_timeseries(fold_idx, fold_result, output_dir)
        plot_fold_crop_fractions(fold_idx, fold_result, output_dir)

    plot_aggregate_timeseries(all_fold_results, output_dir)
    plot_aggregate_cumulative(all_fold_results, output_dir)

    # Save results JSON
    results = {
        "model": "CropMLP",
        "bin_width_sec": args.bin_width_sec,
        "n_folds": len(all_fold_results),
        "experiments": {},
    }

    # Compute ensemble predictions
    for exp_id in sorted(set().union(*(r.keys() for r in all_fold_results.values()))):
        fold_probs = []
        for fold_idx, fold_result in all_fold_results.items():
            if exp_id in fold_result:
                fold_probs.append(fold_result[exp_id]["exp_prob_r"])

        if fold_probs:
            ensemble_prob = float(np.mean(fold_probs))
            results["experiments"][exp_id] = {
                "ensemble_prob_r": ensemble_prob,
                "ensemble_pred": "Resistant" if ensemble_prob > 0.5 else "Susceptible",
                "fold_probs": fold_probs,
                "std": float(np.std(fold_probs)),
                "n_folds": len(fold_probs),
            }

    results_path = output_dir / "blinded_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Also save detailed per-fold results
    detailed = {}
    for fold_idx, fold_result in all_fold_results.items():
        detailed[f"fold{fold_idx}"] = fold_result

    with open(output_dir / "blinded_per_fold_details.json", "w") as f:
        json.dump(detailed, f, indent=2)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("BLINDED TEST RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    for exp_id, res in results["experiments"].items():
        logger.info(
            f"  {exp_id}: P(R)={res['ensemble_prob_r']:.3f} ± {res['std']:.3f} "
            f"→ {res['ensemble_pred']}"
        )
    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"  Plots:   {output_dir}/*.png")
    logger.info(f"  Summary: {results_path}")


if __name__ == "__main__":
    main()
