"""Evaluate no-amp control experiments with 128x128 reflection-padded crops.

Preprocesses, extracts DINO features, runs CropMLP inference, and plots
P(Resistant) trajectories with per-experiment brightness annotations.

Usage:
    PYTHONPATH=/home/mkedz/code python3 -m ast_classifier.scripts.eval_no_amp_128 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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

NO_AMP_DIR = Path("/mnt/f/Data_second_protocol/Susceptible_no_amp")

# Also include blinded experiments for comparison
BLINDED_DIR = Path("/mnt/f/Data_second_protocol/BlindedTest")

COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#1b9e77", "#d95f02", "#7570b3",
    "#e7298a",
]


def discover_no_amp_experiments(no_amp_dir: Path) -> list[tuple[str, Path]]:
    experiments = []
    for exp_dir in sorted(no_amp_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        images_dir = exp_dir / "images"
        if not images_dir.exists():
            continue
        experiments.append((exp_dir.name, images_dir))
    return experiments


def discover_blinded_experiments(blinded_dir: Path) -> list[tuple[str, Path]]:
    experiments = []
    if not blinded_dir.exists():
        return experiments
    for exp_dir in sorted(blinded_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        images_dir = exp_dir / "images"
        if not images_dir.exists():
            continue
        experiments.append((exp_dir.name, images_dir))
    return experiments


def preprocess_experiments(
    experiments: list[tuple[str, Path]],
    preprocessed_dir: Path,
    config: FullConfig,
    device: str,
) -> list[Path]:
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    h5_paths = []
    for exp_id, images_dir in experiments:
        h5_path = preprocessed_dir / f"{exp_id}.h5"
        if h5_path.exists():
            logger.info(f"Skipping preprocessing for {exp_id} (already done)")
            h5_paths.append(h5_path)
            continue
        logger.info(f"Preprocessing {exp_id} (128x128, reflection padding)...")
        try:
            h5_path = extract_experiment(
                image_dir=images_dir,
                output_dir=preprocessed_dir,
                model_path=str(config.paths.yolo_weights),
                batch_size=config.preprocessing.yolo_batch_size,
                crop_size=128,
                conf_threshold=config.preprocessing.yolo_confidence,
                focused_class_name=config.preprocessing.focused_class_name,
                device=device,
                border_mode=cv2.BORDER_REFLECT_101,
            )
            h5_paths.append(h5_path)
        except Exception as e:
            logger.error(f"Failed to preprocess {exp_id}: {e}", exc_info=True)
    return h5_paths


def extract_features(
    h5_paths: list[Path],
    features_dir: Path,
    config: FullConfig,
    device: str,
) -> list[Path]:
    features_dir.mkdir(parents=True, exist_ok=True)
    to_extract = []
    npz_paths = []
    for h5_path in h5_paths:
        npz_path = features_dir / f"{h5_path.stem}.npz"
        npz_paths.append(npz_path)
        if not npz_path.exists():
            to_extract.append((h5_path, npz_path))

    if not to_extract:
        logger.info("All features already extracted")
        return npz_paths

    dino_cfg = config.dino
    device_torch = torch.device(device)
    backbone = ViTSmall(
        img_size=128,
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
            img_size=128,
        )
    return npz_paths


def get_brightness_stats(h5_path: Path, n_sample: int = 1000) -> dict:
    """Get raw and post-CLAHE brightness stats for an experiment."""
    import h5py
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    with h5py.File(h5_path, "r") as h:
        n = min(n_sample, h["crops"].shape[0])
        crops = h["crops"][:n]
        raw_mean = float(crops.mean())
        raw_std = float(crops.std())
        enhanced = np.array([clahe.apply(c) for c in crops])
        clahe_mean = float(enhanced.mean())
        clahe_std = float(enhanced.std())
    return {
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "clahe_mean": clahe_mean,
        "clahe_std": clahe_std,
    }


def run_inference(
    npz_paths: list[Path],
    ckpt_dir: Path,
    device: str,
    bin_width_sec: float = 300.0,
) -> dict:
    device_torch = torch.device(device)
    all_fold_results = {}

    for fold_idx in range(5):
        ckpt_path = ckpt_dir / f"fold{fold_idx}_best.pt"
        if not ckpt_path.exists():
            continue
        model = CropMLP().to(device_torch)
        ckpt = torch.load(ckpt_path, map_location=device_torch, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        per_experiment = {}
        for npz_path in npz_paths:
            exp_id = npz_path.stem
            if not npz_path.exists():
                continue
            data = np.load(npz_path)
            features = data["features"].astype(np.float32)
            timestamps = data["timestamps"].astype(np.float64)
            rel_ts = (timestamps - timestamps.min()).astype(np.float32)
            if len(features) == 0:
                continue

            max_time = float(rel_ts.max())
            n_bins = max(1, int(np.ceil(max_time / bin_width_sec)))

            features_t = torch.from_numpy(features).to(device_torch)
            all_probs = []
            with torch.no_grad():
                for start in range(0, len(features_t), 8192):
                    chunk = features_t[start:start + 8192]
                    logits = model(chunk)
                    probs = F.softmax(logits.float(), dim=-1)[:, 1]
                    all_probs.append(probs.cpu().numpy())
            crop_probs = np.concatenate(all_probs)

            bin_data = []
            for b in range(n_bins):
                t_lo = b * bin_width_sec
                t_hi = (b + 1) * bin_width_sec
                in_bin = (rel_ts >= t_lo) & (rel_ts < t_hi)
                if np.sum(in_bin) == 0:
                    bin_data.append({
                        "bin_center_min": (t_lo + t_hi) / 2 / 60,
                        "n_crops": 0, "mean_prob_r": None,
                    })
                    continue
                bin_data.append({
                    "bin_center_min": (t_lo + t_hi) / 2 / 60,
                    "n_crops": int(np.sum(in_bin)),
                    "mean_prob_r": float(np.mean(crop_probs[in_bin])),
                })

            per_experiment[exp_id] = {
                "exp_prob_r": float(np.mean(crop_probs)),
                "n_total_crops": len(features),
                "max_time_min": max_time / 60,
                "bin_timeseries": bin_data,
            }
        all_fold_results[fold_idx] = per_experiment
    return all_fold_results


def plot_per_fold_timeseries(
    fold_idx: int,
    fold_result: dict,
    brightness: dict,
    output_dir: Path,
    title_prefix: str = "No-Amp Control",
) -> None:
    if not fold_result:
        return
    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (exp_id, exp_data) in enumerate(sorted(fold_result.items())):
        bins = exp_data["bin_timeseries"]
        times = [b["bin_center_min"] for b in bins if b["mean_prob_r"] is not None]
        probs = [b["mean_prob_r"] for b in bins if b["mean_prob_r"] is not None]
        if not times:
            continue

        color = COLORS[i % len(COLORS)]
        bstats = brightness.get(exp_id, {})
        raw_m = bstats.get("raw_mean", 0)
        label = f"{exp_id} (raw μ={raw_m:.0f})"

        ax.plot(times, probs, color=color, marker="o", markersize=3,
                linewidth=1.8, alpha=0.85, label=label)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.axhspan(0.5, 1.0, alpha=0.05, color="red")
    ax.axhspan(0.0, 0.5, alpha=0.05, color="blue")
    ax.text(62, 0.75, "Resistant", fontsize=10, color="red", alpha=0.5, ha="right")
    ax.text(62, 0.25, "Susceptible", fontsize=10, color="blue", alpha=0.5, ha="right")

    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_ylabel("Mean P(Resistant)", fontsize=12)
    ax.set_title(f"{title_prefix} — Fold {fold_idx}", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 65)

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_dir / f"fold{fold_idx}_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_with_brightness(
    all_fold_results: dict,
    brightness: dict,
    output_dir: Path,
    title_prefix: str = "No-Amp + Blinded",
) -> None:
    """Ensemble mean P(R) over time, colored by brightness."""
    exp_bin_probs: dict[str, dict[float, list[float]]] = {}
    for fold_result in all_fold_results.values():
        for exp_id, exp_data in fold_result.items():
            if exp_id not in exp_bin_probs:
                exp_bin_probs[exp_id] = {}
            for b in exp_data["bin_timeseries"]:
                if b["mean_prob_r"] is not None:
                    t = b["bin_center_min"]
                    exp_bin_probs[exp_id].setdefault(t, []).append(b["mean_prob_r"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7),
                                    gridspec_kw={"width_ratios": [3, 1]})

    # Left: timeseries
    for i, exp_id in enumerate(sorted(exp_bin_probs.keys())):
        color = COLORS[i % len(COLORS)]
        bin_probs = exp_bin_probs[exp_id]
        times = sorted(bin_probs.keys())
        means = np.array([np.mean(bin_probs[t]) for t in times])
        stds = np.array([np.std(bin_probs[t]) for t in times])

        bstats = brightness.get(exp_id, {})
        raw_m = bstats.get("raw_mean", 0)
        label = f"{exp_id} (μ={raw_m:.0f})"

        ax1.plot(times, means, "o-", color=color, linewidth=1.8, markersize=3, label=label)
        ax1.fill_between(times, means - stds, means + stds, alpha=0.12, color=color)

    ax1.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax1.axhspan(0.5, 1.0, alpha=0.05, color="red")
    ax1.axhspan(0.0, 0.5, alpha=0.05, color="blue")
    ax1.text(62, 0.75, "Resistant", fontsize=10, color="red", alpha=0.5, ha="right")
    ax1.text(62, 0.25, "Susceptible", fontsize=10, color="blue", alpha=0.5, ha="right")
    ax1.set_xlabel("Time (minutes)", fontsize=12)
    ax1.set_ylabel("Ensemble Mean P(Resistant)", fontsize=12)
    ax1.set_title(f"{title_prefix} — Ensemble Prediction Over Time\n(mean ± std across 5 folds)", fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(0, 65)
    ax1.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Right: brightness vs P(R) scatter
    exp_ids = sorted(exp_bin_probs.keys())
    raw_means = []
    ensemble_probs = []
    for exp_id in exp_ids:
        bstats = brightness.get(exp_id, {})
        raw_means.append(bstats.get("raw_mean", 0))
        all_probs = []
        for fold_result in all_fold_results.values():
            if exp_id in fold_result:
                all_probs.append(fold_result[exp_id]["exp_prob_r"])
        ensemble_probs.append(np.mean(all_probs))

    for i, (rm, ep, eid) in enumerate(zip(raw_means, ensemble_probs, exp_ids)):
        color = COLORS[i % len(COLORS)]
        ax2.scatter(rm, ep, color=color, s=80, zorder=3, edgecolors="k", linewidth=0.5)
        ax2.annotate(eid.replace("_Ampicillin_16mgL", "").replace("_preincubated", ""),
                     (rm, ep), fontsize=6, ha="left", va="bottom",
                     xytext=(3, 3), textcoords="offset points")

    ax2.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax2.set_xlabel("Raw pixel mean (uint8)", fontsize=11)
    ax2.set_ylabel("Ensemble P(Resistant)", fontsize=11)
    ax2.set_title("Brightness vs Prediction", fontsize=11)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "aggregate_brightness_vs_prediction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_brightness_comparison(brightness: dict, output_dir: Path) -> None:
    """Bar chart of raw and CLAHE brightness across all experiments."""
    exp_ids = sorted(brightness.keys())
    raw_means = [brightness[e]["raw_mean"] for e in exp_ids]
    clahe_means = [brightness[e]["clahe_mean"] for e in exp_ids]

    # Shorten labels
    short = [e.replace("_Ampicillin_16mgL", "").replace("_preincubated", "")
             .replace("_Susceptible", "").replace("_exp1", "")
             for e in exp_ids]

    x = np.arange(len(exp_ids))
    w = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w/2, raw_means, w, label="Raw uint8 mean", color="#4575b4", alpha=0.8)
    ax.bar(x + w/2, clahe_means, w, label="Post-CLAHE mean", color="#d73027", alpha=0.8)

    # Add reference lines for training data range
    ax.axhline(60, color="#4575b4", linestyle="--", linewidth=1, alpha=0.5,
               label="Training raw mean (~60)")
    ax.axhline(90, color="#d73027", linestyle="--", linewidth=1, alpha=0.5,
               label="Training CLAHE mean (~90)")

    ax.set_xlabel("Experiment", fontsize=11)
    ax.set_ylabel("Pixel intensity (uint8)", fontsize=11)
    ax.set_title("Brightness Comparison: No-Amp + Blinded vs Training Data", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "brightness_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate no-amp + blinded experiments with 128x128 crops"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("./results_crop_mlp/plots_no_amp_128"),
    )
    parser.add_argument(
        "--ckpt-dir", type=Path,
        default=Path("./results_crop_mlp/checkpoints"),
    )
    parser.add_argument("--include-blinded", action="store_true", default=True)
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    args = parser.parse_args()

    config = FullConfig()
    config.device = args.device
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover experiments
    no_amp_exps = discover_no_amp_experiments(NO_AMP_DIR)
    logger.info(f"Found {len(no_amp_exps)} no-amp experiments")

    blinded_exps = discover_blinded_experiments(BLINDED_DIR) if args.include_blinded else []
    logger.info(f"Found {len(blinded_exps)} blinded experiments")

    all_exps = no_amp_exps + blinded_exps

    # Stage 1: Preprocess
    if not args.skip_preprocess:
        logger.info("\n=== Stage 1: YOLO Preprocessing (128x128, reflection) ===")
        h5_paths = preprocess_experiments(all_exps, config.paths.preprocessed_dir, config, args.device)
    else:
        h5_paths = [config.paths.preprocessed_dir / f"{eid}.h5" for eid, _ in all_exps]

    # Stage 2: Features
    if not args.skip_features:
        logger.info("\n=== Stage 2: DINO Feature Extraction ===")
        npz_paths = extract_features(h5_paths, config.paths.features_dir, config, args.device)
    else:
        npz_paths = [config.paths.features_dir / f"{eid}.npz" for eid, _ in all_exps]

    # Get brightness stats
    logger.info("\n=== Brightness Stats ===")
    brightness = {}
    for h5_path in h5_paths:
        if h5_path.exists():
            stats = get_brightness_stats(h5_path)
            brightness[h5_path.stem] = stats
            logger.info(f"  {h5_path.stem}: raw_mean={stats['raw_mean']:.1f}, clahe_mean={stats['clahe_mean']:.1f}")

    # Stage 3: Inference
    logger.info("\n=== Stage 3: CropMLP Inference ===")
    all_fold_results = run_inference(npz_paths, args.ckpt_dir, args.device)

    # Stage 4: Plots
    logger.info("\n=== Stage 4: Generating Plots ===")
    for fold_idx, fold_result in all_fold_results.items():
        plot_per_fold_timeseries(fold_idx, fold_result, brightness, output_dir)

    plot_aggregate_with_brightness(all_fold_results, brightness, output_dir)
    plot_brightness_comparison(brightness, output_dir)

    # Save results
    results = {"brightness": brightness, "experiments": {}}
    for exp_id in sorted(set().union(*(r.keys() for r in all_fold_results.values()))):
        fold_probs = []
        for fold_result in all_fold_results.values():
            if exp_id in fold_result:
                fold_probs.append(fold_result[exp_id]["exp_prob_r"])
        if fold_probs:
            results["experiments"][exp_id] = {
                "ensemble_prob_r": float(np.mean(fold_probs)),
                "ensemble_pred": "Resistant" if np.mean(fold_probs) > 0.5 else "Susceptible",
                "std": float(np.std(fold_probs)),
                "raw_brightness": brightness.get(exp_id, {}).get("raw_mean", None),
            }

    with open(output_dir / "no_amp_128_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    for exp_id, res in results["experiments"].items():
        bri = res.get("raw_brightness", 0) or 0
        logger.info(
            f"  {exp_id}: P(R)={res['ensemble_prob_r']:.3f} ± {res['std']:.3f} "
            f"→ {res['ensemble_pred']} (brightness={bri:.0f})"
        )
    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
