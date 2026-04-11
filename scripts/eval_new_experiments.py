"""Evaluate new experiments through the full CropMLP pipeline.

End-to-end: YOLO detection → DINO feature extraction → CropMLP inference → plots.

Preprocessing matches the original training pipeline exactly:
  - 128x128 crops with reflection padding (BORDER_REFLECT_101)
  - YOLO confidence threshold 0.25
  - DINO v1 backbone (128x128, no time conditioning)
  - CLAHE + normalization (mean=0.3387, std=0.1173)

All intermediate files are written to --work-dir (default: eval_workdir/)
to avoid touching the original preprocessed/ and features/ directories.

Usage:
    PYTHONPATH=/home/mkedz/code python3 -m ast_classifier.scripts.eval_new_experiments \\
        --input-dir /mnt/f/Data_second_protocol/BlindedTest \\
        --output-dir ./results_crop_mlp/plots_blinded \\
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import h5py
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

# Preprocessing parameters matching original training pipeline
CROP_SIZE = 128
BORDER_MODE = cv2.BORDER_REFLECT_101
CONF_THRESHOLD = 0.25
DINO_IMG_SIZE = 128
DINO_MEAN = 0.3387
DINO_STD = 0.1173

COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#1b9e77", "#d95f02", "#7570b3",
    "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666",
]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_experiments(input_dir: Path) -> list[tuple[str, Path]]:
    """Find experiment directories containing images/ subfolders."""
    experiments = []
    for exp_dir in sorted(input_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        images_dir = exp_dir / "images"
        if not images_dir.exists():
            continue
        experiments.append((exp_dir.name, images_dir))
    return experiments


# ---------------------------------------------------------------------------
# Stage 1: YOLO preprocessing
# ---------------------------------------------------------------------------

def preprocess(
    experiments: list[tuple[str, Path]],
    h5_dir: Path,
    config: FullConfig,
    device: str,
) -> list[Path]:
    """YOLO-OBB detection → 128x128 reflection-padded crops → HDF5."""
    h5_dir.mkdir(parents=True, exist_ok=True)
    h5_paths = []

    for exp_id, images_dir in experiments:
        h5_path = h5_dir / f"{exp_id}.h5"
        if h5_path.exists():
            logger.info(f"Skipping {exp_id} (HDF5 exists)")
            h5_paths.append(h5_path)
            continue

        logger.info(f"Preprocessing {exp_id}...")
        try:
            h5_path = extract_experiment(
                image_dir=images_dir,
                output_dir=h5_dir,
                model_path=str(config.paths.yolo_weights),
                batch_size=config.preprocessing.yolo_batch_size,
                crop_size=CROP_SIZE,
                conf_threshold=CONF_THRESHOLD,
                focused_class_name=config.preprocessing.focused_class_name,
                device=device,
                border_mode=BORDER_MODE,
            )
            h5_paths.append(h5_path)
        except Exception as e:
            logger.error(f"Failed {exp_id}: {e}", exc_info=True)

    return h5_paths


# ---------------------------------------------------------------------------
# Stage 2: DINO feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    h5_paths: list[Path],
    npz_dir: Path,
    config: FullConfig,
    device: str,
) -> list[Path]:
    """DINO v1 backbone → 384-dim embeddings per crop."""
    npz_dir.mkdir(parents=True, exist_ok=True)

    to_extract = []
    npz_paths = []
    for h5_path in h5_paths:
        npz_path = npz_dir / f"{h5_path.stem}.npz"
        npz_paths.append(npz_path)
        if not npz_path.exists():
            to_extract.append((h5_path, npz_path))

    if not to_extract:
        logger.info("All features already extracted")
        return npz_paths

    device_torch = torch.device(device)
    dino_cfg = config.dino
    backbone = ViTSmall(
        img_size=DINO_IMG_SIZE,
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
            mean=DINO_MEAN,
            std=DINO_STD,
            img_size=DINO_IMG_SIZE,
        )

    return npz_paths


# ---------------------------------------------------------------------------
# Stage 3: CropMLP inference
# ---------------------------------------------------------------------------

def run_inference(
    npz_paths: list[Path],
    ckpt_dir: Path,
    device: str,
    bin_width_sec: float = 300.0,
) -> dict[int, dict]:
    """Run all 5-fold CropMLP checkpoints. Returns {fold: {exp_id: results}}."""
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
        logger.info(f"Fold {fold_idx} (val AUC: {ckpt.get('val_auc', 'N/A'):.3f})")

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
                        "n_crops": 0,
                        "frac_resistant": None,
                        "mean_prob_r": None,
                    })
                    continue
                bin_probs = crop_probs[in_bin]
                bin_data.append({
                    "bin_center_min": (t_lo + t_hi) / 2 / 60,
                    "n_crops": int(np.sum(in_bin)),
                    "frac_resistant": float(np.mean(bin_probs > 0.5)),
                    "mean_prob_r": float(np.mean(bin_probs)),
                })

            per_experiment[exp_id] = {
                "exp_prob_r": float(np.mean(crop_probs)),
                "n_total_crops": len(features),
                "max_time_min": max_time / 60,
                "bin_timeseries": bin_data,
            }

            logger.info(
                f"  {exp_id}: P(R)={per_experiment[exp_id]['exp_prob_r']:.3f} "
                f"({len(features)} crops, {max_time/60:.1f} min)"
            )

        all_fold_results[fold_idx] = per_experiment

    return all_fold_results


# ---------------------------------------------------------------------------
# Brightness stats
# ---------------------------------------------------------------------------

def get_brightness(h5_paths: list[Path], n_sample: int = 1000) -> dict[str, float]:
    """Return {exp_id: raw_pixel_mean} for each experiment."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    brightness = {}
    for h5_path in h5_paths:
        if not h5_path.exists():
            continue
        with h5py.File(h5_path, "r") as h:
            n = min(n_sample, h["crops"].shape[0])
            crops = h["crops"][:n]
        brightness[h5_path.stem] = float(crops.mean())
    return brightness


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_fold_timeseries(
    fold_idx: int,
    fold_result: dict,
    brightness: dict[str, float],
    output_dir: Path,
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
        bri = brightness.get(exp_id, 0)
        label = f"{exp_id} (raw \u03bc={bri:.0f}, P(R)={exp_data['exp_prob_r']:.2f})"

        ax.plot(times, probs, color=color, marker="o", markersize=3,
                linewidth=1.8, alpha=0.85, label=label)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.axhspan(0.5, 1.0, alpha=0.05, color="red")
    ax.axhspan(0.0, 0.5, alpha=0.05, color="blue")
    ax.text(62, 0.75, "Resistant", fontsize=10, color="red", alpha=0.5, ha="right")
    ax.text(62, 0.25, "Susceptible", fontsize=10, color="blue", alpha=0.5, ha="right")

    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_ylabel("Mean P(Resistant)", fontsize=12)
    ax.set_title(f"Fold {fold_idx}", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 70)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_dir / f"fold{fold_idx}_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fold_crop_fractions(
    fold_idx: int,
    fold_result: dict,
    brightness: dict[str, float],
    output_dir: Path,
) -> None:
    if not fold_result:
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (exp_id, exp_data) in enumerate(sorted(fold_result.items())):
        bins = exp_data["bin_timeseries"]
        times = [b["bin_center_min"] for b in bins if b["n_crops"] > 0]
        frac_r = [b["frac_resistant"] for b in bins if b["n_crops"] > 0]
        if not times:
            continue

        color = COLORS[i % len(COLORS)]
        bri = brightness.get(exp_id, 0)
        ax.plot(times, frac_r, color=color, marker="o", markersize=3,
                linewidth=1.8, alpha=0.85, label=f"{exp_id} (\u03bc={bri:.0f})")

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_ylabel("Fraction classified as Resistant", fontsize=12)
    ax.set_title(f"Fold {fold_idx} — Crop Classification", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 70)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_dir / f"fold{fold_idx}_crop_fractions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate(
    all_fold_results: dict[int, dict],
    brightness: dict[str, float],
    output_dir: Path,
) -> None:
    """Ensemble mean P(R) over time per experiment (mean +/- std across folds)."""
    exp_bin_probs: dict[str, dict[float, list[float]]] = {}
    for fold_result in all_fold_results.values():
        for exp_id, exp_data in fold_result.items():
            if exp_id not in exp_bin_probs:
                exp_bin_probs[exp_id] = {}
            for b in exp_data["bin_timeseries"]:
                if b["mean_prob_r"] is not None:
                    t = b["bin_center_min"]
                    exp_bin_probs[exp_id].setdefault(t, []).append(b["mean_prob_r"])

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, exp_id in enumerate(sorted(exp_bin_probs.keys())):
        color = COLORS[i % len(COLORS)]
        bin_probs = exp_bin_probs[exp_id]
        times = sorted(bin_probs.keys())
        means = np.array([np.mean(bin_probs[t]) for t in times])
        stds = np.array([np.std(bin_probs[t]) for t in times])

        bri = brightness.get(exp_id, 0)
        ax.plot(times, means, "o-", color=color, linewidth=1.8, markersize=3,
                label=f"{exp_id} (\u03bc={bri:.0f})")
        ax.fill_between(times, means - stds, means + stds, alpha=0.12, color=color)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.axhspan(0.5, 1.0, alpha=0.05, color="red")
    ax.axhspan(0.0, 0.5, alpha=0.05, color="blue")
    ax.text(62, 0.75, "Resistant", fontsize=10, color="red", alpha=0.5, ha="right")
    ax.text(62, 0.25, "Susceptible", fontsize=10, color="blue", alpha=0.5, ha="right")

    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_ylabel("Ensemble Mean P(Resistant)", fontsize=12)
    ax.set_title(
        f"Ensemble Prediction Over Time (mean \u00b1 std across {len(all_fold_results)} folds)",
        fontsize=11,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 70)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "aggregate_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full pipeline: YOLO → DINO → CropMLP → plots"
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Directory containing experiment folders (each with images/ subfolder)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for plots and results JSON",
    )
    parser.add_argument(
        "--work-dir", type=Path, default=Path("./eval_workdir"),
        help="Working directory for intermediate HDF5 and .npz files (default: eval_workdir/)",
    )
    parser.add_argument(
        "--ckpt-dir", type=Path,
        default=Path("./results_crop_mlp/checkpoints"),
        help="Directory containing fold*_best.pt checkpoints",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--skip-preprocess", action="store_true",
        help="Skip YOLO preprocessing (assume HDF5 files exist in work-dir)",
    )
    parser.add_argument(
        "--skip-features", action="store_true",
        help="Skip DINO feature extraction (assume .npz files exist in work-dir)",
    )
    parser.add_argument(
        "--bin-width-sec", type=float, default=300.0,
        help="Evaluation bin width in seconds (default: 300 = 5 min)",
    )
    args = parser.parse_args()

    config = FullConfig()
    config.device = args.device

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_dir = args.work_dir / "preprocessed"
    npz_dir = args.work_dir / "features"

    # Discover
    experiments = discover_experiments(args.input_dir)
    logger.info(f"Found {len(experiments)} experiments in {args.input_dir}")
    for exp_id, images_dir in experiments:
        logger.info(f"  {exp_id}")

    # Stage 1
    if not args.skip_preprocess:
        logger.info("\n=== Stage 1: YOLO Preprocessing ===")
        logger.info(f"  crop_size={CROP_SIZE}, border=REFLECT_101, conf={CONF_THRESHOLD}")
        h5_paths = preprocess(experiments, h5_dir, config, args.device)
    else:
        h5_paths = [h5_dir / f"{eid}.h5" for eid, _ in experiments]
        logger.info("Skipping preprocessing")

    # Stage 2
    if not args.skip_features:
        logger.info("\n=== Stage 2: DINO Feature Extraction ===")
        npz_paths = extract_features(h5_paths, npz_dir, config, args.device)
    else:
        npz_paths = [npz_dir / f"{eid}.npz" for eid, _ in experiments]
        logger.info("Skipping feature extraction")

    # Brightness
    logger.info("\n=== Brightness Stats ===")
    brightness = get_brightness(h5_paths)
    for exp_id, bri in sorted(brightness.items()):
        logger.info(f"  {exp_id}: raw pixel mean = {bri:.1f}")

    # Stage 3
    logger.info("\n=== Stage 3: CropMLP Inference ===")
    all_fold_results = run_inference(
        npz_paths, args.ckpt_dir, args.device, bin_width_sec=args.bin_width_sec,
    )

    # Stage 4
    logger.info("\n=== Stage 4: Plots ===")
    for fold_idx, fold_result in all_fold_results.items():
        plot_fold_timeseries(fold_idx, fold_result, brightness, output_dir)
        plot_fold_crop_fractions(fold_idx, fold_result, brightness, output_dir)
    plot_aggregate(all_fold_results, brightness, output_dir)

    # Save JSON
    results = {
        "pipeline": {
            "crop_size": CROP_SIZE,
            "border_mode": "BORDER_REFLECT_101",
            "conf_threshold": CONF_THRESHOLD,
            "dino_img_size": DINO_IMG_SIZE,
            "dino_mean": DINO_MEAN,
            "dino_std": DINO_STD,
        },
        "brightness": brightness,
        "n_folds": len(all_fold_results),
        "experiments": {},
    }

    for exp_id in sorted(set().union(*(r.keys() for r in all_fold_results.values()))):
        fold_probs = []
        for fold_result in all_fold_results.values():
            if exp_id in fold_result:
                fold_probs.append(fold_result[exp_id]["exp_prob_r"])
        if fold_probs:
            results["experiments"][exp_id] = {
                "ensemble_prob_r": float(np.mean(fold_probs)),
                "ensemble_pred": "Resistant" if np.mean(fold_probs) > 0.5 else "Susceptible",
                "fold_probs": fold_probs,
                "std": float(np.std(fold_probs)),
                "brightness": brightness.get(exp_id),
            }

    # Per-fold details
    detailed = {f"fold{fi}": fr for fi, fr in all_fold_results.items()}

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(output_dir / "per_fold_details.json", "w") as f:
        json.dump(detailed, f, indent=2)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("RESULTS")
    logger.info(f"{'='*60}")
    for exp_id, res in results["experiments"].items():
        bri = res.get("brightness", 0) or 0
        logger.info(
            f"  {exp_id}: P(R)={res['ensemble_prob_r']:.3f} \u00b1 {res['std']:.3f} "
            f"\u2192 {res['ensemble_pred']} (brightness={bri:.0f})"
        )
    logger.info(f"\nPlots: {output_dir}")


if __name__ == "__main__":
    main()
