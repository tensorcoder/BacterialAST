"""CLI entry point for Stage 1: Preprocessing pipeline.

Runs YOLO-OBB detection, crop extraction, and IoU tracking on all experiments.

Usage:
    python -m ast_classifier.scripts.preprocess \
        --data-root /path/to/MainFolder \
        --output-dir /path/to/preprocessed \
        --yolo-weights /path/to/yolo11-obb.pt \
        --num-workers 4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import h5py
import numpy as np

from ..data.preprocessing import YOLOCropExtractor, HDF5CropWriter, extract_experiment, METADATA_DTYPE
from ..data.tracking import BacteriaTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def find_experiments(data_root: Path) -> list[tuple[str, Path, int]]:
    """Discover all experiment directories.

    Returns list of (experiment_id, images_dir, label).
    """
    experiments = []
    for label_name, label_int in [("Susceptible", 0), ("Resistant", 1)]:
        label_dir = data_root / label_name
        if not label_dir.exists():
            logger.warning(f"Label directory not found: {label_dir}")
            continue

        for exp_dir in sorted(label_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            images_dir = exp_dir / "images"
            if not images_dir.exists():
                logger.warning(f"No images dir in {exp_dir}")
                continue
            experiments.append((exp_dir.name, images_dir, label_int))

    return experiments


def preprocess_all(
    data_root: Path,
    output_dir: Path,
    yolo_weights: Path,
    crop_size: int = 96,
    yolo_confidence: float = 0.5,
    yolo_batch_size: int = 16,
    iou_threshold: float = 0.3,
    max_track_age: int = 15,
    min_track_hits: int = 5,
    min_track_length: int = 150,
    device: str = "cuda:0",
) -> None:
    """Run full preprocessing pipeline on all experiments."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover experiments
    experiments = find_experiments(data_root)
    logger.info(f"Found {len(experiments)} experiments")

    # Initialize tracker
    tracker = BacteriaTracker(
        iou_threshold=iou_threshold,
        max_age=max_track_age,
        min_hits=min_track_hits,
        min_track_length=min_track_length,
    )

    # Process each experiment
    for exp_id, images_dir, label in tqdm(experiments, desc="Processing experiments"):
        h5_path = output_dir / f"{exp_id}.h5"
        csv_path = output_dir / f"{exp_id}.csv"

        if h5_path.exists() and csv_path.exists():
            logger.info(f"Skipping {exp_id} (already processed)")
            continue

        logger.info(f"Processing {exp_id} ({images_dir})")

        try:
            # Stage 1a: YOLO detection + crop extraction → HDF5
            extract_experiment(
                image_dir=images_dir,
                output_dir=output_dir,
                model_path=str(yolo_weights),
                batch_size=yolo_batch_size,
                crop_size=crop_size,
                conf_threshold=yolo_confidence,
                device=device,
            )

            # Read metadata from HDF5 to build DataFrame for tracking
            with h5py.File(h5_path, "r") as f:
                metadata = f["metadata"][:]

            if len(metadata) == 0:
                logger.warning(f"No detections in {exp_id}")
                continue

            metadata_df = pd.DataFrame({
                "frame_idx": metadata["frame_idx"],
                "detection_id": metadata["detection_id"],
                "cx": metadata["cx"],
                "cy": metadata["cy"],
                "w": metadata["w"],
                "h": metadata["h"],
                "angle": metadata["angle"],
                "confidence": metadata["confidence"],
            })

            # Stage 1b: IoU tracking
            tracked_df, track_summaries = tracker.process_experiment(metadata_df)

            # Add label and experiment_id
            tracked_df["experiment_id"] = exp_id
            tracked_df["label"] = label

            # Save tracked metadata CSV
            tracked_df.to_csv(csv_path, index=False)

            # Update HDF5 metadata with track_id
            # (Re-open and add track_id to the structured array)
            with h5py.File(h5_path, "a") as f:
                if "track_id" not in f:
                    track_ids = tracked_df["track_id"].values.astype(np.int32)
                    f.create_dataset("track_ids", data=track_ids)
                # Also update the metadata structured array to include track_id
                old_meta = f["metadata"][:]
                new_dtype = np.dtype(list(METADATA_DTYPE.descr) + [("track_id", np.int32)])
                new_meta = np.zeros(len(old_meta), dtype=new_dtype)
                for name in METADATA_DTYPE.names:
                    new_meta[name] = old_meta[name]
                new_meta["track_id"] = tracked_df["track_id"].values.astype(np.int32)
                del f["metadata"]
                f.create_dataset("metadata", data=new_meta)

            n_tracks = len(track_summaries)
            logger.info(
                f"  {exp_id}: {len(metadata_df)} detections, "
                f"{n_tracks} valid tracks"
            )

        except Exception as e:
            logger.error(f"Failed to process {exp_id}: {e}", exc_info=True)
            continue

    logger.info("Preprocessing complete!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1: Preprocess microscopy data for AST classifier"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to MainFolder containing Resistant/ and Susceptible/ dirs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for preprocessed HDF5 and CSV files",
    )
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        required=True,
        help="Path to YOLOv11-OBB .pt weights file",
    )
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--yolo-confidence", type=float, default=0.5)
    parser.add_argument("--yolo-batch-size", type=int, default=16)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--max-track-age", type=int, default=15)
    parser.add_argument("--min-track-hits", type=int, default=5)
    parser.add_argument("--min-track-length", type=int, default=150)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    preprocess_all(
        data_root=args.data_root,
        output_dir=args.output_dir,
        yolo_weights=args.yolo_weights,
        crop_size=args.crop_size,
        yolo_confidence=args.yolo_confidence,
        yolo_batch_size=args.yolo_batch_size,
        iou_threshold=args.iou_threshold,
        max_track_age=args.max_track_age,
        min_track_hits=args.min_track_hits,
        min_track_length=args.min_track_length,
        device=args.device,
    )


if __name__ == "__main__":
    main()
