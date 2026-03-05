"""CLI entry point for Stage 1: Preprocessing pipeline.

Runs YOLO-OBB detection on all experiments, filtering for in-focus bacteria
and storing crops with timestamps in HDF5 format.

Usage:
    python -m ast_classifier.scripts.preprocess \
        --data-root /path/to/MainFolder \
        --output-dir /path/to/preprocessed \
        --yolo-weights /path/to/yolo11-obb.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from ..data.preprocessing import extract_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def find_experiments(data_root: Path) -> list[tuple[str, Path, int]]:
    """Discover all experiment directories including Test/ folder.

    Returns list of (experiment_id, images_dir, label).
    Labels for Test/ experiments are inferred from EC number matching
    against the Resistant/Susceptible folders.
    """
    import re

    experiments = []
    ec_label_map: dict[str, int] = {}

    # Scan Resistant and Susceptible folders
    for label_name, label_int in [("susceptible", 0), ("resistant", 1)]:
        label_dir = data_root / label_name
        if not label_dir.exists():
            label_dir = data_root / label_name.capitalize()
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

            m = re.match(r"^(EC\d+)", exp_dir.name, re.IGNORECASE)
            if m:
                ec_label_map[m.group(1).upper()] = label_int

    # Scan Test folder and infer labels from EC number
    for name in ["Test", "test"]:
        test_dir = data_root / name
        if not test_dir.exists():
            continue
        for exp_dir in sorted(test_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            images_dir = exp_dir / "images"
            if not images_dir.exists():
                logger.warning(f"No images dir in {exp_dir}")
                continue
            m = re.match(r"^(EC\d+)", exp_dir.name, re.IGNORECASE)
            ec_num = m.group(1).upper() if m else None
            if ec_num is None or ec_num not in ec_label_map:
                logger.warning(
                    f"Cannot determine label for Test experiment {exp_dir.name}, skipping."
                )
                continue
            experiments.append((exp_dir.name, images_dir, ec_label_map[ec_num]))
        break  # only process one test dir

    return experiments


def preprocess_all(
    data_root: Path,
    output_dir: Path,
    yolo_weights: Path,
    crop_size: int = 96,
    yolo_confidence: float = 0.5,
    yolo_batch_size: int = 16,
    focused_class_name: str = "Focused",
    device: str = "cuda:0",
) -> None:
    """Run preprocessing pipeline on all experiments."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = find_experiments(data_root)
    logger.info(f"Found {len(experiments)} experiments")

    for exp_id, images_dir, label in tqdm(experiments, desc="Processing experiments"):
        h5_path = output_dir / f"{exp_id}.h5"

        if h5_path.exists():
            logger.info(f"Skipping {exp_id} (already processed)")
            continue

        logger.info(f"Processing {exp_id} ({images_dir})")

        try:
            extract_experiment(
                image_dir=images_dir,
                output_dir=output_dir,
                model_path=str(yolo_weights),
                batch_size=yolo_batch_size,
                crop_size=crop_size,
                conf_threshold=yolo_confidence,
                focused_class_name=focused_class_name,
                device=device,
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
        help="Path to folder containing resistant/ and susceptible/ dirs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for preprocessed HDF5 files",
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
    parser.add_argument("--focused-class-name", type=str, default="Focused")
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    preprocess_all(
        data_root=args.data_root,
        output_dir=args.output_dir,
        yolo_weights=args.yolo_weights,
        crop_size=args.crop_size,
        yolo_confidence=args.yolo_confidence,
        yolo_batch_size=args.yolo_batch_size,
        focused_class_name=args.focused_class_name,
        device=args.device,
    )


if __name__ == "__main__":
    main()
