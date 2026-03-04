"""CLI entry point for training pipeline (Stages 2-5).

Usage:
    # Full pipeline
    python -m ast_classifier.scripts.train --config config.yaml

    # Individual stages
    python -m ast_classifier.scripts.train --stage dino
    python -m ast_classifier.scripts.train --stage extract
    python -m ast_classifier.scripts.train --stage classifier
    python -m ast_classifier.scripts.train --stage calibrate
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from ..config import FullConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_dino(config: FullConfig) -> Path:
    """Stage 2: DINO self-supervised pretraining."""
    from ..training.train_dino import train_dino

    logger.info("=" * 60)
    logger.info("STAGE 2: DINO Self-Supervised Pretraining")
    logger.info("=" * 60)

    backbone_path = train_dino(config)
    logger.info(f"DINO pretraining complete. Backbone saved to: {backbone_path}")
    return backbone_path


def run_extract(config: FullConfig, backbone_path: Path | None = None) -> None:
    """Stage 3: Feature extraction."""
    from ..training.extract_features import extract_all_features

    logger.info("=" * 60)
    logger.info("STAGE 3: Feature Extraction")
    logger.info("=" * 60)

    if backbone_path is None:
        backbone_path = Path(config.paths.checkpoints_dir) / "dino" / "best_backbone.pt"

    if not backbone_path.exists():
        raise FileNotFoundError(
            f"Backbone checkpoint not found: {backbone_path}. "
            "Run DINO pretraining first (--stage dino)."
        )

    extract_all_features(
        backbone_checkpoint=backbone_path,
        preprocessed_dir=config.paths.preprocessed_dir,
        output_dir=config.paths.features_dir,
        batch_size=512,
        num_workers=config.num_workers,
        device_str=config.device,
        embed_dim=config.dino.embed_dim,
        img_size=config.dino.img_size,
        patch_size=config.dino.patch_size,
        depth=config.dino.depth,
        num_heads=config.dino.num_heads,
    )
    logger.info("Feature extraction complete.")


def run_classifier(config: FullConfig) -> Path:
    """Stage 4: Temporal MIL classifier training."""
    from ..training.train_classifier import train_classifier

    logger.info("=" * 60)
    logger.info("STAGE 4: Temporal MIL Classifier Training")
    logger.info("=" * 60)

    features_dir = Path(config.paths.features_dir)
    if not features_dir.exists() or not list(features_dir.glob("*.npz")):
        raise FileNotFoundError(
            f"No features found in {features_dir}. "
            "Run feature extraction first (--stage extract)."
        )

    classifier_path = train_classifier(config)
    logger.info(f"Classifier training complete. Model saved to: {classifier_path}")
    return classifier_path


def run_calibrate(config: FullConfig) -> None:
    """Stage 5: Early-exit calibration."""
    from ..training.calibrate_exit import calibrate_early_exit

    logger.info("=" * 60)
    logger.info("STAGE 5: Early-Exit Calibration")
    logger.info("=" * 60)

    classifier_path = Path(config.paths.checkpoints_dir) / "classifier" / "best_classifier.pt"
    if not classifier_path.exists():
        raise FileNotFoundError(
            f"Classifier checkpoint not found: {classifier_path}. "
            "Run classifier training first (--stage classifier)."
        )

    result = calibrate_early_exit(config)
    logger.info("Calibration complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AST Classifier Training Pipeline"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["dino", "extract", "classifier", "calibrate", "all"],
        default="all",
        help="Which stage to run (default: all)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override data root path",
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=None,
        help="Override preprocessed data directory",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=None,
        help="Override features directory",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=None,
        help="Override checkpoints directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (default: cuda:0)",
    )
    parser.add_argument(
        "--backbone-path",
        type=Path,
        default=None,
        help="Path to pretrained backbone (for extract stage)",
    )

    args = parser.parse_args()

    # Build config
    config = FullConfig()
    config.device = args.device

    if args.data_root:
        config.paths.data_root = args.data_root
    if args.preprocessed_dir:
        config.paths.preprocessed_dir = args.preprocessed_dir
    if args.features_dir:
        config.paths.features_dir = args.features_dir
    if args.checkpoints_dir:
        config.paths.checkpoints_dir = args.checkpoints_dir

    # Ensure directories exist
    for d in [
        config.paths.preprocessed_dir,
        config.paths.features_dir,
        config.paths.checkpoints_dir,
        config.paths.logs_dir,
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.seed)

    if args.stage == "dino":
        run_dino(config)
    elif args.stage == "extract":
        run_extract(config, args.backbone_path)
    elif args.stage == "classifier":
        run_classifier(config)
    elif args.stage == "calibrate":
        run_calibrate(config)
    elif args.stage == "all":
        backbone_path = run_dino(config)
        run_extract(config, backbone_path)
        run_classifier(config)
        run_calibrate(config)


if __name__ == "__main__":
    main()
