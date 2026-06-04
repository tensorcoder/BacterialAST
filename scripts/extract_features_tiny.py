"""Extract per-crop features from one of the two new ViT-Tiny DINO backbones.

The two checkpoints are byte-compatible with ``models.backbone.ViTSmall``
(which is misnamed — it's a generic ViT). Both were trained from scratch
at 128x128 (reflection-padded) by ``train_vit_tiny_scratch.py`` with
``embed_dim=192, depth=12, num_heads=3, time_conditioned=False``. The only
difference between the two runs is ``max_crops_per_experiment``:

    5k:  ./checkpoints/dino_vit_tiny/dino/best_backbone.pt      (261,732 crops)
    10k: ./checkpoints/dino_vit_tiny_10k/dino/best_backbone.pt  (516,732 crops)

Usage (from /home/mkedz/code):
    PYTHONPATH=/home/mkedz/code ./ast_classifier/.venv/bin/python \
        -m ast_classifier.scripts.extract_features_tiny --variant 5k
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..training.extract_features import extract_all_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


VARIANTS = {
    "5k": {
        "checkpoint": Path("./checkpoints/dino_vit_tiny/dino/best_backbone.pt"),
        "features_dir": Path("./features_tiny_5k"),
    },
    "10k": {
        "checkpoint": Path("./checkpoints/dino_vit_tiny_10k/dino/best_backbone.pt"),
        "features_dir": Path("./features_tiny_10k"),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-crop features from a ViT-Tiny DINO backbone."
    )
    parser.add_argument(
        "--variant", choices=sorted(VARIANTS.keys()), required=True,
        help="Which Tiny variant to use (5k = 5000 crops/exp during DINO, "
             "10k = 10000).",
    )
    parser.add_argument(
        "--preprocessed-dir", type=Path,
        default=Path("./preprocessed"),
        help="Directory of 128x128 HDF5 crop files (default: ./preprocessed)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
    )
    parser.add_argument(
        "--batch-size", type=int, default=512,
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
    )
    args = parser.parse_args()

    v = VARIANTS[args.variant]
    ckpt = v["checkpoint"]
    out_dir = v["features_dir"]
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not args.preprocessed_dir.exists():
        raise FileNotFoundError(
            f"Preprocessed dir not found: {args.preprocessed_dir}"
        )

    logger.info(f"Variant: dino_vit_tiny ({args.variant})")
    logger.info(f"  checkpoint: {ckpt}")
    logger.info(f"  preprocessed_dir: {args.preprocessed_dir}")
    logger.info(f"  output features_dir: {out_dir}")

    extract_all_features(
        backbone_checkpoint=ckpt,
        preprocessed_dir=args.preprocessed_dir,
        output_dir=out_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_str=args.device,
        # ViT-Tiny architecture (matches train_vit_tiny_scratch.py)
        embed_dim=192,
        depth=12,
        num_heads=3,
        img_size=128,
        patch_size=16,
        # Tiny was trained NOT time-conditioned
        time_conditioned=False,
        time_quantize_sec=0.0,
        # Same normalisation as training (post-CLAHE stats)
        dataset_mean=0.3387,
        dataset_std=0.1173,
    )

    logger.info(f"Done. Features in {out_dir}")


if __name__ == "__main__":
    main()
