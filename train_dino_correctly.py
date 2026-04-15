"""Reproduce the DINO training run that produced ``checkpoints/dino/best_backbone.pt``.

The default ``config.py`` has drifted away from the settings that actually
trained the production backbone (best loss 0.815 at epoch 43). This script
overrides ``DINOConfig`` with the exact values pickled inside the
checkpoint, then calls the standard ``train_dino`` loop.

Two settings are critical to avoid mode collapse on this dataset:
- Reduced augmentation magnitudes — particularly ``aug_brightness=0.03``
  (the default 0.3 is ~4x the data's own dynamic range and crushes the
  microscopy contrast the model needs to learn from).
- Reduced prototypes — ``head_output_dim=4096`` instead of 65536. With
  ~207K crops, 65536 prototypes drives the teacher softmax to the uniform
  distribution and freezes the loss at ln(out_dim).

Crops must be preprocessed at 128x128 with ``cv2.BORDER_REFLECT_101``
canvas padding (see ``data/preprocessing.py:_rectify_obb_crop``). This
script asserts that on startup.

Usage (from the parent of the ast_classifier directory):
    PYTHONPATH=/home/mkedz/code python3 -m ast_classifier.train_dino_correctly \\
        --device cuda:0 \\
        --output-dir ./checkpoints/dino_repro \\
        --log-dir ./logs/dino_repro
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import torch

from .config import FullConfig
from .training.train_dino import train_dino

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def apply_checkpoint_settings(config: FullConfig) -> None:
    """Overwrite the config in place with the settings pickled into the
    production ``best_backbone.pt`` checkpoint."""
    cfg = config.dino

    # Backbone (ViT-Small)
    cfg.img_size = 128
    cfg.patch_size = 16
    cfg.embed_dim = 384
    cfg.depth = 12
    cfg.num_heads = 6
    cfg.mlp_ratio = 4.0
    cfg.drop_path_rate = 0.1
    # The production best_backbone.pt student_state_dict contains no
    # time_proj weights, so the backbone that produced it was trained
    # without time conditioning. The `True` value previously pickled into
    # the checkpoint's config field was incorrect.
    cfg.time_conditioned = False
    cfg.time_quantize_sec = 0.0

    # Projection head
    cfg.head_hidden_dim = 2048
    cfg.head_bottleneck_dim = 256
    cfg.head_output_dim = 4096  # NOT 65536 — collapses with this dataset size
    cfg.head_nlayers = 3

    # Optimisation
    cfg.batch_size = 64
    cfg.epochs = 100
    cfg.base_lr = 5e-4
    cfg.min_lr = 1e-6
    cfg.weight_decay_start = 0.04
    cfg.weight_decay_end = 0.4
    cfg.warmup_epochs = 10
    cfg.grad_clip = 3.0

    # Teacher EMA / temperature
    cfg.ema_momentum_start = 0.996
    cfg.ema_momentum_end = 1.0
    cfg.teacher_temp_start = 0.04
    cfg.teacher_temp_end = 0.07
    cfg.teacher_temp_warmup_epochs = 30
    cfg.student_temp = 0.1
    cfg.center_momentum = 0.9

    # Multi-crop
    cfg.n_global_crops = 2
    cfg.n_local_crops = 6
    cfg.global_crop_scale = (0.7, 1.0)
    cfg.local_crop_scale = (0.3, 0.6)
    cfg.local_crop_size = 64

    # Dataset
    cfg.max_crops_per_experiment = 5000

    # Normalisation (post-CLAHE statistics measured on the actual crops)
    cfg.dataset_mean = 0.3387
    cfg.dataset_std = 0.1173

    # Augmentation — magnitudes scaled to the data's narrow dynamic range
    cfg.use_clahe = True
    cfg.aug_brightness = 0.03  # NOT 0.3 — collapses the model
    cfg.aug_contrast = 0.3
    cfg.aug_noise_std_max = 0.01
    cfg.aug_defocus_max = 3

    # Crop preprocessing must match (asserted below)
    config.preprocessing.crop_size = 128


def assert_crops_match(preprocessed_dir: Path, expected_size: int = 128) -> None:
    """Sanity-check that the preprocessed HDF5 crops are the expected size.

    Crops must have been generated with ``crop_size=128`` and
    ``border_mode=cv2.BORDER_REFLECT_101``. The reflection-padding choice
    cannot be recovered from the saved arrays, but the size mismatch is
    caught here.
    """
    h5_files = sorted(preprocessed_dir.glob("**/*.h5"))
    if not h5_files:
        raise FileNotFoundError(
            f"No HDF5 crop files found under {preprocessed_dir}. "
            "Run preprocessing with crop_size=128 and "
            "border_mode=cv2.BORDER_REFLECT_101 first."
        )
    with h5py.File(h5_files[0], "r") as h:
        shape = h["crops"].shape
    if len(shape) < 3 or shape[1] != expected_size or shape[2] != expected_size:
        raise ValueError(
            f"Preprocessed crops in {h5_files[0]} have shape {shape}, "
            f"expected (N, {expected_size}, {expected_size}). Re-run "
            f"preprocessing with crop_size={expected_size} and "
            f"border_mode=cv2.BORDER_REFLECT_101."
        )
    logger.info(
        f"Verified crops: {len(h5_files)} HDF5 files, first has shape {shape}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the DINO training run for best_backbone.pt"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=None,
        help="Override path to preprocessed HDF5 crops",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./checkpoints/dino_repro"),
        help="Where to write checkpoints (default keeps the original "
        "best_backbone.pt safe by writing to ./checkpoints/dino_repro)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("./logs/dino_repro"),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override total epochs (default 100, as in the original run)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader workers"
    )
    parser.add_argument(
        "--skip-shape-check",
        action="store_true",
        help="Skip the 128x128 crop sanity check",
    )
    args = parser.parse_args()

    config = FullConfig()
    apply_checkpoint_settings(config)

    config.device = args.device
    config.num_workers = args.num_workers

    if args.preprocessed_dir is not None:
        config.paths.preprocessed_dir = args.preprocessed_dir
    if args.epochs is not None:
        config.dino.epochs = args.epochs

    # train_dino() writes to ``<checkpoints_dir>/dino/best_backbone.pt`` and
    # ``<logs_dir>/dino/``. Point checkpoints_dir at the requested output_dir
    # so files land at ``<output_dir>/dino/best_backbone.pt`` (e.g.
    # ``checkpoints/dino_repro/dino/best_backbone.pt``) and the production
    # ``checkpoints/dino/best_backbone.pt`` is not overwritten.
    config.paths.checkpoints_dir = Path(args.output_dir)
    config.paths.logs_dir = Path(args.log_dir)

    if not args.skip_shape_check:
        assert_crops_match(Path(config.paths.preprocessed_dir), expected_size=128)

    torch.manual_seed(config.seed)

    logger.info("DINO reproduction config (key fields):")
    logger.info(f"  img_size={config.dino.img_size}, "
                f"local_crop_size={config.dino.local_crop_size}")
    logger.info(f"  head_output_dim={config.dino.head_output_dim} "
                f"(critical: 65536 collapses on this dataset)")
    logger.info(f"  aug_brightness={config.dino.aug_brightness} "
                f"(critical: 0.3 collapses on this dataset)")
    logger.info(f"  dataset mean/std = "
                f"{config.dino.dataset_mean}/{config.dino.dataset_std}")
    logger.info(f"  epochs={config.dino.epochs}, batch_size={config.dino.batch_size}")
    logger.info(f"  preprocessed_dir = {config.paths.preprocessed_dir}")
    logger.info(f"  checkpoints -> {config.paths.checkpoints_dir}/dino")
    logger.info(f"  logs        -> {config.paths.logs_dir}/dino")

    train_dino(config)


if __name__ == "__main__":
    main()
