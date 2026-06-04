"""Train a ViT-Tiny DINO backbone from scratch — apples-to-apples comparison
against the production ViT-Small reproduction (``train_dino_correctly.py``).

Hypothesis
----------
The ViT-Small backbone (~22M params) is over-sized for ~207K effective crops.
ViT-Tiny (~5.4M params) better matches the data scale and should be at least
as good downstream. To attribute any difference to backbone size alone, every
non-architectural setting is held fixed to the production reproduction:

- Same crops (128x128 with ``cv2.BORDER_REFLECT_101`` reflection padding)
- Same prototypes (``head_output_dim = 4096``)
- Same augmentations (``aug_brightness = 0.03``, etc.)
- Same optimizer, schedules, EMA momenta, multi-crop, dataset cap, epochs

Only differences from ``train_dino_correctly.py``:

- ``embed_dim``: 384 -> 192
- ``num_heads``: 6 -> 3
- (``depth``, ``patch_size``, ``mlp_ratio`` unchanged — this is the
  canonical ViT-Tiny/16 width specification)

Result is ViT-Tiny/16 at 1-channel 128x128 input: ~5.4M backbone params.
The DINO head is unchanged (in_dim picks up the new ``embed_dim`` from cfg).

Usage (from the parent of the ast_classifier directory):
    PYTHONPATH=/home/mkedz/code python3 -m ast_classifier.train_vit_tiny_scratch \\
        --device cuda:0 \\
        --output-dir ./checkpoints/dino_vit_tiny \\
        --log-dir ./logs/dino_vit_tiny
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


def apply_vit_tiny_settings(config: FullConfig) -> None:
    """Apply ViT-Tiny backbone config; everything else matches the production
    ViT-Small reproduction in ``train_dino_correctly.py``."""
    cfg = config.dino

    # Backbone (ViT-Tiny) — the ONLY change from the production config
    cfg.img_size = 128
    cfg.patch_size = 16
    cfg.embed_dim = 192      # was 384 (ViT-S)
    cfg.depth = 12
    cfg.num_heads = 3        # was 6 (ViT-S); head_dim stays 64
    cfg.mlp_ratio = 4.0
    cfg.drop_path_rate = 0.1
    cfg.time_conditioned = False
    cfg.time_quantize_sec = 0.0

    # Projection head — identical to production run
    cfg.head_hidden_dim = 2048
    cfg.head_bottleneck_dim = 256
    cfg.head_output_dim = 4096  # NOT 65536 — collapses with this dataset size
    cfg.head_nlayers = 3

    # Optimisation — identical
    cfg.batch_size = 64
    cfg.epochs = 100
    cfg.base_lr = 5e-4
    cfg.min_lr = 1e-6
    cfg.weight_decay_start = 0.04
    cfg.weight_decay_end = 0.4
    cfg.warmup_epochs = 10
    cfg.grad_clip = 3.0

    # Teacher EMA / temperature — identical
    cfg.ema_momentum_start = 0.996
    cfg.ema_momentum_end = 1.0
    cfg.teacher_temp_start = 0.04
    cfg.teacher_temp_end = 0.07
    cfg.teacher_temp_warmup_epochs = 30
    cfg.student_temp = 0.1
    cfg.center_momentum = 0.9

    # Multi-crop — identical
    cfg.n_global_crops = 2
    cfg.n_local_crops = 6
    cfg.global_crop_scale = (0.7, 1.0)
    cfg.local_crop_scale = (0.3, 0.6)
    cfg.local_crop_size = 64

    # Dataset — identical
    cfg.max_crops_per_experiment = 5000

    # Normalisation — identical (post-CLAHE statistics)
    cfg.dataset_mean = 0.3387
    cfg.dataset_std = 0.1173

    # Augmentation — identical to production (NOT the original DINO defaults)
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


def count_backbone_params(config: FullConfig) -> int:
    """Build the backbone once just to report its parameter count."""
    from .models.backbone import ViTSmall  # generic; accepts any width/depth
    cfg = config.dino
    model = ViTSmall(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        in_channels=1,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        drop_path_rate=cfg.drop_path_rate,
        time_conditioned=cfg.time_conditioned,
        time_quantize_sec=cfg.time_quantize_sec,
    )
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a ViT-Tiny DINO backbone from scratch "
                    "(apples-to-apples vs the production ViT-Small reproduction)"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=None,
        help="Override path to preprocessed HDF5 crops "
             "(default: same as production reproduction)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./checkpoints/dino_vit_tiny"),
        help="Where to write checkpoints "
             "(lands at ``<output-dir>/dino/best_backbone.pt``)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("./logs/dino_vit_tiny"),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override total epochs (default 100, matches production run)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader workers"
    )
    parser.add_argument(
        "--skip-shape-check",
        action="store_true",
        help="Skip the 128x128 crop sanity check",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a previous full-state checkpoint (e.g. "
             "checkpoints/dino_vit_tiny/dino/last.pt) to resume training "
             "from. Must have been saved by the current train_dino — "
             "backbone-only checkpoints cannot be resumed cleanly.",
    )
    parser.add_argument(
        "--max-crops-per-experiment",
        type=int,
        default=None,
        help="Override the per-experiment crop cap (default: 5000 from "
             "apply_vit_tiny_settings)",
    )
    args = parser.parse_args()

    config = FullConfig()
    apply_vit_tiny_settings(config)

    config.device = args.device
    config.num_workers = args.num_workers

    if args.preprocessed_dir is not None:
        config.paths.preprocessed_dir = args.preprocessed_dir
    if args.epochs is not None:
        config.dino.epochs = args.epochs
    if args.max_crops_per_experiment is not None:
        config.dino.max_crops_per_experiment = args.max_crops_per_experiment

    # train_dino() writes to ``<checkpoints_dir>/dino/best_backbone.pt`` and
    # ``<logs_dir>/dino/``. Point them at the ViT-Tiny output paths so the
    # production ViT-Small checkpoint is not overwritten.
    config.paths.checkpoints_dir = Path(args.output_dir)
    config.paths.logs_dir = Path(args.log_dir)

    if not args.skip_shape_check:
        assert_crops_match(Path(config.paths.preprocessed_dir), expected_size=128)

    torch.manual_seed(config.seed)

    n_params = count_backbone_params(config)

    logger.info("ViT-Tiny DINO training config (key fields):")
    logger.info(f"  backbone: embed_dim={config.dino.embed_dim}, "
                f"depth={config.dino.depth}, num_heads={config.dino.num_heads}")
    logger.info(f"  backbone params: {n_params:,} (~{n_params/1e6:.2f}M)")
    logger.info(f"  img_size={config.dino.img_size}, "
                f"local_crop_size={config.dino.local_crop_size}")
    logger.info(f"  head_output_dim={config.dino.head_output_dim} "
                f"(held constant vs production)")
    logger.info(f"  aug_brightness={config.dino.aug_brightness} "
                f"(held constant vs production)")
    logger.info(f"  dataset mean/std = "
                f"{config.dino.dataset_mean}/{config.dino.dataset_std}")
    logger.info(f"  epochs={config.dino.epochs}, batch_size={config.dino.batch_size}")
    logger.info(f"  preprocessed_dir = {config.paths.preprocessed_dir}")
    logger.info(f"  checkpoints -> {config.paths.checkpoints_dir}/dino")
    logger.info(f"  logs        -> {config.paths.logs_dir}/dino")

    train_dino(config, resume_from=args.resume)


if __name__ == "__main__":
    main()
