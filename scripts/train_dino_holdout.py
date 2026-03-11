"""Train DINO with strain-holdout: exclude specified strains from pretraining.

Saves checkpoints to a separate directory so the original DINO model is preserved.

Usage:
    PYTHONPATH=/path/to/parent python3 -m ast_classifier.scripts.train_dino_holdout \
        --exclude-strains EC35,EC40,EC33,EC39 \
        --max-crops 20000 \
        --output-dir ./checkpoints/dino_holdout \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import logging
import math
import re
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..config import FullConfig
from ..data.augmentations import DINOMicroscopyAugmentation
from ..data.dataset import DINOCropDataset
from ..models.backbone import ViTSmall
from ..models.dino import DINOHead, DINOLoss, DINOWrapper
from ..training.train_dino import cosine_schedule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class FilteredDINOCropDataset(DINOCropDataset):
    """DINOCropDataset that can exclude experiments by strain (EC number)."""

    def __init__(
        self,
        hdf5_dir: Path,
        max_crops_per_experiment: int = 5000,
        transform=None,
        exclude_strains: set[str] | None = None,
    ):
        self.exclude_strains = exclude_strains or set()
        super().__init__(
            hdf5_dir=hdf5_dir,
            max_crops_per_experiment=max_crops_per_experiment,
            transform=transform,
        )

    def _build_index(self) -> None:
        """Override to filter out excluded strains."""
        import h5py
        import random

        hdf5_files = sorted(self.hdf5_dir.glob("**/*.h5"))
        included = 0
        excluded = 0

        for h5_path in hdf5_files:
            # Extract EC number from filename
            ec_match = re.match(r"^(EC\d+)", h5_path.stem, re.IGNORECASE)
            if ec_match:
                ec_num = ec_match.group(1).upper()
                if ec_num in self.exclude_strains:
                    excluded += 1
                    continue

            with h5py.File(h5_path, "r") as f:
                n_crops = f["crops"].shape[0]
            if n_crops == 0:
                continue

            indices = list(range(n_crops))
            if n_crops > self.max_crops:
                indices = sorted(random.sample(indices, self.max_crops))

            for idx in indices:
                self.index.append((h5_path, idx))
            included += 1

        logger.info(
            f"Dataset: {included} experiments included, {excluded} excluded "
            f"(strains: {self.exclude_strains}), {len(self.index)} total crops"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="DINO training with strain holdout")
    parser.add_argument(
        "--exclude-strains", type=str, required=True,
        help="Comma-separated list of EC strains to exclude (e.g. EC35,EC40,EC33,EC39)",
    )
    parser.add_argument("--max-crops", type=int, default=20000,
                        help="Max crops per experiment (default: 20000)")
    parser.add_argument("--output-dir", type=Path, default=Path("./checkpoints/dino_holdout"),
                        help="Checkpoint output directory")
    parser.add_argument("--log-dir", type=Path, default=Path("./logs/dino_holdout"))
    parser.add_argument("--preprocessed-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=15,
                        help="Stop if loss hasn't improved for N epochs (default: 15)")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Path to checkpoint to resume from (e.g. checkpoints/dino_holdout/best_backbone.pt)")
    args = parser.parse_args()

    # Parse excluded strains
    exclude_strains = {s.strip().upper() for s in args.exclude_strains.split(",")}
    logger.info(f"Excluding strains: {sorted(exclude_strains)}")

    config = FullConfig()
    config.device = args.device
    cfg = config.dino

    if args.epochs is not None:
        cfg.epochs = args.epochs
    cfg.max_crops_per_experiment = args.max_crops

    if args.preprocessed_dir:
        config.paths.preprocessed_dir = args.preprocessed_dir

    device = torch.device(config.device)

    # Directories
    ckpt_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    # Build models
    backbone_kwargs = dict(
        img_size=cfg.img_size,
        in_channels=1,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        drop_path_rate=cfg.drop_path_rate,
    )
    head_kwargs = dict(
        in_dim=cfg.embed_dim,
        hidden_dim=cfg.head_hidden_dim,
        bottleneck_dim=cfg.head_bottleneck_dim,
        out_dim=cfg.head_output_dim,
    )

    dino = DINOWrapper(
        backbone_kwargs=backbone_kwargs,
        head_kwargs=head_kwargs,
    ).to(device)

    dino_loss_fn = DINOLoss(
        out_dim=cfg.head_output_dim,
        teacher_temp=cfg.teacher_temp_start,
        student_temp=cfg.student_temp,
        center_momentum=cfg.center_momentum,
    ).to(device)

    # Dataset with strain filtering
    transform = DINOMicroscopyAugmentation(
        global_crop_size=cfg.img_size,
        local_crop_size=cfg.local_crop_size,
        global_scale=cfg.global_crop_scale,
        local_scale=cfg.local_crop_scale,
        n_global_crops=cfg.n_global_crops,
        n_local_crops=cfg.n_local_crops,
        brightness=cfg.aug_brightness,
        contrast=cfg.aug_contrast,
        noise_std_range=(0.0, cfg.aug_noise_std_max),
        defocus_range=(0, cfg.aug_defocus_max),
        mean=(cfg.dataset_mean,),
        std=(cfg.dataset_std,),
        use_clahe=cfg.use_clahe,
    )
    dataset = FilteredDINOCropDataset(
        hdf5_dir=config.paths.preprocessed_dir,
        max_crops_per_experiment=cfg.max_crops_per_experiment,
        transform=transform,
        exclude_strains=exclude_strains,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(dino.student_backbone.parameters()) + list(dino.student_head.parameters()),
        lr=cfg.base_lr * cfg.batch_size / 256,
        weight_decay=cfg.weight_decay_start,
    )

    # Schedules
    lr_schedule = cosine_schedule(
        cfg.base_lr * cfg.batch_size / 256, cfg.min_lr, cfg.epochs,
        warmup_epochs=cfg.warmup_epochs,
    )
    wd_schedule = cosine_schedule(cfg.weight_decay_start, cfg.weight_decay_end, cfg.epochs)
    momentum_schedule = cosine_schedule(cfg.ema_momentum_start, cfg.ema_momentum_end, cfg.epochs)
    teacher_temp_schedule = cosine_schedule(
        cfg.teacher_temp_start, cfg.teacher_temp_end, cfg.epochs,
        warmup_epochs=cfg.teacher_temp_warmup_epochs,
        warmup_value=cfg.teacher_temp_start,
    )

    scaler = torch.amp.GradScaler("cuda")

    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float("inf")
    patience_counter = 0

    if args.resume is not None:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        dino.student_backbone.load_state_dict(ckpt["student_state_dict"])
        dino.teacher_backbone.load_state_dict(ckpt["teacher_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt["loss"]
        # Load heads if saved (newer checkpoints)
        if "student_head_state_dict" in ckpt:
            dino.student_head.load_state_dict(ckpt["student_head_state_dict"])
            dino.teacher_head.load_state_dict(ckpt["teacher_head_state_dict"])
            logger.info("Loaded DINO head state dicts")
        else:
            # Older checkpoint without heads — sync teacher head from student
            dino.teacher_head.load_state_dict(dino.student_head.state_dict())
            logger.info("Heads not in checkpoint, re-initialized and synced")
        if "loss_center" in ckpt:
            dino_loss_fn.center.copy_(ckpt["loss_center"])
            logger.info("Loaded loss center")
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
            logger.info("Loaded GradScaler state")
        logger.info(
            f"Resumed from epoch {ckpt['epoch']+1}, loss={ckpt['loss']:.4f}. "
            f"Continuing from epoch {start_epoch+1}."
        )
        del ckpt

    batches_per_epoch = len(dataloader)
    est_time_per_batch = 0.4  # seconds, rough estimate
    remaining_epochs = cfg.epochs - start_epoch
    est_epoch_min = batches_per_epoch * est_time_per_batch / 60
    est_total_hours = est_epoch_min * remaining_epochs / 60

    logger.info(
        f"Starting DINO pretraining (strain-holdout): {len(dataset)} crops, "
        f"{batches_per_epoch} batches/epoch, {remaining_epochs} epochs remaining"
    )
    logger.info(
        f"Estimated time: ~{est_epoch_min:.0f} min/epoch, ~{est_total_hours:.0f} hours total"
    )

    # Save config for reproducibility
    import json
    config_info = {
        "exclude_strains": sorted(exclude_strains),
        "max_crops_per_experiment": cfg.max_crops_per_experiment,
        "n_crops": len(dataset),
        "n_experiments_included": batches_per_epoch,
        "epochs": cfg.epochs,
    }
    with open(ckpt_dir / "training_config.json", "w") as f:
        json.dump(config_info, f, indent=2)

    early_stop_patience = args.early_stop_patience

    for epoch in range(start_epoch, cfg.epochs):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[epoch]
            param_group["weight_decay"] = wd_schedule[epoch]

        dino_loss_fn.teacher_temp = teacher_temp_schedule[epoch]

        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"DINO Epoch {epoch+1}/{cfg.epochs}")
        for batch_idx, (global_crops, local_crops) in enumerate(pbar):
            all_crops = []
            for gc in global_crops:
                all_crops.append(gc.to(device, non_blocking=True))
            for lc in local_crops:
                all_crops.append(lc.to(device, non_blocking=True))

            n_global = len(global_crops)

            with torch.amp.autocast("cuda"):
                student_outputs = dino.forward_student(all_crops)
                with torch.no_grad():
                    teacher_outputs = dino.forward_teacher(all_crops[:n_global])
                loss = dino_loss_fn(student_outputs, teacher_outputs)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(dino.student_backbone.parameters()) + list(dino.student_head.parameters()),
                cfg.grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                dino.update_teacher(momentum_schedule[epoch])
                dino_loss_fn.update_center(torch.cat(teacher_outputs, dim=0))

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_schedule[epoch]:.6f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        writer.add_scalar("train/dino_loss", avg_loss, epoch)
        writer.add_scalar("train/lr", lr_schedule[epoch], epoch)
        writer.add_scalar("train/ema_momentum", momentum_schedule[epoch], epoch)

        logger.info(
            f"Epoch {epoch+1}/{cfg.epochs} - Loss: {avg_loss:.4f} "
            f"LR: {lr_schedule[epoch]:.6f}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            save_path = ckpt_dir / "best_backbone.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "student_state_dict": dino.student_backbone.state_dict(),
                    "teacher_state_dict": dino.teacher_backbone.state_dict(),
                    "student_head_state_dict": dino.student_head.state_dict(),
                    "teacher_head_state_dict": dino.teacher_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_center": dino_loss_fn.center.clone(),
                    "scaler_state_dict": scaler.state_dict(),
                    "loss": avg_loss,
                    "config": cfg,
                    "exclude_strains": sorted(exclude_strains),
                },
                save_path,
            )
            logger.info(f"Saved best backbone to {save_path}")
        else:
            patience_counter += 1
            logger.info(
                f"No improvement for {patience_counter}/{early_stop_patience} epochs "
                f"(best: {best_loss:.4f}, current: {avg_loss:.4f})"
            )

        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "student_state_dict": dino.student_backbone.state_dict(),
                    "teacher_state_dict": dino.teacher_backbone.state_dict(),
                    "student_head_state_dict": dino.student_head.state_dict(),
                    "teacher_head_state_dict": dino.teacher_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_center": dino_loss_fn.center.clone(),
                    "scaler_state_dict": scaler.state_dict(),
                    "loss": avg_loss,
                    "exclude_strains": sorted(exclude_strains),
                },
                ckpt_dir / f"checkpoint_epoch{epoch+1}.pt",
            )

        if patience_counter >= early_stop_patience:
            logger.info(
                f"Early stopping at epoch {epoch+1}: loss hasn't improved for "
                f"{early_stop_patience} epochs. Best loss: {best_loss:.4f}"
            )
            break

    writer.close()
    logger.info(f"DINO pretraining (strain-holdout) complete. Best loss: {best_loss:.4f}")
    logger.info(f"Best backbone saved to: {ckpt_dir / 'best_backbone.pt'}")


if __name__ == "__main__":
    main()
