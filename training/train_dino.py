"""Stage 2: DINO self-supervised pretraining of ViT-Small backbone."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..config import DINOConfig, FullConfig
from ..data.augmentations import DINOMicroscopyAugmentation
from ..data.dataset import DINOCropDataset
from ..models.backbone import ViTSmall
from ..models.dino import DINOHead, DINOLoss, DINOWrapper

logger = logging.getLogger(__name__)


def cosine_schedule(base_value: float, final_value: float, epochs: int, warmup_epochs: int = 0, warmup_value: float = 0.0) -> list[float]:
    """Cosine schedule with optional linear warmup."""
    schedule = []
    if warmup_epochs > 0:
        for i in range(warmup_epochs):
            schedule.append(warmup_value + (base_value - warmup_value) * i / warmup_epochs)
    for i in range(epochs - warmup_epochs):
        progress = i / max(1, epochs - warmup_epochs - 1)
        schedule.append(final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * progress)))
    return schedule


def train_dino(config: FullConfig) -> Path:
    """Run DINO pretraining and return path to saved backbone checkpoint."""
    cfg = config.dino
    device = torch.device(config.device)

    # Setup logging
    log_dir = Path(config.paths.logs_dir) / "dino"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    ckpt_dir = Path(config.paths.checkpoints_dir) / "dino"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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

    # DINO loss
    dino_loss_fn = DINOLoss(
        out_dim=cfg.head_output_dim,
        teacher_temp=cfg.teacher_temp_start,
        student_temp=cfg.student_temp,
        center_momentum=cfg.center_momentum,
    ).to(device)

    # Dataset and dataloader
    transform = DINOMicroscopyAugmentation(
        global_crop_size=cfg.img_size,
        local_crop_size=cfg.local_crop_size,
        global_crop_scale=cfg.global_crop_scale,
        local_crop_scale=cfg.local_crop_scale,
        n_global_crops=cfg.n_global_crops,
        n_local_crops=cfg.n_local_crops,
    )
    dataset = DINOCropDataset(
        hdf5_dir=config.paths.preprocessed_dir,
        max_crops_per_experiment=cfg.max_crops_per_experiment,
        transform=transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer (only student parameters)
    optimizer = torch.optim.AdamW(
        list(dino.student_backbone.parameters()) + list(dino.student_head.parameters()),
        lr=cfg.base_lr * cfg.batch_size / 256,
        weight_decay=cfg.weight_decay_start,
    )

    # Schedules
    lr_schedule = cosine_schedule(
        cfg.base_lr * cfg.batch_size / 256,
        cfg.min_lr,
        cfg.epochs,
        warmup_epochs=cfg.warmup_epochs,
    )
    wd_schedule = cosine_schedule(
        cfg.weight_decay_start, cfg.weight_decay_end, cfg.epochs
    )
    momentum_schedule = cosine_schedule(
        cfg.ema_momentum_start, cfg.ema_momentum_end, cfg.epochs
    )
    teacher_temp_schedule = cosine_schedule(
        cfg.teacher_temp_start,
        cfg.teacher_temp_end,
        cfg.epochs,
        warmup_epochs=cfg.teacher_temp_warmup_epochs,
        warmup_value=cfg.teacher_temp_start,
    )

    scaler = torch.amp.GradScaler("cuda")

    logger.info(
        f"Starting DINO pretraining: {len(dataset)} crops, "
        f"{len(dataloader)} batches/epoch, {cfg.epochs} epochs"
    )

    best_loss = float("inf")

    for epoch in range(cfg.epochs):
        # Update learning rate and weight decay
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[epoch]
            param_group["weight_decay"] = wd_schedule[epoch]

        # Update teacher temperature
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

            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(dino.student_backbone.parameters()) + list(dino.student_head.parameters()),
                cfg.grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()

            # EMA update teacher + center
            with torch.no_grad():
                dino.update_teacher(momentum_schedule[epoch])
                dino_loss_fn.update_center(torch.cat(teacher_outputs, dim=0))

            epoch_loss += loss.item()
            n_batches += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{lr_schedule[epoch]:.6f}",
            )

        # Epoch logging
        avg_loss = epoch_loss / max(n_batches, 1)
        writer.add_scalar("train/dino_loss", avg_loss, epoch)
        writer.add_scalar("train/lr", lr_schedule[epoch], epoch)
        writer.add_scalar("train/ema_momentum", momentum_schedule[epoch], epoch)

        logger.info(
            f"Epoch {epoch+1}/{cfg.epochs} - Loss: {avg_loss:.4f} "
            f"LR: {lr_schedule[epoch]:.6f}"
        )

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = ckpt_dir / "best_backbone.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "student_state_dict": dino.student_backbone.state_dict(),
                    "teacher_state_dict": dino.teacher_backbone.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "config": cfg,
                },
                save_path,
            )
            logger.info(f"Saved best backbone to {save_path}")

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "student_state_dict": dino.student_backbone.state_dict(),
                    "teacher_state_dict": dino.teacher_backbone.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                ckpt_dir / f"checkpoint_epoch{epoch+1}.pt",
            )

    writer.close()

    best_path = ckpt_dir / "best_backbone.pt"
    logger.info(f"DINO pretraining complete. Best backbone: {best_path}")
    return best_path
