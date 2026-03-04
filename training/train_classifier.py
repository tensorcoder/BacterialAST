"""Stage 4: Train the Temporal MIL classifier on pre-extracted features."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..config import ClassifierConfig, FullConfig
from ..data.dataset import TemporalMILDataset, build_experiment_list, create_splits
from ..models.classifier import TemporalMILClassifier

logger = logging.getLogger(__name__)


class TimeAwareLoss(nn.Module):
    """Cross-entropy loss weighted by time: earlier correct predictions
    are worth more.

    L = lambda(t) * CE(logits, label)
    lambda(t) = 1 + alpha * (1 - t/T_max)
    """

    def __init__(
        self,
        alpha: float = 2.0,
        label_smoothing: float = 0.05,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        time_fractions: torch.Tensor,
    ) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        time_weight = 1.0 + self.alpha * (1.0 - time_fractions)
        return (ce * time_weight).mean()


class TemporalConsistencyLoss(nn.Module):
    """Encourages predictions to be consistent across adjacent time windows."""

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        prev_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        if prev_logits is None:
            return torch.tensor(0.0, device=logits.device)
        # Penalize if current prediction is worse than previous
        curr_ce = F.cross_entropy(logits, labels, reduction="none")
        prev_ce = F.cross_entropy(prev_logits, labels, reduction="none")
        return F.relu(curr_ce - prev_ce).mean()


class AttentionEntropyRegularizer(nn.Module):
    """Encourages attention weights to be neither too uniform nor too peaked."""

    def __init__(self, target_entropy_ratio: float = 0.5):
        super().__init__()
        self.target_ratio = target_entropy_ratio

    def forward(
        self,
        attention_weights: torch.Tensor,
        track_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Compute entropy of attention weights
        eps = 1e-8
        entropy = -(attention_weights * torch.log(attention_weights + eps)).sum(dim=-1)

        # Max possible entropy (uniform over valid tracks)
        n_valid = track_mask.float().sum(dim=-1)
        max_entropy = torch.log(n_valid + eps)

        # Penalize deviation from target ratio
        ratio = entropy / (max_entropy + eps)
        return ((ratio - self.target_ratio) ** 2).mean()


def train_classifier(config: FullConfig) -> Path:
    """Train the Temporal MIL classifier and return path to best checkpoint."""
    cfg = config.classifier
    device = torch.device(config.device)

    # Setup logging
    log_dir = Path(config.paths.logs_dir) / "classifier"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    ckpt_dir = Path(config.paths.checkpoints_dir) / "classifier"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Build experiment list and splits
    experiments = build_experiment_list(
        config.paths.data_root, config.paths.features_dir
    )
    train_exps, val_exps, test_exps = create_splits(
        experiments,
        train_ratio=config.data_split.train_ratio,
        val_ratio=config.data_split.val_ratio,
        test_ratio=config.data_split.test_ratio,
        seed=config.data_split.random_seed,
    )

    logger.info(
        f"Splits: {len(train_exps)} train, {len(val_exps)} val, {len(test_exps)} test"
    )

    # Datasets
    train_dataset = TemporalMILDataset(
        feature_dir=config.paths.features_dir,
        experiments=train_exps,
        time_windows_sec=cfg.time_windows,
        time_window_weights=cfg.time_window_weights,
        max_tracks=cfg.max_tracks,
        max_frames_per_track=cfg.max_frames_per_track,
        frame_subsample_rate=cfg.frame_subsample_rate,
        feature_dim=cfg.feature_dim,
        random_window=True,
    )
    val_dataset = TemporalMILDataset(
        feature_dir=config.paths.features_dir,
        experiments=val_exps,
        time_windows_sec=cfg.time_windows,
        max_tracks=cfg.max_tracks,
        max_frames_per_track=cfg.max_frames_per_track,
        frame_subsample_rate=cfg.frame_subsample_rate,
        feature_dim=cfg.feature_dim,
        random_window=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Model
    model = TemporalMILClassifier(
        feature_dim=cfg.feature_dim,
        temporal_hidden_dim=cfg.temporal_hidden_dim,
        temporal_num_layers=cfg.temporal_num_layers,
        temporal_num_heads=cfg.temporal_num_heads,
        temporal_ffn_dim=cfg.temporal_ffn_dim,
        mil_hidden_dim=cfg.mil_hidden_dim,
        population_feat_dim=cfg.population_feat_dim,
        classifier_hidden_dim=cfg.classifier_hidden_dim,
        num_classes=cfg.num_classes,
        dropout=cfg.dropout,
        delta_scales=cfg.delta_scales,
        micro_batch_size=cfg.micro_batch_size,
    ).to(device)

    logger.info(
        f"Model parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    # Loss functions
    time_loss = TimeAwareLoss(
        alpha=cfg.time_loss_alpha,
        label_smoothing=cfg.label_smoothing,
    ).to(device)
    consistency_loss = TemporalConsistencyLoss()
    entropy_reg = AttentionEntropyRegularizer()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # LR scheduler
    total_steps = len(train_loader) * cfg.epochs // cfg.gradient_accumulation
    warmup_steps = len(train_loader) * cfg.warmup_epochs // cfg.gradient_accumulation

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.5 * 0.99 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.amp.GradScaler("cuda")

    # Training loop
    best_val_auc = 0.0
    patience_counter = 0

    for epoch in range(cfg.epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{cfg.epochs}")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch_gpu = {
                "track_features": batch["track_features"].to(device, non_blocking=True),
                "track_mask": batch["track_mask"].to(device, non_blocking=True),
                "seq_mask": batch["seq_mask"].to(device, non_blocking=True),
                "time_fraction": batch["time_fraction"].to(device, non_blocking=True),
            }
            labels = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                output = model(batch_gpu)
                logits = output["logits"]

                # Time-aware loss
                loss_main = time_loss(logits, labels, batch_gpu["time_fraction"])

                # Attention entropy regularizer
                loss_entropy = entropy_reg(
                    output["attention_weights"], batch_gpu["track_mask"]
                )

                loss = (
                    loss_main
                    + cfg.attention_entropy_weight * loss_entropy
                )

            # Gradient accumulation
            scaled_loss = loss / cfg.gradient_accumulation
            scaler.scale(scaled_loss).backward()

            if (batch_idx + 1) % cfg.gradient_accumulation == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validate
        val_auc, val_loss, val_acc = validate(
            model, val_loader, time_loss, device
        )

        # Logging
        writer.add_scalar("train/loss", avg_train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/auc", val_auc, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        logger.info(
            f"Epoch {epoch+1}/{cfg.epochs} - "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val AUC: {val_auc:.4f} "
            f"Val Acc: {val_acc:.4f}"
        )

        # Checkpoint
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            save_path = ckpt_dir / "best_classifier.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": val_auc,
                    "val_loss": val_loss,
                    "config": cfg,
                },
                save_path,
            )
            logger.info(f"New best model (AUC={val_auc:.4f}) saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                logger.info(
                    f"Early stopping at epoch {epoch+1} "
                    f"(no improvement for {patience_counter} epochs)"
                )
                break

    writer.close()

    best_path = ckpt_dir / "best_classifier.pt"
    logger.info(f"Classifier training complete. Best AUC: {best_val_auc:.4f}")
    return best_path


def validate(
    model: TemporalMILClassifier,
    val_loader: DataLoader,
    loss_fn: TimeAwareLoss,
    device: torch.device,
) -> tuple[float, float, float]:
    """Validate and return (AUC, loss, accuracy)."""
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch_gpu = {
                "track_features": batch["track_features"].to(device, non_blocking=True),
                "track_mask": batch["track_mask"].to(device, non_blocking=True),
                "seq_mask": batch["seq_mask"].to(device, non_blocking=True),
                "time_fraction": batch["time_fraction"].to(device, non_blocking=True),
            }
            labels = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                output = model(batch_gpu)
                loss = loss_fn(output["logits"], labels, batch_gpu["time_fraction"])

            probs = F.softmax(output["logits"].float(), dim=-1)[:, 1]
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            total_loss += loss.item()
            n_batches += 1

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5

    accuracy = ((all_probs > 0.5).astype(int) == all_labels).mean()
    avg_loss = total_loss / max(n_batches, 1)

    return auc, avg_loss, accuracy
