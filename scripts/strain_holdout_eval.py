"""Strain-holdout cross-validation: hold out 2R + 2S strains per fold.

Evaluates whether the classifier learns antibiotic response biology
rather than strain-specific morphological fingerprints.

Usage:
    PYTHONPATH=/path/to/parent python3 -m ast_classifier.scripts.strain_holdout_eval \
        --features-dir ./features \
        --checkpoints-dir ./checkpoints \
        --output-dir ./results_strain_holdout \
        --n-folds 5 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import FullConfig
from ..data.dataset import (
    ExperimentMeta,
    PopulationTemporalDataset,
    population_temporal_collate,
)
from ..models.classifier import PopulationTemporalClassifier
from ..utils.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_ec(experiment_id: str) -> str | None:
    import re
    m = re.match(r"^(EC\d+)", experiment_id, re.IGNORECASE)
    return m.group(1).upper() if m else None


def build_strain_grouped_experiments(
    features_dir: Path,
    data_root: Path,
) -> dict[int, dict[str, list[ExperimentMeta]]]:
    """Group experiments by label and strain.

    Returns {label: {strain: [ExperimentMeta, ...]}}.
    """
    features_dir = Path(features_dir)
    data_root = Path(data_root)
    groups: dict[int, dict[str, list[ExperimentMeta]]] = {0: {}, 1: {}}

    for label_name, label_int in [("susceptible", 0), ("resistant", 1)]:
        label_dir = None
        for d in data_root.iterdir():
            if d.is_dir() and d.name.lower() == label_name:
                label_dir = d
                break
        if label_dir is None:
            continue

        for exp_dir in sorted(label_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            exp_id = exp_dir.name
            ec = _extract_ec(exp_id)
            if ec is None:
                continue
            feat_path = features_dir / f"{exp_id}.npz"
            if not feat_path.exists():
                continue
            meta = ExperimentMeta(
                experiment_id=exp_id, label=label_int, features_path=feat_path
            )
            groups[label_int].setdefault(ec, []).append(meta)

    # Also include Test/ experiments (assigned to their strain group)
    test_dir = None
    for d in data_root.iterdir():
        if d.is_dir() and d.name.lower() == "test":
            test_dir = d
            break
    if test_dir is not None:
        # Build ec->label map from above
        ec_label = {}
        for label_int, strain_dict in groups.items():
            for ec in strain_dict:
                ec_label[ec] = label_int

        for exp_dir in sorted(test_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            exp_id = exp_dir.name
            ec = _extract_ec(exp_id)
            if ec is None or ec not in ec_label:
                continue
            feat_path = features_dir / f"{exp_id}.npz"
            if not feat_path.exists():
                continue
            label_int = ec_label[ec]
            meta = ExperimentMeta(
                experiment_id=exp_id, label=label_int, features_path=feat_path
            )
            groups[label_int].setdefault(ec, []).append(meta)

    return groups


def generate_folds(
    groups: dict[int, dict[str, list[ExperimentMeta]]],
    n_holdout_per_class: int = 2,
    n_folds: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Generate strain-holdout folds.

    Each fold holds out n_holdout_per_class strains from each class.
    """
    rng = random.Random(seed)
    r_strains = sorted(groups[1].keys())
    s_strains = sorted(groups[0].keys())

    logger.info(
        f"Strains: {len(r_strains)} resistant ({r_strains}), "
        f"{len(s_strains)} susceptible ({s_strains})"
    )

    # Generate unique holdout combinations
    all_r_combos = list(itertools.combinations(r_strains, n_holdout_per_class))
    all_s_combos = list(itertools.combinations(s_strains, n_holdout_per_class))
    all_folds = [(r, s) for r in all_r_combos for s in all_s_combos]
    rng.shuffle(all_folds)

    folds = []
    for r_holdout, s_holdout in all_folds[:n_folds]:
        holdout_strains = set(r_holdout) | set(s_holdout)
        test_exps = []
        train_val_exps = []

        for label_int, strain_dict in groups.items():
            for ec, exps in strain_dict.items():
                if ec in holdout_strains:
                    test_exps.extend(exps)
                else:
                    train_val_exps.extend(exps)

        # Split train_val into train/val (hold out 1 experiment per strain for val)
        val_exps = []
        train_exps = []
        strain_seen: set[str] = set()
        rng.shuffle(train_val_exps)
        for exp in train_val_exps:
            ec = _extract_ec(exp.experiment_id)
            if ec not in strain_seen and len(val_exps) < max(3, len(train_val_exps) // 5):
                val_exps.append(exp)
                strain_seen.add(ec)
            else:
                train_exps.append(exp)

        folds.append({
            "holdout_r": list(r_holdout),
            "holdout_s": list(s_holdout),
            "train": train_exps,
            "val": val_exps,
            "test": test_exps,
        })

    return folds


# ---------------------------------------------------------------------------
# Training (simplified single-fold)
# ---------------------------------------------------------------------------

class TimeAwareLoss(nn.Module):
    def __init__(self, alpha: float = 2.0, label_smoothing: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits, labels, time_fractions):
        ce = F.cross_entropy(
            logits, labels, label_smoothing=self.label_smoothing, reduction="none"
        )
        time_weight = 1.0 + self.alpha * (1.0 - time_fractions)
        return (ce * time_weight).mean()


def train_fold(
    fold_idx: int,
    fold: dict,
    config: FullConfig,
    ckpt_dir: Path,
) -> Path:
    """Train classifier for one fold, return path to best checkpoint."""
    cfg = config.classifier
    device = torch.device(config.device)

    train_exps = fold["train"]
    val_exps = fold["val"]

    logger.info(
        f"Fold {fold_idx}: {len(train_exps)} train, {len(val_exps)} val, "
        f"{len(fold['test'])} test | Holdout R: {fold['holdout_r']}, S: {fold['holdout_s']}"
    )

    train_dataset = PopulationTemporalDataset(
        feature_dir=config.paths.features_dir,
        experiments=train_exps,
        time_bin_width_sec=cfg.time_bin_width_sec,
        time_windows_sec=cfg.time_windows,
        time_window_weights=cfg.time_window_weights,
        max_crops_per_bin=cfg.max_crops_per_bin,
        feature_dim=cfg.feature_dim,
        random_window=True,
        samples_per_experiment=8,
    )
    val_dataset = PopulationTemporalDataset(
        feature_dir=config.paths.features_dir,
        experiments=val_exps,
        time_bin_width_sec=cfg.time_bin_width_sec,
        time_windows_sec=cfg.time_windows,
        max_crops_per_bin=cfg.max_crops_per_bin,
        feature_dim=cfg.feature_dim,
        random_window=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True,
        collate_fn=population_temporal_collate,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
        collate_fn=population_temporal_collate,
    )

    model = PopulationTemporalClassifier(
        feature_dim=cfg.feature_dim,
        temporal_hidden_dim=cfg.temporal_hidden_dim,
        temporal_num_layers=cfg.temporal_num_layers,
        temporal_num_heads=cfg.temporal_num_heads,
        temporal_ffn_dim=cfg.temporal_ffn_dim,
        classifier_hidden_dim=cfg.classifier_hidden_dim,
        num_classes=cfg.num_classes,
        dropout=cfg.dropout,
        max_count_normalizer=float(cfg.max_crops_per_bin),
        use_delta_features=cfg.use_delta_features,
        bin_encoder_type=cfg.bin_encoder_type,
        bin_attn_heads=cfg.bin_attn_heads,
    ).to(device)

    loss_fn = TimeAwareLoss(alpha=cfg.time_loss_alpha, label_smoothing=cfg.label_smoothing).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = len(train_loader) * cfg.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.5 * 0.99 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")

    best_val_auc = 0.0
    patience_counter = 0
    save_path = ckpt_dir / f"fold{fold_idx}_best.pt"

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch_gpu = {
                k: batch[k].to(device, non_blocking=True)
                for k in ["bin_features", "bin_mask", "crop_mask", "bin_times", "bin_counts", "time_fraction"]
            }
            labels = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                output = model(batch_gpu)
                loss = loss_fn(output["logits"], labels, batch_gpu["time_fraction"])

                # Auxiliary per-bin loss
                if cfg.bin_aux_loss_weight > 0 and "bin_logits" in output:
                    bin_mask = batch_gpu["bin_mask"]
                    bin_labels = labels.unsqueeze(1).expand_as(bin_mask)
                    valid_logits = output["bin_logits"][bin_mask]
                    valid_labels = bin_labels[bin_mask]
                    if valid_logits.numel() > 0:
                        aux_loss = F.cross_entropy(
                            valid_logits, valid_labels,
                            label_smoothing=cfg.label_smoothing,
                        )
                        loss = loss + cfg.bin_aux_loss_weight * aux_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validate
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch_gpu = {
                    k: batch[k].to(device, non_blocking=True)
                    for k in ["bin_features", "bin_mask", "crop_mask", "bin_times", "bin_counts", "time_fraction"]
                }
                with torch.amp.autocast("cuda"):
                    output = model(batch_gpu)
                probs = F.softmax(output["logits"].float(), dim=-1)[:, 1]
                all_probs.append(probs.cpu().numpy())
                all_labels.append(batch["label"].numpy())

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            val_auc = 0.5

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict(), "config": cfg}, save_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                break

        if (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(
                f"  Fold {fold_idx} Epoch {epoch+1} - Loss: {avg_loss:.4f} Val AUC: {val_auc:.4f}"
            )

    logger.info(f"  Fold {fold_idx} best val AUC: {best_val_auc:.4f}")
    return save_path


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_fold(
    fold_idx: int,
    fold: dict,
    ckpt_path: Path,
    config: FullConfig,
) -> dict:
    """Evaluate a trained fold on its held-out strains at multiple time points."""
    cfg = config.classifier
    device = torch.device(config.device)

    model = PopulationTemporalClassifier(
        feature_dim=cfg.feature_dim,
        temporal_hidden_dim=cfg.temporal_hidden_dim,
        temporal_num_layers=cfg.temporal_num_layers,
        temporal_num_heads=cfg.temporal_num_heads,
        temporal_ffn_dim=cfg.temporal_ffn_dim,
        classifier_hidden_dim=cfg.classifier_hidden_dim,
        num_classes=cfg.num_classes,
        dropout=cfg.dropout,
        max_count_normalizer=float(cfg.max_crops_per_bin),
        use_delta_features=cfg.use_delta_features,
        bin_encoder_type=cfg.bin_encoder_type,
        bin_attn_heads=cfg.bin_attn_heads,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_exps = fold["test"]
    eval_times = [60, 120, 180, 300, 600, 900, 1800, 3600]

    accuracy_vs_time = {}
    all_results = {}

    for window_sec in eval_times:
        dataset = PopulationTemporalDataset(
            feature_dir=config.paths.features_dir,
            experiments=test_exps,
            time_bin_width_sec=cfg.time_bin_width_sec,
            time_windows_sec=[window_sec],
            max_crops_per_bin=cfg.max_crops_per_bin,
            feature_dim=cfg.feature_dim,
            random_window=False,
        )
        loader = DataLoader(
            dataset, batch_size=len(test_exps), shuffle=False,
            num_workers=0, collate_fn=population_temporal_collate,
        )

        with torch.no_grad():
            for batch in loader:
                batch_gpu = {
                    k: batch[k].to(device, non_blocking=True)
                    for k in ["bin_features", "bin_mask", "crop_mask", "bin_times", "bin_counts", "time_fraction"]
                }
                with torch.amp.autocast("cuda"):
                    output = model(batch_gpu)
                probs = F.softmax(output["logits"].float(), dim=-1)[:, 1].cpu().numpy()
                labels = batch["label"].numpy()
                preds = (probs > 0.5).astype(int)

        acc = (preds == labels).mean()
        accuracy_vs_time[window_sec] = float(acc)
        all_results[window_sec] = {"probs": probs, "labels": labels, "preds": preds}

    # Full metrics at 60 min
    last = all_results[3600]
    metrics = compute_metrics(last["labels"], last["preds"], last["probs"])

    try:
        auroc_60 = roc_auc_score(last["labels"], last["probs"])
    except ValueError:
        auroc_60 = 0.5

    return {
        "fold": fold_idx,
        "holdout_r": fold["holdout_r"],
        "holdout_s": fold["holdout_s"],
        "n_train": len(fold["train"]),
        "n_val": len(fold["val"]),
        "n_test": len(fold["test"]),
        "accuracy_vs_time": accuracy_vs_time,
        "metrics_60min": asdict(metrics),
        "auroc_60min": auroc_60,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Strain-holdout cross-validation")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--features-dir", type=Path, default=None)
    parser.add_argument("--checkpoints-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = FullConfig()
    config.device = args.device
    if args.data_root:
        config.paths.data_root = args.data_root
    if args.features_dir:
        config.paths.features_dir = args.features_dir
    if args.checkpoints_dir:
        config.paths.checkpoints_dir = args.checkpoints_dir

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    torch.manual_seed(config.seed)

    # Build strain-grouped experiments
    groups = build_strain_grouped_experiments(
        config.paths.features_dir, config.paths.data_root
    )
    for label, strain_dict in groups.items():
        label_name = "Resistant" if label == 1 else "Susceptible"
        for ec, exps in sorted(strain_dict.items()):
            logger.info(f"  {label_name} {ec}: {len(exps)} experiments")

    # Generate folds
    folds = generate_folds(groups, n_holdout_per_class=2, n_folds=args.n_folds, seed=args.seed)

    # Train and evaluate each fold
    all_fold_results = []

    for i, fold in enumerate(folds):
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {i+1}/{len(folds)}")
        logger.info(f"{'='*60}")

        ckpt_path = train_fold(i, fold, config, ckpt_dir)
        result = evaluate_fold(i, fold, ckpt_path, config)
        all_fold_results.append(result)

        logger.info(f"  Fold {i} test AUROC@60min: {result['auroc_60min']:.4f}")
        for t in [60, 300, 900, 3600]:
            if t in result["accuracy_vs_time"]:
                logger.info(f"  Fold {i} accuracy@{t//60}min: {result['accuracy_vs_time'][t]:.4f}")

    # Aggregate results
    logger.info(f"\n{'='*60}")
    logger.info("AGGREGATE RESULTS (strain-holdout)")
    logger.info(f"{'='*60}")

    eval_times = [60, 120, 180, 300, 600, 900, 1800, 3600]
    agg_acc = {t: [] for t in eval_times}
    agg_auroc = []

    for r in all_fold_results:
        agg_auroc.append(r["auroc_60min"])
        for t in eval_times:
            if t in r["accuracy_vs_time"]:
                agg_acc[t].append(r["accuracy_vs_time"][t])

    logger.info(f"\nAUROC@60min: {np.mean(agg_auroc):.4f} +/- {np.std(agg_auroc):.4f}")
    logger.info(f"\nAccuracy vs time (mean +/- std across {len(folds)} folds):")
    for t in eval_times:
        vals = agg_acc[t]
        if vals:
            logger.info(f"  {t//60:3d} min: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # Save results
    # Convert numpy arrays to lists for JSON serialization
    serializable = []
    for r in all_fold_results:
        sr = dict(r)
        sr["accuracy_vs_time"] = {str(k): v for k, v in sr["accuracy_vs_time"].items()}
        serializable.append(sr)

    summary = {
        "n_folds": len(folds),
        "mean_auroc_60min": float(np.mean(agg_auroc)),
        "std_auroc_60min": float(np.std(agg_auroc)),
        "mean_accuracy_vs_time": {
            str(t): float(np.mean(agg_acc[t])) for t in eval_times if agg_acc[t]
        },
        "std_accuracy_vs_time": {
            str(t): float(np.std(agg_acc[t])) for t in eval_times if agg_acc[t]
        },
        "folds": serializable,
    }

    with open(output_dir / "strain_holdout_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_dir / 'strain_holdout_results.json'}")


if __name__ == "__main__":
    main()
