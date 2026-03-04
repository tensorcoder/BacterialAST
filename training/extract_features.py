"""Stage 3: Extract ViT features from all bacteria crops using pretrained backbone."""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..models.backbone import ViTSmall

logger = logging.getLogger(__name__)


class HDF5InferenceDataset(Dataset):
    """Loads all crops from a single HDF5 file for feature extraction."""

    def __init__(self, h5_path: Path, mean: float = 0.5, std: float = 0.25):
        self.h5_path = Path(h5_path)
        self.mean = mean
        self.std = std
        with h5py.File(self.h5_path, "r") as f:
            self.length = f["crops"].shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        with h5py.File(self.h5_path, "r") as f:
            crop = f["crops"][idx]  # (96, 96) uint8
        # Normalize
        tensor = torch.from_numpy(crop.astype(np.float32) / 255.0)
        tensor = (tensor - self.mean) / self.std
        return tensor.unsqueeze(0)  # (1, 96, 96)


def extract_features_for_experiment(
    backbone: ViTSmall,
    h5_path: Path,
    output_path: Path,
    batch_size: int = 512,
    num_workers: int = 4,
    device: torch.device = torch.device("cuda:0"),
) -> None:
    """Extract features for all crops in one experiment HDF5 file."""
    dataset = HDF5InferenceDataset(h5_path)
    if len(dataset) == 0:
        logger.warning(f"Empty HDF5: {h5_path}")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Load metadata from HDF5
    with h5py.File(h5_path, "r") as f:
        metadata = f["metadata"][:]

    all_features = []
    backbone.eval()

    with torch.no_grad(), torch.amp.autocast("cuda"):
        for batch in dataloader:
            batch = batch.to(device, non_blocking=True)
            features = backbone(batch)  # (B, 384)
            all_features.append(features.cpu().half().numpy())

    features = np.concatenate(all_features, axis=0)  # (N, 384) float16

    # Save as npz
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=features,
        frame_idx=metadata["frame_idx"],
        track_id=metadata["track_id"] if "track_id" in metadata.dtype.names else np.zeros(len(metadata), dtype=np.int32),
        detection_id=metadata["detection_id"],
        cx=metadata["cx"],
        cy=metadata["cy"],
        w=metadata["w"],
        h=metadata["h"],
        angle=metadata["angle"],
        confidence=metadata["confidence"],
    )
    logger.debug(f"Saved {len(features)} features to {output_path}")


def extract_all_features(
    backbone_checkpoint: Path,
    preprocessed_dir: Path,
    output_dir: Path,
    batch_size: int = 512,
    num_workers: int = 4,
    device_str: str = "cuda:0",
    embed_dim: int = 384,
    img_size: int = 96,
    patch_size: int = 16,
    depth: int = 12,
    num_heads: int = 6,
) -> None:
    """Extract features for all experiments using pretrained backbone."""
    device = torch.device(device_str)

    # Load backbone
    backbone = ViTSmall(
        img_size=img_size,
        in_channels=1,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
    ).to(device)

    checkpoint = torch.load(backbone_checkpoint, map_location=device, weights_only=True)
    if "student_state_dict" in checkpoint:
        backbone.load_state_dict(checkpoint["student_state_dict"])
    else:
        backbone.load_state_dict(checkpoint)
    backbone.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all HDF5 files
    h5_files = sorted(Path(preprocessed_dir).glob("**/*.h5"))
    logger.info(f"Found {len(h5_files)} experiment HDF5 files")

    for h5_path in tqdm(h5_files, desc="Extracting features"):
        exp_id = h5_path.stem
        output_path = output_dir / f"{exp_id}.npz"

        if output_path.exists():
            logger.debug(f"Skipping {exp_id} (already extracted)")
            continue

        extract_features_for_experiment(
            backbone=backbone,
            h5_path=h5_path,
            output_path=output_path,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )

    logger.info(f"Feature extraction complete. Output: {output_dir}")
