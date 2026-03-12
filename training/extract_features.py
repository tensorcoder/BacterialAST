"""Stage 3: Extract ViT features from all bacteria crops using pretrained backbone."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..models.backbone import ViTSmall

logger = logging.getLogger(__name__)


class HDF5InferenceDataset(Dataset):
    """Loads all crops from a single HDF5 file for feature extraction.

    Also returns relative timestamps (seconds since experiment start)
    for time-conditioned backbones.
    """

    _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __init__(
        self,
        h5_path: Path,
        mean: float = 0.3387,
        std: float = 0.1173,
        use_clahe: bool = True,
    ):
        self.h5_path = Path(h5_path)
        self.mean = mean
        self.std = std
        self.use_clahe = use_clahe
        with h5py.File(self.h5_path, "r") as f:
            self.length = f["crops"].shape[0]
            timestamps = f["metadata"]["timestamp"][:]
            t_start = float(timestamps.min())
            self._rel_times = (timestamps - t_start).astype(np.float32)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        with h5py.File(self.h5_path, "r") as f:
            crop = f["crops"][idx]  # (96, 96) uint8
        if self.use_clahe:
            crop = self._clahe.apply(crop)
        tensor = torch.from_numpy(crop.astype(np.float32) / 255.0)
        tensor = (tensor - self.mean) / self.std
        time_sec = torch.tensor(self._rel_times[idx], dtype=torch.float32)
        return tensor.unsqueeze(0), time_sec  # (1, 96, 96), scalar


def extract_features_for_experiment(
    backbone: ViTSmall,
    h5_path: Path,
    output_path: Path,
    batch_size: int = 512,
    num_workers: int = 4,
    device: torch.device = torch.device("cuda:0"),
    mean: float = 0.3387,
    std: float = 0.1173,
    use_clahe: bool = True,
) -> None:
    """Extract features for all crops in one experiment HDF5 file."""
    dataset = HDF5InferenceDataset(h5_path, mean=mean, std=std, use_clahe=use_clahe)
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
    time_conditioned = getattr(backbone, "time_conditioned", False)

    with torch.no_grad(), torch.amp.autocast("cuda"):
        for crops, times in dataloader:
            crops = crops.to(device, non_blocking=True)
            if time_conditioned:
                times = times.to(device, non_blocking=True)
                features = backbone(crops, time=times)  # (B, 384)
            else:
                features = backbone(crops)  # (B, 384)
            all_features.append(features.cpu().half().numpy())

    features = np.concatenate(all_features, axis=0)  # (N, 384) float16

    # Save with timestamps (not frame_idx or track_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=features,
        timestamps=metadata["timestamp"],
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
    dataset_mean: float = 0.3387,
    dataset_std: float = 0.1173,
    time_conditioned: bool = True,
) -> None:
    """Extract features for all experiments using pretrained backbone."""
    device = torch.device(device_str)

    backbone = ViTSmall(
        img_size=img_size,
        in_channels=1,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        time_conditioned=time_conditioned,
    ).to(device)

    checkpoint = torch.load(backbone_checkpoint, map_location=device, weights_only=False)
    if "student_state_dict" in checkpoint:
        backbone.load_state_dict(checkpoint["student_state_dict"])
    else:
        backbone.load_state_dict(checkpoint)
    backbone.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
            mean=dataset_mean,
            std=dataset_std,
        )

    logger.info(f"Feature extraction complete. Output: {output_dir}")
