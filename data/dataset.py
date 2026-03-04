"""Dataset classes for all training stages."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class DINOCropDataset(Dataset):
    """Loads bacteria crops from HDF5 files for DINO self-supervised pretraining.

    Samples uniformly across experiments to prevent dominant experiments
    from biasing the learned representation.
    """

    def __init__(
        self,
        hdf5_dir: Path,
        max_crops_per_experiment: int = 5000,
        transform: Optional[Callable] = None,
    ):
        self.hdf5_dir = Path(hdf5_dir)
        self.transform = transform
        self.max_crops = max_crops_per_experiment

        # Build index: list of (hdf5_path, crop_index)
        self.index: list[tuple[Path, int]] = []
        self._build_index()

    def _build_index(self) -> None:
        hdf5_files = sorted(self.hdf5_dir.glob("**/*.h5"))
        for h5_path in hdf5_files:
            with h5py.File(h5_path, "r") as f:
                n_crops = f["crops"].shape[0]
            if n_crops == 0:
                continue
            # Cap per experiment and sample uniformly
            indices = list(range(n_crops))
            if n_crops > self.max_crops:
                indices = sorted(random.sample(indices, self.max_crops))
            for idx in indices:
                self.index.append((h5_path, idx))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        h5_path, crop_idx = self.index[idx]
        with h5py.File(h5_path, "r") as f:
            crop = f["crops"][crop_idx]  # (96, 96) uint8

        image = Image.fromarray(crop, mode="L")

        if self.transform is not None:
            global_crops, local_crops = self.transform(image)
            return global_crops, local_crops

        # Fallback: return single tensor
        arr = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, 96, 96)
        return [tensor, tensor], []


class TemporalPairDataset(Dataset):
    """Yields pairs of the same tracked bacterium at different timepoints
    for temporal contrastive learning (DynaCLR-style).
    """

    def __init__(
        self,
        hdf5_dir: Path,
        track_csv_dir: Path,
        tau_range: tuple[int, int] = (5, 150),
        transform: Optional[Callable] = None,
        max_pairs_per_experiment: int = 2000,
    ):
        self.hdf5_dir = Path(hdf5_dir)
        self.tau_range = tau_range
        self.transform = transform

        # Build index of valid pairs
        self.pairs: list[tuple[Path, int, int]] = []  # (h5_path, idx_t, idx_t_tau)
        self._build_pairs(track_csv_dir, max_pairs_per_experiment)

    def _build_pairs(
        self, track_csv_dir: Path, max_pairs: int
    ) -> None:
        track_csvs = sorted(Path(track_csv_dir).glob("**/*.csv"))
        for csv_path in track_csvs:
            df = pd.read_csv(csv_path)
            if "track_id" not in df.columns:
                continue

            h5_name = csv_path.stem + ".h5"
            h5_path = self.hdf5_dir / h5_name
            if not h5_path.exists():
                # Try finding it in subdirectories
                matches = list(self.hdf5_dir.glob(f"**/{h5_name}"))
                if not matches:
                    continue
                h5_path = matches[0]

            experiment_pairs = []
            for track_id, group in df.groupby("track_id"):
                group = group.sort_values("frame_idx").reset_index(drop=True)
                if len(group) < self.tau_range[0] + 1:
                    continue

                # Sample pairs from this track
                for _ in range(min(20, len(group))):
                    t_idx = random.randint(0, len(group) - self.tau_range[0] - 1)
                    tau = random.randint(
                        self.tau_range[0],
                        min(self.tau_range[1], len(group) - t_idx - 1),
                    )
                    # These are row indices in the dataframe, map to HDF5 indices
                    # The detection_id column maps to HDF5 row index
                    idx_t = group.iloc[t_idx].name  # original df index
                    idx_t_tau = group.iloc[t_idx + tau].name
                    experiment_pairs.append((h5_path, int(idx_t), int(idx_t_tau)))

            if len(experiment_pairs) > max_pairs:
                experiment_pairs = random.sample(experiment_pairs, max_pairs)
            self.pairs.extend(experiment_pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        h5_path, idx_t, idx_t_tau = self.pairs[idx]
        with h5py.File(h5_path, "r") as f:
            crop_t = f["crops"][idx_t]  # (96, 96) uint8
            crop_t_tau = f["crops"][idx_t_tau]

        img_t = Image.fromarray(crop_t, mode="L")
        img_tau = Image.fromarray(crop_t_tau, mode="L")

        if self.transform is not None:
            # Use a simple single-crop transform for contrastive pairs
            t_t = self.transform(img_t)
            t_tau = self.transform(img_tau)
            return t_t, t_tau

        # Fallback
        t_t = torch.from_numpy(crop_t.astype(np.float32) / 255.0).unsqueeze(0)
        t_tau = torch.from_numpy(crop_t_tau.astype(np.float32) / 255.0).unsqueeze(0)
        return t_t, t_tau


@dataclass
class ExperimentMeta:
    experiment_id: str
    label: int  # 0=susceptible, 1=resistant
    features_path: Path
    antibiotic_id: str = ""
    dosage: str = ""


class TemporalMILDataset(Dataset):
    """Main dataset for Stage 4: Temporal MIL classifier training.

    Each sample is one (experiment, time_window) pair containing
    pre-extracted feature sequences for all tracked bacteria.
    """

    def __init__(
        self,
        feature_dir: Path,
        experiments: list[ExperimentMeta],
        time_windows_sec: list[float] | None = None,
        time_window_weights: list[float] | None = None,
        max_tracks: int = 64,
        max_frames_per_track: int = 512,
        frame_subsample_rate: int = 5,
        feature_dim: int = 384,
        fps: float = 5.0,
        random_window: bool = True,
    ):
        self.feature_dir = Path(feature_dir)
        self.experiments = experiments
        self.max_tracks = max_tracks
        self.max_frames = max_frames_per_track
        self.subsample_rate = frame_subsample_rate
        self.feature_dim = feature_dim
        self.fps = fps
        self.random_window = random_window

        if time_windows_sec is None:
            self.time_windows = [60, 120, 180, 300, 600, 900, 1800, 3600]
        else:
            self.time_windows = time_windows_sec

        if time_window_weights is None:
            self.window_weights = [0.20, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05, 0.05]
        else:
            self.window_weights = time_window_weights

        # Normalize weights
        total = sum(self.window_weights)
        self.window_weights = [w / total for w in self.window_weights]

        if random_window:
            # During training: one sample per experiment, window sampled randomly
            self.samples = [(exp, None) for exp in self.experiments]
        else:
            # During eval: all windows for each experiment
            self.samples = [
                (exp, tw) for exp in self.experiments for tw in self.time_windows
            ]

        # Cache loaded features
        self._cache: dict[str, dict] = {}

    def _load_features(self, exp: ExperimentMeta) -> dict:
        if exp.experiment_id in self._cache:
            return self._cache[exp.experiment_id]

        npz_path = self.feature_dir / f"{exp.experiment_id}.npz"
        if not npz_path.exists():
            # Try with path from experiment meta
            npz_path = exp.features_path

        data = np.load(npz_path)
        result = {
            "features": data["features"],  # (N, 384) float16
            "frame_idx": data["frame_idx"],  # (N,) int32
            "track_id": data["track_id"],  # (N,) int32
        }
        self._cache[exp.experiment_id] = result
        return result

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        exp, fixed_window = self.samples[idx]
        data = self._load_features(exp)

        # Select time window
        if fixed_window is not None:
            window_sec = fixed_window
        else:
            window_sec = random.choices(
                self.time_windows, weights=self.window_weights, k=1
            )[0]

        max_frame = int(window_sec * self.fps)

        # Filter to frames within window
        mask = data["frame_idx"] <= max_frame
        features = data["features"][mask]
        frame_idx = data["frame_idx"][mask]
        track_ids = data["track_id"][mask]

        # Get unique tracks
        unique_tracks = np.unique(track_ids)

        # Limit number of tracks (sample if too many)
        if len(unique_tracks) > self.max_tracks:
            unique_tracks = np.sort(
                np.random.choice(unique_tracks, self.max_tracks, replace=False)
            )

        n_tracks = min(len(unique_tracks), self.max_tracks)

        # Build per-track feature sequences
        track_features = torch.zeros(
            self.max_tracks, self.max_frames, self.feature_dim
        )
        track_mask = torch.zeros(self.max_tracks, dtype=torch.bool)
        seq_mask = torch.zeros(self.max_tracks, self.max_frames, dtype=torch.bool)

        for i, tid in enumerate(unique_tracks):
            track_mask[i] = True
            t_mask = track_ids == tid
            t_features = features[t_mask].astype(np.float32)
            t_frames = frame_idx[t_mask]

            # Sort by frame
            sort_idx = np.argsort(t_frames)
            t_features = t_features[sort_idx]
            t_frames = t_frames[sort_idx]

            # Subsample
            if self.subsample_rate > 1:
                sub_idx = np.arange(0, len(t_features), self.subsample_rate)
                t_features = t_features[sub_idx]
                t_frames = t_frames[sub_idx]

            # Truncate to max_frames
            seq_len = min(len(t_features), self.max_frames)
            track_features[i, :seq_len] = torch.from_numpy(t_features[:seq_len])
            seq_mask[i, :seq_len] = True

        # Time fraction for the classifier
        max_possible_frame = int(3600 * self.fps)
        time_fraction = min(max_frame / max_possible_frame, 1.0)

        return {
            "track_features": track_features,  # (max_tracks, max_frames, 384)
            "track_mask": track_mask,  # (max_tracks,)
            "seq_mask": seq_mask,  # (max_tracks, max_frames)
            "time_fraction": torch.tensor(time_fraction, dtype=torch.float32),
            "label": torch.tensor(exp.label, dtype=torch.long),
            "experiment_id": exp.experiment_id,
            "window_sec": window_sec,
        }


def build_experiment_list(
    data_root: Path,
    preprocessed_dir: Path,
) -> list[ExperimentMeta]:
    """Scan the data directory structure and build experiment metadata list."""
    experiments = []
    data_root = Path(data_root)

    for label_name, label_int in [("Susceptible", 0), ("Resistant", 1)]:
        label_dir = data_root / label_name
        if not label_dir.exists():
            continue

        for exp_dir in sorted(label_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            exp_id = exp_dir.name
            # Parse experiment folder name: BacteriaID_AntibioticID_Dosage_ExpID
            parts = exp_id.split("_")
            antibiotic_id = parts[1] if len(parts) > 1 else ""
            dosage = parts[2] if len(parts) > 2 else ""

            features_path = Path(preprocessed_dir) / f"{exp_id}.npz"

            experiments.append(
                ExperimentMeta(
                    experiment_id=exp_id,
                    label=label_int,
                    features_path=features_path,
                    antibiotic_id=antibiotic_id,
                    dosage=dosage,
                )
            )

    return experiments


def create_splits(
    experiments: list[ExperimentMeta],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[ExperimentMeta], list[ExperimentMeta], list[ExperimentMeta]]:
    """Stratified split by experiment, ensuring all frames from one experiment
    go to the same split. Stratified by (label, antibiotic_id).
    """
    rng = np.random.RandomState(seed)

    # Group by (label, antibiotic_id) for stratification
    groups: dict[tuple, list[ExperimentMeta]] = {}
    for exp in experiments:
        key = (exp.label, exp.antibiotic_id)
        groups.setdefault(key, []).append(exp)

    train, val, test = [], [], []

    for key, exps in groups.items():
        rng.shuffle(exps)
        n = len(exps)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        train.extend(exps[:n_train])
        val.extend(exps[n_train : n_train + n_val])
        test.extend(exps[n_train + n_val :])

    return train, val, test
