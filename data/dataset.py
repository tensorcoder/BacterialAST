"""Dataset classes for all training stages."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np
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

        arr = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, 96, 96)
        return [tensor, tensor], []


@dataclass
class ExperimentMeta:
    experiment_id: str
    label: int  # 0=susceptible, 1=resistant
    features_path: Path


class PopulationTemporalDataset(Dataset):
    """Main dataset for classifier training using population-level temporal analysis.

    Each sample is one (experiment, time_window) pair. For the selected time
    window, crops are binned by timestamp into fixed-width time bins.  Each bin
    contains the pre-extracted backbone features for all focused bacteria
    detected in that time interval.

    The model receives per-bin feature tensors and learns how the population
    distribution shifts over the course of the experiment.
    """

    def __init__(
        self,
        feature_dir: Path,
        experiments: list[ExperimentMeta],
        time_bin_width_sec: float = 120.0,
        time_windows_sec: list[float] | None = None,
        time_window_weights: list[float] | None = None,
        max_crops_per_bin: int = 256,
        feature_dim: int = 384,
        max_experiment_sec: float = 3600.0,
        random_window: bool = True,
        samples_per_experiment: int = 1,
    ):
        self.feature_dir = Path(feature_dir)
        self.experiments = experiments
        self.time_bin_width = time_bin_width_sec
        self.max_crops_per_bin = max_crops_per_bin
        self.feature_dim = feature_dim
        self.max_experiment_sec = max_experiment_sec
        self.random_window = random_window

        if time_windows_sec is None:
            self.time_windows = [60, 120, 180, 300, 600, 900, 1800, 3600]
        else:
            self.time_windows = time_windows_sec

        if time_window_weights is None:
            self.window_weights = [0.20, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05, 0.05]
        else:
            self.window_weights = time_window_weights

        total = sum(self.window_weights)
        self.window_weights = [w / total for w in self.window_weights]

        if random_window:
            # Repeat each experiment N times per epoch — each gets a different
            # random window and random crop subsample, acting as augmentation.
            self.samples = [
                (exp, None)
                for exp in self.experiments
                for _ in range(samples_per_experiment)
            ]
        else:
            self.samples = [
                (exp, tw) for exp in self.experiments for tw in self.time_windows
            ]

        self._cache: dict[str, dict] = {}

    def _load_features(self, exp: ExperimentMeta) -> dict:
        if exp.experiment_id in self._cache:
            return self._cache[exp.experiment_id]

        npz_path = self.feature_dir / f"{exp.experiment_id}.npz"
        if not npz_path.exists():
            npz_path = exp.features_path

        data = np.load(npz_path)
        # Timestamps are relative to experiment start (seconds from first frame)
        timestamps = data["timestamps"].astype(np.float64)
        t_start = timestamps.min()
        relative_timestamps = timestamps - t_start

        result = {
            "features": data["features"],  # (N, 384) float16
            "timestamps": relative_timestamps,  # (N,) float64, seconds from start
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

        # Filter to crops within the time window
        timestamps = data["timestamps"]
        mask = timestamps <= window_sec
        features = data["features"][mask].astype(np.float32)
        ts = timestamps[mask]

        # Compute time bins
        n_bins = max(1, int(np.ceil(window_sec / self.time_bin_width)))

        # Build per-bin feature tensors
        bin_features = torch.zeros(n_bins, self.max_crops_per_bin, self.feature_dim)
        bin_mask = torch.zeros(n_bins, dtype=torch.bool)
        crop_mask = torch.zeros(n_bins, self.max_crops_per_bin, dtype=torch.bool)
        bin_times = torch.zeros(n_bins, dtype=torch.float32)
        bin_counts = torch.zeros(n_bins, dtype=torch.float32)

        for b in range(n_bins):
            t_lo = b * self.time_bin_width
            t_hi = (b + 1) * self.time_bin_width
            bin_center = (t_lo + t_hi) / 2.0
            bin_times[b] = bin_center

            # Select crops in this bin
            in_bin = (ts >= t_lo) & (ts < t_hi)
            bin_feats = features[in_bin]

            if len(bin_feats) == 0:
                continue

            bin_mask[b] = True
            bin_counts[b] = len(bin_feats)

            # Subsample if too many crops
            if len(bin_feats) > self.max_crops_per_bin:
                indices = np.random.choice(
                    len(bin_feats), self.max_crops_per_bin, replace=False
                )
                bin_feats = bin_feats[indices]

            n_crops = len(bin_feats)
            bin_features[b, :n_crops] = torch.from_numpy(bin_feats)
            crop_mask[b, :n_crops] = True

        time_fraction = min(window_sec / self.max_experiment_sec, 1.0)

        return {
            "bin_features": bin_features,  # (n_bins, max_crops_per_bin, 384)
            "bin_mask": bin_mask,  # (n_bins,) — which bins have data
            "crop_mask": crop_mask,  # (n_bins, max_crops_per_bin)
            "bin_times": bin_times,  # (n_bins,) — bin center times in seconds
            "bin_counts": bin_counts,  # (n_bins,) — number of crops per bin
            "time_fraction": torch.tensor(time_fraction, dtype=torch.float32),
            "label": torch.tensor(exp.label, dtype=torch.long),
            "experiment_id": exp.experiment_id,
            "window_sec": window_sec,
        }


def population_temporal_collate(batch: list[dict]) -> dict:
    """Custom collate that pads variable-length time bin dimensions.

    Different samples may have different numbers of time bins (due to
    different time windows).  This function pads all bin-dimension tensors
    to the maximum number of bins in the batch.  Padded bins have
    bin_mask=False and are ignored by the model at every stage.
    """
    max_bins = max(item["bin_features"].shape[0] for item in batch)
    max_crops = batch[0]["bin_features"].shape[1]
    feat_dim = batch[0]["bin_features"].shape[2]

    B = len(batch)
    bin_features = torch.zeros(B, max_bins, max_crops, feat_dim)
    bin_mask = torch.zeros(B, max_bins, dtype=torch.bool)
    crop_mask = torch.zeros(B, max_bins, max_crops, dtype=torch.bool)
    bin_times = torch.zeros(B, max_bins, dtype=torch.float32)
    bin_counts = torch.zeros(B, max_bins, dtype=torch.float32)
    time_fraction = torch.zeros(B, dtype=torch.float32)
    labels = torch.zeros(B, dtype=torch.long)
    experiment_ids = []
    window_secs = []

    for i, item in enumerate(batch):
        n = item["bin_features"].shape[0]
        bin_features[i, :n] = item["bin_features"]
        bin_mask[i, :n] = item["bin_mask"]
        crop_mask[i, :n] = item["crop_mask"]
        bin_times[i, :n] = item["bin_times"]
        bin_counts[i, :n] = item["bin_counts"]
        time_fraction[i] = item["time_fraction"]
        labels[i] = item["label"]
        experiment_ids.append(item["experiment_id"])
        window_secs.append(item["window_sec"])

    return {
        "bin_features": bin_features,
        "bin_mask": bin_mask,
        "crop_mask": crop_mask,
        "bin_times": bin_times,
        "bin_counts": bin_counts,
        "time_fraction": time_fraction,
        "label": labels,
        "experiment_id": experiment_ids,
        "window_sec": window_secs,
    }


def _extract_ec_number(experiment_id: str) -> str | None:
    """Extract EC number prefix from experiment ID.

    Examples::

        EC35_Ampicillin_16mgL_preincubated_2_TEM40  →  EC35
        EC126_Ampicillin_16mgL_preincubated          →  EC126

    Returns None if no EC prefix is found.
    """
    import re
    m = re.match(r"^(EC\d+)", experiment_id, re.IGNORECASE)
    return m.group(1).upper() if m else None


def _find_label_dir(data_root: Path, name: str) -> Path | None:
    """Find a subdirectory case-insensitively."""
    for d in data_root.iterdir():
        if d.is_dir() and d.name.lower() == name.lower():
            return d
    return None


def build_experiment_list(
    data_root: Path,
    features_dir: Path,
) -> list[ExperimentMeta]:
    """Scan the data directory structure and build experiment metadata list.

    Expected structure::

        data_root/
        ├── Resistant/
        │   ├── EC35_Ampicillin_.../
        │   │   └── images/
        │   └── ...
        ├── Susceptible/
        │   ├── EC126_Ampicillin_.../
        │   │   └── images/
        │   └── ...
        └── Test/
            ├── EC35_Ampicillin_.../   ← label inferred from EC number
            │   └── images/
            └── ...

    Labels for Test/ experiments are inferred from EC number: if EC35 appears
    in Resistant/, then all EC35 experiments are resistant.
    """
    experiments = []
    data_root = Path(data_root)
    features_dir = Path(features_dir)

    # Step 1: Build EC-number → label mapping from Resistant/Susceptible folders
    ec_label_map: dict[str, int] = {}

    for label_name, label_int in [("susceptible", 0), ("resistant", 1)]:
        label_dir = _find_label_dir(data_root, label_name)
        if label_dir is None:
            continue

        for exp_dir in sorted(label_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            images_dir = exp_dir / "images"
            if not images_dir.exists():
                continue

            exp_id = exp_dir.name
            ec_num = _extract_ec_number(exp_id)
            if ec_num is not None:
                ec_label_map[ec_num] = label_int

            experiments.append(
                ExperimentMeta(
                    experiment_id=exp_id,
                    label=label_int,
                    features_path=features_dir / f"{exp_id}.npz",
                )
            )

    # Step 2: Add Test/ experiments with labels inferred from EC number
    test_dir = _find_label_dir(data_root, "test")
    if test_dir is not None:
        for exp_dir in sorted(test_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            images_dir = exp_dir / "images"
            if not images_dir.exists():
                continue

            exp_id = exp_dir.name
            ec_num = _extract_ec_number(exp_id)
            if ec_num is None or ec_num not in ec_label_map:
                import logging
                logging.getLogger(__name__).warning(
                    f"Cannot determine label for Test experiment {exp_id} "
                    f"(EC number: {ec_num}), skipping."
                )
                continue

            label_int = ec_label_map[ec_num]
            experiments.append(
                ExperimentMeta(
                    experiment_id=exp_id,
                    label=label_int,
                    features_path=features_dir / f"{exp_id}.npz",
                )
            )

    return experiments


def create_splits(
    experiments: list[ExperimentMeta],
    data_root: Path | None = None,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[ExperimentMeta], list[ExperimentMeta], list[ExperimentMeta]]:
    """Split experiments into train/val/test.

    If ``data_root`` is provided, experiments from the Test/ folder become
    the test set and the Resistant/Susceptible experiments are split into
    train/val.  Otherwise, a stratified random split is used.
    """
    rng = np.random.RandomState(seed)

    if data_root is not None:
        # Determine which experiments came from Test/ folder
        test_dir = _find_label_dir(Path(data_root), "test")
        test_exp_ids: set[str] = set()
        if test_dir is not None:
            for d in test_dir.iterdir():
                if d.is_dir():
                    test_exp_ids.add(d.name)

        test = [e for e in experiments if e.experiment_id in test_exp_ids]
        train_val = [e for e in experiments if e.experiment_id not in test_exp_ids]
    else:
        train_val = experiments
        test = []

    # Stratified train/val split
    groups: dict[int, list[ExperimentMeta]] = {}
    for exp in train_val:
        groups.setdefault(exp.label, []).append(exp)

    train, val = [], []
    for label, exps in groups.items():
        rng.shuffle(exps)
        n = len(exps)
        n_val = max(1, int(n * val_ratio))
        val.extend(exps[:n_val])
        train.extend(exps[n_val:])

    return train, val, test
