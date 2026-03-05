"""Centralized configuration for the AST classifier pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PathConfig:
    data_root: Path = Path("/mnt/f/Data_second_protocol")
    preprocessed_dir: Path = Path("./preprocessed")
    features_dir: Path = Path("./features")
    checkpoints_dir: Path = Path("./checkpoints")
    logs_dir: Path = Path("./logs")
    yolo_weights: Path = Path("/mnt/c/users/mkedz/Documents/PhD/PhD_code/yolo11/vertical_obb_100epo_best.pt")


@dataclass
class PreprocessingConfig:
    yolo_confidence: float = 0.5
    crop_size: int = 128
    yolo_batch_size: int = 16
    focused_class_name: str = "focused"  # YOLO class name for in-focus bacteria
    fps: float = 5.0


@dataclass
class DINOConfig:
    # Architecture
    img_size: int = 128
    patch_size: int = 16
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.1
    # DINO head
    head_hidden_dim: int = 2048
    head_bottleneck_dim: int = 256
    head_output_dim: int = 65536
    head_nlayers: int = 3
    # Training
    batch_size: int = 64
    epochs: int = 100
    base_lr: float = 5e-4
    min_lr: float = 1e-6
    weight_decay_start: float = 0.04
    weight_decay_end: float = 0.4
    warmup_epochs: int = 10
    grad_clip: float = 3.0
    # EMA
    ema_momentum_start: float = 0.996
    ema_momentum_end: float = 1.0
    # Temperature
    teacher_temp_start: float = 0.04
    teacher_temp_end: float = 0.07
    teacher_temp_warmup_epochs: int = 30
    student_temp: float = 0.1
    center_momentum: float = 0.9
    # Multi-crop
    n_global_crops: int = 2
    n_local_crops: int = 6
    global_crop_scale: tuple = (0.7, 1.0)
    local_crop_scale: tuple = (0.3, 0.6)
    local_crop_size: int = 64
    # Dataset
    max_crops_per_experiment: int = 5000


@dataclass
class ClassifierConfig:
    # Architecture
    feature_dim: int = 384
    temporal_hidden_dim: int = 256
    temporal_num_layers: int = 4
    temporal_num_heads: int = 4
    temporal_ffn_dim: int = 512
    population_feat_dim: int = 64
    classifier_hidden_dim: int = 128
    num_classes: int = 2
    dropout: float = 0.1
    # Population binning
    time_bin_width_sec: float = 120.0  # 2-minute bins (configurable)
    max_crops_per_bin: int = 256
    # Training
    batch_size: int = 16
    gradient_accumulation: int = 2
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_epochs: int = 10
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    early_stopping_patience: int = 30
    # Time-aware loss
    time_loss_alpha: float = 2.0
    attention_entropy_weight: float = 0.01
    # Time windows (seconds) for evaluation
    time_windows: list = field(
        default_factory=lambda: [60, 120, 180, 300, 600, 900, 1800, 3600]
    )
    time_window_weights: list = field(
        default_factory=lambda: [0.20, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05, 0.05]
    )


@dataclass
class EarlyExitConfig:
    patience: int = 3
    confidence_threshold: float = 0.85
    eval_interval_sec: int = 30
    min_time_sec: int = 60
    max_time_sec: int = 3600
    # Calibration sweep
    patience_range: list = field(default_factory=lambda: [1, 2, 3, 5, 8])
    threshold_range: list = field(
        default_factory=lambda: [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    )
    # Learned halting (optional)
    use_learned_halting: bool = False
    halting_hidden_dim: int = 32
    halting_lambda: float = 0.1


@dataclass
class DataSplitConfig:
    val_ratio: float = 0.15  # fraction of R/S experiments held out for validation
    random_seed: int = 42
    # Test set comes from the Test/ folder (pre-defined split)


@dataclass
class FullConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    dino: DINOConfig = field(default_factory=DINOConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    early_exit: EarlyExitConfig = field(default_factory=EarlyExitConfig)
    data_split: DataSplitConfig = field(default_factory=DataSplitConfig)
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda:0"
