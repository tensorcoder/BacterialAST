# AST Classifier: Population-Level Temporal Analysis for Rapid Antimicrobial Susceptibility Testing

A deep learning pipeline that distinguishes antibiotic-resistant from susceptible *E. coli* bacteria using time-lapse phase-contrast microscopy. The system analyses how the **population-level morphological distribution** of in-focus bacteria shifts over the course of a 1-hour experiment after antibiotic exposure, enabling classification potentially within minutes rather than the traditional 16-24 hours required by culture-based methods.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Design Choices and Justifications](#design-choices-and-justifications)
- [Installation](#installation)
- [Data Layout](#data-layout)
- [Usage](#usage)
  - [Stage 1: Preprocessing](#stage-1-preprocessing)
  - [Stage 2: DINO Pretraining](#stage-2-dino-pretraining)
  - [Stage 3: Feature Extraction](#stage-3-feature-extraction)
  - [Stage 4: Classifier Training](#stage-4-classifier-training)
  - [Stage 5: Early-Exit Calibration](#stage-5-early-exit-calibration)
  - [Evaluation](#evaluation)
  - [Full Pipeline](#full-pipeline)
- [Configuration Reference](#configuration-reference)
- [Model Architecture Details](#model-architecture-details)
- [Augmentation Strategy](#augmentation-strategy)
- [Loss Functions](#loss-functions)
- [Metrics and Evaluation](#metrics-and-evaluation)
- [Visualization](#visualization)
- [Project Structure](#project-structure)
- [Hardware Requirements](#hardware-requirements)
- [Literature and References](#literature-and-references)

---

## Overview

### The Problem

When a patient presents with a bacterial infection, clinicians must determine which antibiotics the bacteria are susceptible to. Traditional antimicrobial susceptibility testing (AST) requires growing bacterial cultures for 16-24 hours -- time during which patients receive empirical (potentially inappropriate) antibiotic therapy. This delay contributes to treatment failures and the broader crisis of antimicrobial resistance.

### The Approach

This system uses a finetuned YOLOv11-OBB model to detect bacteria in 100x phase-contrast microscopy images, classifying each detection as **focused**, **unfocused**, or **vertical**. Only focused detections are retained. Rather than tracking individual bacteria (which is infeasible at 5 FPS due to flow speed), the pipeline analyses how the **population distribution** of bacteria morphology changes over the 1-hour experiment. After antibiotic exposure, susceptible populations exhibit increasing morphological heterogeneity (elongation, blebbing, lysis, growth arrest) while resistant populations remain homogeneous.

The pipeline operates in five stages:

```
Stage 1: Detection + Cropping      Raw BMP frames --> focused bacteria crops with timestamps
Stage 2: DINO Pretraining           Crops --> self-supervised ViT-Small backbone
Stage 3: Feature Extraction         Crops --> cached 384-dim feature vectors
Stage 4: Population Temporal        Time-binned features --> resistant/susceptible prediction
         Classifier
Stage 5: Early-Exit Calibration     Validation set --> optimal halting thresholds
```

### Why Population-Level Rather Than Individual Tracking?

Individual bacterium tracking is not feasible in this setup for two reasons:

1. **Flow speed + 5 FPS framerate:** Bacteria in the microfluidic channel move too far between consecutive frames for IoU-based tracking to link detections reliably.
2. **Short visibility (~2 seconds):** Each bacterium passes through the field of view in approximately 10 frames -- far too brief to observe meaningful morphological change in a single individual.

However, the **population** changes dramatically over a 1-hour experiment. Susceptible bacteria respond to the antibiotic, altering their morphological distribution. The key signal is not how one bacterium changes, but how the statistical properties of the entire population shift over time.

---

## Architecture

```
                               Raw BMP Frames (1280x1024, grayscale, 5fps)
                                              |
                                    YOLOv11-OBB Detection
                                    (focused / unfocused / vertical)
                                              |
                                    Filter: keep "focused" only
                                              |
                              Oriented Bounding Box Crop Extraction
                              (affine rectification, size-preserving)
                                              |
                                    128x128 Grayscale Crops
                          (native pixel size, padded to square canvas)
                                   + timestamp from filename
                                              |
                            ViT-Small Backbone (DINO pretrained)
                                      CLS token: 384-dim
                                              |
                        Bin crops by time (configurable, default 2 min)
                                              |
                    +----------+---------+---------+---------+----------+
                    |          |         |         |         |          |
                Bin 0-2m   Bin 2-4m  Bin 4-6m  Bin 6-8m  ...    Bin 58-60m
               (N0, 384)  (N1, 384) (N2, 384) (N3, 384)       (Nk, 384)
                    |          |         |         |         |          |
                Population statistics per bin:
                mean, std, skewness, kurtosis + normalised count
                    |          |         |         |         |          |
                PopulationBinEncoder: MLP(4*384+1 --> 256)
                    |          |         |         |         |          |
                Per-bin embeddings (256-dim each)
                    |          |         |         |         |          |
                    +----------+---------+---------+---------+----------+
                                              |
                        Continuous-Time Positional Encoding
                       (sinusoidal encoding of bin centre times)
                                              |
                        4-layer Transformer Encoder (256-dim)
                                              |
                        Gated Attention over time bins
                      (which time periods matter most?)
                                   (B, 256)
                                      |
                               time_fraction (1-dim)
                                      |
                              Classifier Head
                       (MLP: 257 --> 128 --> 128 --> 2)
                                      |
                              logits (B, 2)
                                      |
                             Early-Exit Policy
                      (confidence + patience threshold)
                                      |
                        Resistant / Susceptible prediction
                        + time-to-prediction (seconds)
```

---

## Design Choices and Justifications

### 1. Self-Supervised Pretraining with DINO

**Choice:** Pretrain a Vision Transformer (ViT-Small) using DINO (self-distillation with no labels) before supervised fine-tuning.

**Justification:** With ~600K images containing many bacteria each, we have millions of unlabeled bacteria crops but only experiment-level labels (resistant vs susceptible). DINO learns morphological representations without labels by training a student network to match a momentum-averaged teacher network across different augmented views of the same image. This has been shown to outperform supervised pretraining for cell phenotyping tasks (Cell-DINO, PLOS Computational Biology 2025), and ViTs specifically outperform CNNs for bacterial classification from phase-contrast microscopy (Hallstrom et al., PLOS ONE 2025). The self-supervised features capture morphological structure that transfers well to downstream classification.

### 2. Vision Transformer over CNN

**Choice:** ViT-Small (384-dim, 12 layers, 6 heads) rather than ResNet or EfficientNet.

**Justification:** ViTs learn global attention patterns across the entire image from the first layer, while CNNs build local-to-global hierarchies. For bacteria at 100x magnification, subtle morphological features (membrane irregularities, nucleoid condensation, elongation) may appear at various spatial scales simultaneously. Hallstrom et al. (2025) directly compared ViT and ResNet architectures for bacterial species identification from microfluidic time-lapse microscopy and found ViT consistently outperformed ResNet with lower variance. The 128x128 input with 16x16 patches produces 64 tokens (8x8 grid), making the model efficient enough for a single RTX 3090.

### 3. Population-Level Temporal Analysis (Not Individual Tracking)

**Choice:** Bin all detected bacteria by time window and model the evolution of population-level statistics, rather than tracking individual bacteria.

**Justification:** In the microfluidic flow setup, bacteria travel through the field of view in approximately 2 seconds (~10 frames at 5 FPS). The flow speed makes IoU-based tracking between frames unreliable -- bacteria move too far between consecutive frames for bounding box overlap to provide meaningful associations. However, the population of bacteria observed at any given time is a sample from the overall population. As susceptible bacteria respond to the antibiotic, the statistical distribution of this population changes: morphological heterogeneity increases (some bacteria elongating, some blebbing, some lysing) while resistant populations remain homogeneous. By computing population statistics (mean, std, skewness, kurtosis) of the backbone features within configurable time bins (default: 2 minutes), we capture this distributional shift as a temporal signal without requiring individual tracking.

### 4. Size-Preserving Crop Extraction

**Choice:** Place each detected bacterium at its native pixel size on a fixed 128x128 canvas with reflected border fill, rather than resizing to fit the canvas.

**Justification:** Bacteria detected by the OBB model are rod-shaped with typical dimensions of 40-85 x 15-20 pixels (aspect ratios of 2.5:1 to 4.6:1). Naively resizing these rectangles to a square would destroy two critical morphological signals: (1) **absolute size** -- susceptible bacteria elongate under antibiotic exposure, becoming physically longer in pixels, and (2) **aspect ratio** -- the shape difference between a normal rod and an elongated filament. By centring each OBB crop on a 128x128 canvas at native resolution (with the surrounding area filled by reflected border to match the local background texture), both size and shape are preserved. A small bacterium (40x15px) occupies a small region of the canvas; an elongated one (85x20px) is visibly larger. The 99th percentile of detection sizes is 97px, so the 128px canvas accommodates virtually all bacteria without any rescaling. Only the rare <1% of detections exceeding 128px are downscaled to fit.

### 5. Continuous-Time Positional Encoding

**Choice:** Use sinusoidal positional encoding based on actual timestamps (seconds) rather than integer sequence positions.

**Justification:** Time bins have real-world temporal meaning -- the morphological changes at 5 minutes vs 30 minutes carry different biological significance. By encoding the actual bin centre time (in seconds) using sinusoidal functions, the temporal transformer can learn time-dependent patterns such as "susceptible populations typically show increased heterogeneity after 10-15 minutes of exposure." This is more meaningful than discrete position indices, especially when experiments may have slightly different durations or when evaluating at arbitrary time windows during early exit.

### 6. Gated Attention Over Time Bins

**Choice:** Use gated attention pooling (Ilse et al., 2018) over the temporal transformer outputs to aggregate across time bins.

**Justification:** Not all time bins are equally informative. The earliest bins (before antibiotic takes effect) and the latest bins (when changes are obvious) may be less discriminative than the critical transition period. Gated attention learns to weight each time bin by its informativeness, providing both an aggregated experiment representation and interpretable attention weights showing which time periods most influenced the classification. The gating mechanism (sigmoid gate multiplied with tanh activation) is more expressive than simple attention and can suppress uninformative time bins entirely.

### 7. Population Distribution Features

**Choice:** Compute statistical moments (mean, std, skewness, kurtosis) of the per-crop feature distribution within each time bin.

**Justification:** The *distribution* of morphological features across the population carries the key biological signal. Resistant populations remain homogeneous (low std, near-zero skewness) while susceptible populations become heterogeneous over time:
- **Std** increases as susceptible bacteria diverge in phenotype
- **Skewness** changes as the distribution becomes asymmetric (e.g., most cells elongating but some lysing)
- **Kurtosis** captures whether the distribution is concentrated (resistant, uniform response) or has heavy tails (susceptible, diverse responses)

The normalised crop count per bin also provides signal -- resistant bacteria continue dividing (increasing count) while susceptible populations may decrease.

### 8. Time-Aware Loss

**Choice:** Weight the cross-entropy loss by `lambda(t) = 1 + alpha * (1 - t/T_max)` where alpha=2.0, giving 3x weight to predictions at t=0 vs t=T_max.

**Justification:** The clinical value of a prediction decreases with time -- a correct prediction at 5 minutes is far more valuable than one at 55 minutes. By weighting early correct predictions more heavily, the model is incentivized to learn features that discriminate resistance/susceptibility as early as possible. Combined with multi-scale temporal window sampling (biased toward shorter windows during training), this drives the model to extract discriminative signal from the earliest time bins.

### 9. Confidence-Based Early Exit

**Choice:** At inference, evaluate every 30 seconds and halt when the model produces N consecutive predictions with softmax confidence above a threshold.

**Justification:** This patience-based mechanism prevents premature exit on noisy early predictions while allowing rapid exit once the model is consistently confident. The threshold and patience are calibrated on a validation set by sweeping over a grid of values and finding Pareto-optimal operating points that balance accuracy vs speed. Temperature scaling (Guo et al., 2017) ensures that the model's confidence scores are well-calibrated (i.e., a prediction with 90% confidence is correct ~90% of the time).

### 10. HDF5 Storage Instead of Individual Files

**Choice:** Store all crops from each experiment in a single HDF5 file rather than millions of individual image files.

**Justification:** With ~65K focused crops per experiment x ~42 experiments, the dataset contains approximately 2.7 million individual bacteria crops. Storing these as individual PNG/BMP files would create massive filesystem overhead (inode exhaustion, slow directory listing, degraded I/O performance). HDF5 provides chunked, optionally compressed storage with fast random access by index.

### 11. Offline Feature Caching

**Choice:** Extract ViT features for all crops once and cache them as .npz files, then train the temporal classifier entirely on pre-extracted features.

**Justification:** The ViT backbone has 21.7M parameters and processes 128x128 images -- running it during every training epoch of the classifier would be prohibitively expensive. By extracting features once (384-dim float16), the classifier training operates in feature-space rather than image-space. This makes the classifier training loop extremely fast (minutes per epoch vs hours) and allows extensive hyperparameter search.

### 12. Pre-Defined Test Split from Folder Structure

**Choice:** Use the `Test/` folder as a fixed held-out test set, with labels inferred from the EC number (strain identifier) matching against the `Resistant/` and `Susceptible/` folders.

**Justification:** The same bacterial strain (e.g., EC35) is always either resistant or susceptible -- this is an intrinsic property of the strain. By using the strain identifier to propagate labels from the labelled folders to the Test folder, we maintain a pre-defined test set that represents unseen experimental replicates of known strains. This avoids data leakage and ensures the model generalises across different experimental runs, not just different time windows of the same run.

---

## Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (tested on RTX 3090, 24GB VRAM)
- CUDA 12.4+ and cuDNN

### Setup

```bash
# Navigate to the project
cd /path/to/ast_classifier

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >=2.2.0 | Deep learning framework |
| `torchvision` | >=0.17.0 | Image transforms |
| `ultralytics` | >=8.1.0 | YOLOv11-OBB inference |
| `h5py` | >=3.10.0 | HDF5 data storage |
| `opencv-python` | >=4.9.0 | Image processing, affine transforms |
| `scikit-learn` | >=1.4.0 | Metrics, t-SNE, PCA |
| `numpy` | >=1.26.0 | Numerical operations |
| `scipy` | >=1.12.0 | Scientific computing |
| `matplotlib` | >=3.8.0 | Plotting |
| `seaborn` | >=0.13.0 | Statistical visualization |
| `tensorboard` | >=2.15.0 | Training monitoring |
| `tqdm` | >=4.66.0 | Progress bars |
| `Pillow` | >=10.2.0 | Image I/O |

### Verifying Installation

```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
PYTHONPATH=/path/to/parent python3 -c "from ast_classifier.config import FullConfig; print('Config OK')"
```

---

## Data Layout

The system expects the following directory structure:

```
Data_second_protocol/
├── Resistant/
│   ├── EC35_Ampicillin_16mgL_preincubated_2_TEM40/
│   │   └── images/
│   │       ├── image_1753104669.58981.bmp
│   │       ├── image_1753104669.80470.bmp
│   │       └── ...  (~14,500 BMP files per experiment)
│   └── ...  (11 experiments)
├── Susceptible/
│   ├── EC126_Ampicillin_16mgL_preincubated/
│   │   └── images/
│   │       └── ...
│   └── ...  (16 experiments)
└── Test/
    ├── EC35_Ampicillin_16mgL_preincubated_3_TEM40/
    │   └── images/
    │       └── ...
    └── ...  (15 experiments, labels inferred from EC number)
```

- **Top level:** Three directories: `Resistant/` and `Susceptible/` providing the binary label, plus `Test/` for held-out evaluation
- **Experiment folders:** Named as `EC{number}_{Antibiotic}_{Dose}_{details}` (e.g., `EC35_Ampicillin_16mgL_preincubated_2_TEM40`)
- **Test labels:** Inferred from EC number matching -- if EC35 appears in `Resistant/`, all EC35 experiments in `Test/` are labelled resistant
- **Images:** Inside an `images/` subdirectory, BMP files named `image_{unix_timestamp.milliseconds}.bmp`
- **Image format:** 1280x1024 grayscale BMP at 5 frames per second, ~1 hour per experiment (~14,500 frames)

### YOLO Model

The YOLOv11-OBB model was pretrained on DOTAv1 and finetuned on bacteria microscopy images. It outputs three classes:

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | `focused` | In-focus bacteria (used for analysis) |
| 1 | `unfocused` | Out-of-focus bacteria (discarded) |
| 2 | `vertical` | Vertically oriented bacteria (discarded) |

### Configuration

Paths are configured in `config.py`:

```python
data_root: Path = Path("/mnt/f/Data_second_protocol")
yolo_weights: Path = Path("/mnt/c/users/mkedz/Documents/PhD/PhD_code/yolo11/vertical_obb_100epo_best.pt")
```

Or pass them as CLI arguments (see Usage below).

---

## Usage

### Stage 1: Preprocessing

Detects bacteria with YOLOv11-OBB, filters to in-focus detections only, extracts oriented bounding box crops at native pixel size (centred on a 128x128 canvas), and stores them with timestamps in HDF5 format.

```bash
PYTHONPATH=/path/to/parent python3 -m ast_classifier.scripts.preprocess \
    --data-root /mnt/f/Data_second_protocol \
    --output-dir ./preprocessed \
    --yolo-weights /path/to/vertical_obb_100epo_best.pt
```

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-root` | *required* | Path to data folder with Resistant/, Susceptible/, and Test/ subdirectories |
| `--output-dir` | *required* | Output directory for HDF5 crop files |
| `--yolo-weights` | *required* | Path to YOLOv11-OBB `.pt` weights file |
| `--crop-size` | 128 | Output crop canvas size (square); bacteria are placed at native pixel size |
| `--yolo-confidence` | 0.5 | Minimum detection confidence |
| `--yolo-batch-size` | 16 | Frames per YOLO inference batch |
| `--focused-class-name` | focused | YOLO class name for in-focus bacteria |
| `--device` | cuda:0 | Inference device |

**Outputs per experiment:**

- `{experiment_id}.h5` -- HDF5 file containing:
  - `/crops` dataset: `(N, 128, 128)` uint8 grayscale crops (bacteria at native pixel size, centred on canvas)
  - `/metadata` structured array: `timestamp`, `detection_id`, `cx`, `cy`, `w`, `h`, `angle`, `confidence`

**Expected runtime:** ~26 minutes per experiment, ~18 hours total for 42 experiments on RTX 3090

**Expected output:** ~65K focused crops per experiment, ~2.7M crops total

---

### Stage 2: DINO Pretraining

Trains a ViT-Small backbone with self-supervised DINO on all bacteria crops, learning morphological representations without labels.

```bash
PYTHONPATH=/path/to/parent python3 -m ast_classifier.scripts.train \
    --stage dino \
    --preprocessed-dir ./preprocessed \
    --checkpoints-dir ./checkpoints
```

**What happens:**

1. Loads bacteria crops from all HDF5 files (capped at 5,000 per experiment for balance)
2. Applies multi-crop augmentation: 2 global crops (128x128) + 6 local crops (64x64)
3. Trains student-teacher ViT-Small pair with DINO loss for 100 epochs
4. Saves best backbone checkpoint to `checkpoints/dino/best_backbone.pt`

**Expected runtime:** ~2-4 hours on RTX 3090

**VRAM usage:** ~10 GB

**Monitoring:** TensorBoard logs in `logs/dino/`

```bash
tensorboard --logdir ./logs/dino
```

---

### Stage 3: Feature Extraction

Extracts 384-dimensional CLS token features for every crop using the pretrained backbone.

```bash
PYTHONPATH=/path/to/parent python3 -m ast_classifier.scripts.train \
    --stage extract \
    --preprocessed-dir ./preprocessed \
    --features-dir ./features \
    --checkpoints-dir ./checkpoints
```

**Outputs:** One `.npz` file per experiment containing:
- `features`: `(N, 384)` float16 feature vectors
- `timestamps`: `(N,)` float64 unix timestamps

**Expected runtime:** ~30 minutes on RTX 3090

---

### Stage 4: Classifier Training

Trains the Population Temporal classifier on pre-extracted features.

```bash
PYTHONPATH=/path/to/parent python3 -m ast_classifier.scripts.train \
    --stage classifier \
    --data-root /mnt/f/Data_second_protocol \
    --features-dir ./features \
    --checkpoints-dir ./checkpoints
```

**What happens:**

1. Builds experiment list from data directory; Test/ experiments become test set, Resistant/Susceptible split into train/val
2. Creates `PopulationTemporalDataset` with time-binned features and randomly sampled time windows
3. For each sample:
   - Bins pre-extracted features by timestamp into configurable time windows (default 2 minutes)
   - Computes per-bin population statistics (mean, std, skewness, kurtosis + crop count)
   - Encodes bin sequence with continuous-time positional encoding + 4-layer Transformer
   - Pools over time bins with gated attention
   - Classifies with time-conditioned MLP head
4. Trains with time-aware loss (early correct predictions weighted 3x higher)
5. Early stopping on validation AUROC (patience=30 epochs)

**Expected runtime:** ~1-2 hours on RTX 3090

**VRAM usage:** ~8-12 GB

**Monitoring:** TensorBoard logs in `logs/classifier/`

---

### Stage 5: Early-Exit Calibration

Calibrates the early-exit policy on the validation set.

```bash
PYTHONPATH=/path/to/parent python3 -m ast_classifier.scripts.train \
    --stage calibrate \
    --data-root /mnt/f/Data_second_protocol \
    --features-dir ./features \
    --checkpoints-dir ./checkpoints
```

**What happens:**

1. Evaluates the trained classifier at every 30-second interval from 60s to 3600s
2. Fits a temperature scaler on validation logits (LBFGS optimization)
3. Applies temperature scaling to all predictions
4. Sweeps over patience x threshold grid to find Pareto-optimal operating points
5. Reports optimal configurations for 90%, 95%, 99% accuracy targets

**Outputs:**
- `checkpoints/classifier/temperature_scaler.pt` -- calibrated temperature parameter
- `checkpoints/classifier/calibration_result.pkl` -- full calibration results

---

### Evaluation

Runs full evaluation on the test set with early-exit simulation.

```bash
PYTHONPATH=/path/to/parent python3 -m ast_classifier.scripts.evaluate \
    --data-root /mnt/f/Data_second_protocol \
    --features-dir ./features \
    --checkpoints-dir ./checkpoints \
    --output-dir ./results
```

**Outputs:**

- `evaluation_results.json` -- Full metrics including:
  - Classification metrics at 60 minutes (accuracy, AUROC, AUPRC, sensitivity, specificity, F1, MCC)
  - Early-exit metrics (accuracy with halting policy)
  - Time-to-prediction analysis (mean/median exit time, time to reach 90/95/99% accuracy)
  - Accuracy at fixed time points (5, 10, 15, 30 minutes)
- `accuracy_vs_time.png` -- Accuracy improvement curve over observation time
- `exit_time_distribution.png` -- Histogram of early-exit times
- `pareto_front.png` -- Accuracy vs speed trade-off

---

### Full Pipeline

Run all stages automatically (with preprocessing monitoring):

```bash
./scripts/run_full_pipeline.sh
```

This script:
1. Waits for preprocessing to complete (if running)
2. Runs DINO pretraining, feature extraction, classifier training, calibration, and evaluation in sequence
3. Logs each stage to `logs/`

Or run stages 2-5 sequentially via the train script:

```bash
PYTHONPATH=/path/to/parent python3 -m ast_classifier.scripts.train \
    --stage all \
    --data-root /mnt/f/Data_second_protocol \
    --preprocessed-dir ./preprocessed \
    --features-dir ./features \
    --checkpoints-dir ./checkpoints
```

---

## Configuration Reference

All configuration is centralized in `config.py` using Python dataclasses.

### PathConfig

| Field | Default | Description |
|-------|---------|-------------|
| `data_root` | `/mnt/f/Data_second_protocol` | Root data directory (Resistant/, Susceptible/, Test/) |
| `preprocessed_dir` | `./preprocessed` | HDF5 output from Stage 1 |
| `features_dir` | `./features` | NPZ feature files from Stage 3 |
| `checkpoints_dir` | `./checkpoints` | Model checkpoints (all stages) |
| `logs_dir` | `./logs` | TensorBoard logs |
| `yolo_weights` | (see config.py) | YOLOv11-OBB weights file |

### PreprocessingConfig

| Field | Default | Description |
|-------|---------|-------------|
| `yolo_confidence` | 0.5 | YOLO detection confidence threshold |
| `crop_size` | 128 | Output crop canvas size (pixels, square); bacteria placed at native size |
| `yolo_batch_size` | 16 | Frames per YOLO inference batch |
| `focused_class_name` | `focused` | YOLO class name for in-focus bacteria |
| `fps` | 5.0 | Camera frame rate |

### DINOConfig

| Field | Default | Description |
|-------|---------|-------------|
| `img_size` | 128 | Input image size |
| `patch_size` | 16 | ViT patch size (128/16 = 8x8 = 64 patches) |
| `embed_dim` | 384 | Transformer embedding dimension |
| `depth` | 12 | Number of transformer blocks |
| `num_heads` | 6 | Attention heads (384/6 = 64 dim/head) |
| `mlp_ratio` | 4.0 | FFN expansion ratio |
| `drop_path_rate` | 0.1 | Stochastic depth rate |
| `batch_size` | 64 | Training batch size |
| `epochs` | 100 | Training epochs |
| `base_lr` | 5e-4 | Base learning rate (scaled by batch_size/256) |
| `warmup_epochs` | 10 | Linear LR warmup |
| `max_crops_per_experiment` | 5000 | Cap per experiment for balanced sampling |

### ClassifierConfig

| Field | Default | Description |
|-------|---------|-------------|
| `feature_dim` | 384 | Input feature dimension (from backbone) |
| `temporal_hidden_dim` | 256 | Temporal encoder / bin embedding dimension |
| `temporal_num_layers` | 4 | Temporal transformer depth |
| `temporal_num_heads` | 4 | Temporal attention heads |
| `temporal_ffn_dim` | 512 | Temporal FFN dimension |
| `classifier_hidden_dim` | 128 | Classifier MLP hidden dimension |
| `time_bin_width_sec` | 120.0 | Time bin width in seconds (configurable) |
| `max_crops_per_bin` | 256 | Maximum crops per time bin (subsampled if exceeded) |
| `batch_size` | 16 | Training batch size (experiments) |
| `gradient_accumulation` | 2 | Effective batch = 32 |
| `epochs` | 200 | Maximum training epochs |
| `lr` | 1e-3 | Learning rate |
| `early_stopping_patience` | 30 | Epochs without improvement before stopping |
| `time_loss_alpha` | 2.0 | Early prediction reward strength |
| `attention_entropy_weight` | 0.01 | Attention regularization weight |

### EarlyExitConfig

| Field | Default | Description |
|-------|---------|-------------|
| `patience` | 3 | Consecutive confident predictions required |
| `confidence_threshold` | 0.85 | Minimum softmax confidence |
| `eval_interval_sec` | 30 | Seconds between evaluations |
| `min_time_sec` | 60 | Minimum observation time before exit |
| `max_time_sec` | 3600 | Maximum observation time |

### DataSplitConfig

| Field | Default | Description |
|-------|---------|-------------|
| `val_ratio` | 0.15 | Fraction of R/S experiments held out for validation |
| `random_seed` | 42 | Reproducible splitting |

Test set is pre-defined by the `Test/` folder.

---

## Model Architecture Details

### ViT-Small Backbone

```
Input: (B, 1, 128, 128) grayscale image
  |
PatchEmbed: Conv2d(1, 384, kernel_size=16, stride=16)
  --> (B, 64, 384)  [8x8 grid of patches]
  |
Prepend CLS token + add positional embeddings
  --> (B, 65, 384)
  |
12x TransformerBlock:
  |   LayerNorm -> Multi-Head Self-Attention (6 heads, 64 dim/head)
  |   + residual connection + DropPath
  |   LayerNorm -> MLP (384 -> 1536 -> 384, GELU activation)
  |   + residual connection + DropPath
  |
Final LayerNorm
  |
Output: CLS token --> (B, 384)
```

**Parameters:** ~21.7M

### Population Temporal Classifier

```
Input: bin_features (B, T, N, 384), bin_times, masks, time_fraction

PopulationBinEncoder (per bin):
  Masked mean, std, skewness, kurtosis of crop features -> (B, 4*384)
  Append normalised crop count -> (B, 4*384 + 1)
  MLP(1537, 256) -> GELU -> Linear(256, 256)
  -> (B, 256) per-bin embedding

ContinuousTimeEncoding:
  Sinusoidal encoding of bin centre times (seconds)
  -> (B, T, 256)

PopulationTemporalEncoder:
  bin_embeddings + time_encoding -> (B, T, 256)
  4-layer TransformerEncoder (d=256, 4 heads, ffn=512, GELU, pre-norm)
  -> (B, T, 256) contextualized bin representations

GatedAttentionMIL (over time bins):
  V = tanh(Linear(256, 128) @ h)
  U = sigmoid(Linear(256, 128) @ h)
  a = softmax(Linear(128, 1) @ (V * U))  [masked]
  experiment_repr = sum(a * h) -> (B, 256)

ClassifierHead:
  cat(experiment_repr, time_frac) -> (B, 257)
  Linear(257, 128) -> LN -> GELU -> Dropout
  Linear(128, 128) -> LN -> GELU -> Dropout
  Linear(128, 2) -> (B, 2) logits
```

**Parameters:** ~1.5M (operates on pre-extracted features, very fast to train)

---

## Augmentation Strategy

Augmentations are designed for brightfield microscopy physics:

| Augmentation | Parameters | Rationale |
|-------------|-----------|-----------|
| RandomResizedCrop | Global: 128x128 scale (0.7, 1.0), Local: 64x64 scale (0.3, 0.6) | DINO multi-crop strategy; local crops capture sub-cellular detail |
| RandomRotation | 180 degrees | Bacteria have no canonical orientation in brightfield |
| RandomHorizontalFlip | p=0.5 | Symmetry augmentation |
| RandomVerticalFlip | p=0.5 | Symmetry augmentation |
| RandomIntensityJitter | brightness=0.3, contrast=0.3 | Simulates illumination variation between imaging sessions |
| RandomGaussianNoise | std in (0.0, 0.05) | Simulates camera sensor noise |
| RandomDefocusBlur | radius in (0, 3) | Simulates slight defocus drift during acquisition |
| Normalize | mean=0.5, std=0.25 | Empirical microscopy normalization |

**Not included:**
- Color jitter (images are grayscale)
- Elastic deformation (would distort the morphological features we want to detect)
- Cutout/erasing (risk removing the bacteria entirely from small crops)

---

## Loss Functions

### Time-Aware Cross-Entropy Loss

```
L_time = lambda(t) * CE(logits, label)
lambda(t) = 1 + alpha * (1 - t / T_max)
```

Where `alpha = 2.0`. At t=0, the weight is 3.0 (3x emphasis); at t=T_max, it is 1.0 (normal). This incentivizes the model to develop discriminative features that work early in the observation period.

### Attention Entropy Regularizer

```
L_entropy = (H(attention) / H_max - target_ratio)^2
```

Prevents attention weights from collapsing to a single time bin (too peaked) or spreading uniformly (uninformative). Target entropy ratio is 0.5, encouraging moderate concentration.

### Total Training Loss

```
L_total = L_time + 0.01 * L_entropy
```

### DINO Loss (Pretraining)

Cross-entropy between teacher softmax (centered, sharpened) and student log-softmax across all cross-view pairs (standard DINO formulation).

---

## Metrics and Evaluation

### Classification Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| Balanced Accuracy | (Sensitivity + Specificity) / 2, robust to class imbalance |
| AUROC | Area under receiver operating characteristic curve |
| AUPRC | Area under precision-recall curve |
| Sensitivity | True positive rate (correctly identifying resistant) |
| Specificity | True negative rate (correctly identifying susceptible) |
| F1 Score | Harmonic mean of precision and recall |
| MCC | Matthews correlation coefficient (-1 to +1, handles imbalance) |

### Time-to-Prediction Metrics

| Metric | Description |
|--------|-------------|
| Mean/Median Exit Time | Average time at which the model makes a confident prediction |
| Time to 90/95/99% Accuracy | Minimum observation time to reach each accuracy threshold |
| Accuracy at 5/10/15/30 min | Classification accuracy at fixed observation durations |

---

## Visualization

The `utils/visualization.py` module generates publication-quality figures:

| Function | Output | Purpose |
|----------|--------|---------|
| `plot_accuracy_vs_time` | Line plot | Shows how accuracy improves with observation duration; overlays 90% and 95% thresholds |
| `plot_exit_time_distribution` | Histogram | Distribution of when the model halts, with median line |
| `plot_attention_heatmap` | Bar chart | Which time bins the model attends to most (interpretability) |
| `plot_tsne_embeddings` | Scatter plot | t-SNE of DINO features colored by resistance label |
| `plot_population_heterogeneity` | Dual line plot | Feature variance over time for susceptible vs resistant populations |
| `plot_pareto_front` | Scatter + line | Accuracy vs speed trade-off across all calibration configurations |

---

## Project Structure

```
ast_classifier/
├── __init__.py                         # Package definition
├── config.py                           # All configuration dataclasses
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
│
├── data/                               # Data handling
│   ├── __init__.py
│   ├── preprocessing.py                # YOLO detection, focused filtering, OBB crop extraction, HDF5 storage
│   ├── augmentations.py                # Microscopy-specific augmentations for DINO
│   └── dataset.py                      # DINOCropDataset, PopulationTemporalDataset
│
├── models/                             # Neural network architectures
│   ├── __init__.py                     # Re-exports all model classes
│   ├── backbone.py                     # ViT-Small (128x128 grayscale, 1-channel)
│   ├── dino.py                         # DINO head, loss, student-teacher wrapper
│   ├── temporal_encoder.py             # PopulationBinEncoder + PopulationTemporalEncoder
│   ├── mil_aggregator.py               # Gated attention MIL + population feature extractor
│   ├── classifier.py                   # PopulationTemporalClassifier end-to-end
│   └── early_exit.py                   # Confidence-based exit, temperature scaling, learned halting
│
├── training/                           # Training loops
│   ├── __init__.py
│   ├── train_dino.py                   # Stage 2: DINO self-supervised pretraining
│   ├── extract_features.py             # Stage 3: Batch feature extraction
│   ├── train_classifier.py             # Stage 4: Population Temporal classifier training
│   └── calibrate_exit.py              # Stage 5: Early-exit threshold calibration
│
├── utils/                              # Evaluation and visualization
│   ├── __init__.py
│   ├── metrics.py                      # Classification, time-to-prediction metrics
│   └── visualization.py               # Publication-quality plotting functions
│
└── scripts/                            # CLI entry points
    ├── __init__.py
    ├── preprocess.py                   # Stage 1: YOLO detection + crop extraction
    ├── train.py                        # Stages 2-5: Training pipeline
    ├── evaluate.py                     # Full evaluation with early-exit simulation
    └── run_full_pipeline.sh            # Automated end-to-end pipeline
```

---

## Hardware Requirements

### Minimum (RTX 3090 / 24GB VRAM)

| Stage | VRAM Usage | Time Estimate |
|-------|-----------|---------------|
| Preprocessing | ~2 GB | ~18 hours (42 experiments) |
| DINO Pretraining | ~14 GB | ~3-5 hours |
| Feature Extraction | ~4 GB | ~30 minutes |
| Classifier Training | ~8-12 GB | ~1-2 hours |
| Calibration | ~4 GB | ~15 minutes |
| Evaluation | ~4 GB | ~15 minutes |
| **Total** | | **~22-25 hours** |

### Disk Space

| Data | Size |
|------|------|
| Raw images (input) | ~42 x 14.5K BMPs (~1.8TB) |
| Preprocessed HDF5 files | ~18-25 GB |
| Extracted features (NPZ) | ~2-4 GB |
| Checkpoints | ~500 MB |
| **Total additional** | **~22-30 GB** |

### Scaling

- **Multi-GPU:** Increase batch sizes proportionally. DINO supports `DistributedDataParallel` out of the box
- **More VRAM:** Increase `batch_size`, increase `max_crops_per_bin`
- **Less VRAM (16GB):** Reduce `batch_size` to 32 for DINO, increase `gradient_accumulation` for classifier

---

## Literature and References

This pipeline draws on and integrates techniques from the following research:

### Rapid AST from Microscopy
- Lim et al. (2018). "Phenotypic AST with Deep Learning Video Microscopy." *Analytical Chemistry*
- Zagajewski et al. (2023). "Deep learning and single-cell phenotyping for rapid AST." *Communications Biology*
- Hallstrom et al. (2025). "Rapid label-free identification of bacterial species." *PLOS ONE*

### Self-Supervised Learning
- Caron et al. (2021). "Emerging Properties in Self-Supervised Vision Transformers (DINO)." *ICCV*
- Oquab et al. (2024). "DINOv2: Learning Robust Visual Features without Supervision." *TMLR*
- Cell-DINO (2025). "Cell-DINO for cell fluorescent microscopy." *PLOS Computational Biology*

### Multi-Instance Learning
- Ilse et al. (2018). "Attention-based Deep Multiple Instance Learning." *ICML*
- TransMIL (2021). "Transformer-based Correlated MIL." *NeurIPS*

### Early Classification of Time Series
- CALIMERA (2023). "Cost-aware early classification." *Information Processing and Management*
- Early-Exit DNN Survey (2024). *ACM Computing Surveys*

### Bacterial Image Analysis
- DeepBacs (2022). "Multi-task bacterial image analysis." *Communications Biology*
- DeLTA 2.0 (2022). "Deep learning for bacterial cell tracking." *PLOS Computational Biology*

### Confidence Calibration
- Guo et al. (2017). "On Calibration of Modern Neural Networks." *ICML*
