# AST Classifier: Temporal Multi-Instance Learning for Rapid Antimicrobial Susceptibility Testing

A deep learning pipeline that distinguishes antibiotic-resistant from susceptible *E. coli* bacteria using time-lapse brightfield microscopy. The system detects subtle morphological changes in individual bacteria after antibiotic exposure and classifies resistance/susceptibility as early as possible -- potentially within minutes rather than the traditional 16-24 hours required by culture-based methods.

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
- [Tracking Algorithm](#tracking-algorithm)
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

This system uses an existing YOLOv11-OBB model to detect individual bacteria in 100x brightfield microscopy images, then tracks them across time-lapse frames captured at 5 fps over one hour. After antibiotic exposure, susceptible bacteria undergo subtle morphological changes (elongation, blebbing, lysis, growth arrest) while resistant bacteria continue dividing normally. These changes may be imperceptible to the human eye in early frames but are detectable by a neural network trained to recognize them.

The pipeline operates in five stages:

```
Stage 1: Detection + Tracking    Raw BMP frames --> individual bacteria crops + temporal tracks
Stage 2: DINO Pretraining        Crops --> self-supervised ViT-Small backbone
Stage 3: Feature Extraction       Crops --> cached 384-dim feature vectors
Stage 4: Temporal MIL Classifier  Feature sequences --> resistant/susceptible prediction
Stage 5: Early-Exit Calibration   Validation set --> optimal halting thresholds
```

### Why Not a Simple CNN?

A single-frame CNN classifier cannot capture the key biological signal: **morphological change over time**. The distinction between resistant and susceptible bacteria is not what they look like at any single moment, but how their appearance evolves after antibiotic exposure. This requires temporal modeling of individual bacteria trajectories and population-level statistical analysis.

---

## Architecture

```
                               Raw BMP Frames (1280x1024, grayscale, 5fps)
                                              |
                                    YOLOv11-OBB Detection
                                              |
                              Oriented Bounding Box Crop Extraction
                                     (affine rectification)
                                              |
                                      96x96 Grayscale Crops
                                              |
                                    IoU-based SORT Tracking
                                  (link bacteria across frames)
                                              |
                            ViT-Small Backbone (DINO pretrained)
                                      CLS token: 384-dim
                                              |
                        +-----------+---------+---------+-----------+
                        |           |         |         |           |
                    Track 1     Track 2   Track 3   Track 4    Track N
                    (T, 384)   (T, 384)  (T, 384)  (T, 384)   (T, 384)
                        |           |         |         |           |
                    Delta Features: feature[t] - feature[t-k]
                    at scales k = {1, 5, 25, 125} frames
                        |           |         |         |           |
                    Temporal Transformer Encoder (4 layers, 256-dim)
                        |           |         |         |           |
                    Per-track representations (256-dim each)
                        |           |         |         |           |
                        +-----------+---------+---------+-----------+
                                              |
                        +---------------------+---------------------+
                        |                                           |
              Gated Attention MIL                    Population Feature Extractor
          (attention-weighted aggregation)           (mean, std, skew, kurtosis)
                   (B, 256)                                   (B, 64)
                        |                                           |
                        +-------------------+---+-------------------+
                                            |   |
                                     time_fraction (1-dim)
                                            |
                                    Classifier Head
                            (MLP: 321 --> 128 --> 128 --> 2)
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

**Justification:** With ~500K images containing many bacteria each, we have millions of unlabeled bacteria crops but only experiment-level labels (resistant vs susceptible). DINO learns morphological representations without labels by training a student network to match a momentum-averaged teacher network across different augmented views of the same image. This has been shown to outperform supervised pretraining for cell phenotyping tasks (Cell-DINO, PLOS Computational Biology 2025), and ViTs specifically outperform CNNs for bacterial classification from phase-contrast microscopy (Hallstrom et al., PLOS ONE 2025). The self-supervised features capture morphological structure that transfers well to downstream classification.

### 2. Vision Transformer over CNN

**Choice:** ViT-Small (384-dim, 12 layers, 6 heads) rather than ResNet or EfficientNet.

**Justification:** ViTs learn global attention patterns across the entire image from the first layer, while CNNs build local-to-global hierarchies. For bacteria at 100x magnification, subtle morphological features (membrane irregularities, nucleoid condensation, elongation) may appear at various spatial scales simultaneously. Hallstrom et al. (2025) directly compared ViT and ResNet architectures for bacterial species identification from microfluidic time-lapse microscopy and found ViT consistently outperformed ResNet with lower variance. The 96x96 input with 16x16 patches produces only 36 tokens, making the model efficient enough for a single RTX 3090.

### 3. Temporal Contrastive Learning (DynaCLR-style)

**Choice:** After 50 epochs of standard DINO, add a temporal contrastive loss that pulls together embeddings of the same bacterium at different timepoints.

**Justification:** Standard DINO treats each crop independently, but the key biological signal is temporal -- how a bacterium changes over time. By adding an NT-Xent contrastive loss where positive pairs are the same tracked bacterium at times *t* and *t + tau*, the backbone learns to encode features that are temporally smooth for the same individual while discriminating between different bacteria. This is inspired by DynaCLR (2024), which demonstrated that cell-aware and time-aware contrastive encoding captures dynamic phenotypic changes in time-lapse microscopy. Starting at epoch 50 allows the backbone to first learn general morphological features before refining temporal coherence.

### 4. Multi-Scale Delta Features

**Choice:** Explicitly compute `feature[t] - feature[t-k]` at scales k = {1, 5, 25, 125} and feed these alongside raw features into the temporal encoder.

**Justification:** The classifier must detect morphological *change*, not just morphological *state*. A susceptible bacterium elongating after ciprofloxacin exposure looks different from one that hasn't been exposed, but the difference is captured most clearly by comparing its current appearance to its past appearance. Computing temporal differences at multiple scales captures changes at different rates:
- k=1 (0.2s): instantaneous movement/focus drift noise
- k=5 (1s): rapid morphological changes (membrane blebbing)
- k=25 (5s): moderate changes (early elongation)
- k=125 (25s): slow changes (growth arrest, gradual lysis)

These deltas are projected through a learned linear layer to extract the most informative change signals, then summed with the raw features before entering the temporal transformer.

### 5. Per-Bacterium Temporal Transformer

**Choice:** A 4-layer Transformer encoder processes each bacterium's feature trajectory independently, producing a fixed-size temporal representation.

**Justification:** Each tracked bacterium has a sequence of feature vectors over time (up to 512 frames at 1fps effective rate). A Transformer encoder with self-attention can learn which timepoints are most informative for each bacterium -- early timepoints may be uninformative, while the critical morphological change might occur at a specific moment. The self-attention mechanism allows the model to attend to these critical transition points. We use sinusoidal positional encodings so the model understands temporal ordering. Mean-pooling over valid timesteps (masked for variable-length tracks) produces the final per-bacterium representation. At 256 dimensions with 4 heads and 4 layers, this is lightweight enough to process all tracks efficiently.

### 6. Multi-Instance Learning with Gated Attention

**Choice:** Frame the problem as Multi-Instance Learning (MIL) with gated attention aggregation (Ilse et al., 2018).

**Justification:** Each experiment contains 30-80 tracked bacteria, but the label (resistant/susceptible) applies to the entire experiment, not to individual bacteria. This is the textbook MIL formulation: a "bag" of instances with a bag-level label. Not every bacterium in a susceptible experiment will show obvious changes at the same time -- some may be in different growth phases, some may persist. Gated attention learns to weight individual bacteria by their informativeness:
- `V = tanh(W_1 h)` captures content relevance
- `U = sigmoid(W_2 h)` acts as a gate
- `a = softmax(W_3 (V * U))` produces attention weights

This provides both an aggregated bag representation and interpretable attention weights showing which bacteria most influenced the classification.

### 7. Population Distribution Features

**Choice:** In addition to attention-weighted aggregation, compute statistical moments (mean, std, skewness, kurtosis) of the per-bacterium feature distribution.

**Justification:** The attention mechanism selects the "most informative" bacteria, but the *distribution* of responses across the population carries biological signal. Resistant populations remain homogeneous (all bacteria continue growing normally), while susceptible populations become heterogeneous over time (some lysing, some elongating, some persisting). Population statistics capture this heterogeneity directly:
- **Std** increases as susceptible bacteria diverge in phenotype
- **Skewness** changes as the distribution becomes asymmetric (e.g., most cells elongating but some lysing)
- **Kurtosis** captures whether the distribution is concentrated (resistant, uniform response) or has heavy tails (susceptible, diverse responses)

### 8. Time-Aware Loss

**Choice:** Weight the cross-entropy loss by `lambda(t) = 1 + alpha * (1 - t/T_max)` where alpha=2.0, giving 3x weight to predictions at t=0 vs t=T_max.

**Justification:** The clinical value of a prediction decreases with time -- a correct prediction at 5 minutes is far more valuable than one at 55 minutes. By weighting early correct predictions more heavily, the model is incentivized to learn features that discriminate resistance/susceptibility as early as possible. Combined with multi-scale temporal window sampling (biased toward shorter windows during training), this drives the model to extract discriminative signal from the earliest frames.

### 9. Confidence-Based Early Exit

**Choice:** At inference, evaluate every 30 seconds and halt when the model produces N consecutive predictions with softmax confidence above a threshold.

**Justification:** This patience-based mechanism prevents premature exit on noisy early predictions while allowing rapid exit once the model is consistently confident. The threshold and patience are calibrated on a validation set by sweeping over a grid of values and finding Pareto-optimal operating points that balance accuracy vs speed. Temperature scaling (Guo et al., 2017) ensures that the model's confidence scores are well-calibrated (i.e., a prediction with 90% confidence is correct ~90% of the time). An optional learned halting policy (LSTM-based, trained with REINFORCE) is available for more sophisticated decision-making.

### 10. HDF5 Storage Instead of Individual Files

**Choice:** Store all crops from each experiment in a single HDF5 file rather than millions of individual image files.

**Justification:** With ~20 focused bacteria per frame x 18,000 frames per experiment x ~500 experiments, the dataset contains approximately 10 million individual bacteria crops. Storing these as individual PNG/BMP files would create massive filesystem overhead (inode exhaustion, slow directory listing, degraded I/O performance). HDF5 provides chunked, optionally compressed storage with fast random access by index, reducing the filesystem burden to ~1,000 files while maintaining efficient data loading for training.

### 11. Offline Feature Caching

**Choice:** Extract ViT features for all crops once and cache them as .npz files, then train the temporal MIL classifier entirely on pre-extracted features.

**Justification:** The ViT backbone has 21.7M parameters and processes 96x96 images -- running it during every training epoch of the classifier would be prohibitively expensive. By extracting features once (384-dim float16, ~7.3GB total for 10M crops), the classifier training operates in feature-space rather than image-space. This makes the classifier training loop extremely fast (minutes per epoch vs hours) and allows extensive hyperparameter search and ablation studies. The feature extraction itself takes only ~30 minutes on an RTX 3090.

### 12. Stratified Experiment-Level Splitting

**Choice:** Split train/val/test by experiment ID (not by frame or time window), stratified by (label, antibiotic_id).

**Justification:** All bacteria in a single experiment share identical environmental conditions (same antibiotic, same dosage, same bacterial strain, same imaging session). If frames from the same experiment appeared in both training and test sets, the model could overfit to experiment-specific artifacts (illumination, focus plane, background texture) rather than learning generalizable resistance phenotypes. Stratification by label and antibiotic ensures each split has representative coverage.

---

## Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (tested on RTX 3090, 24GB VRAM)
- CUDA 11.8+ and cuDNN

### Setup

```bash
# Clone or navigate to the project
cd /path/to/ast_classifier

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >=2.2.0 | Deep learning framework |
| `torchvision` | >=0.17.0 | Image transforms |
| `ultralytics` | >=8.1.0 | YOLOv11-OBB inference |
| `timm` | >=0.9.12 | Vision Transformer utilities |
| `h5py` | >=3.10.0 | HDF5 data storage |
| `opencv-python` | >=4.9.0 | Image processing, affine transforms |
| `shapely` | >=2.0.0 | OBB IoU computation via polygon intersection |
| `lap` | >=0.4.0 | Hungarian algorithm for tracking |
| `scikit-learn` | >=1.4.0 | Metrics, t-SNE, PCA |
| `pandas` | >=2.1.0 | Metadata management |
| `numpy` | >=1.26.0 | Numerical operations |
| `scipy` | >=1.12.0 | Scientific computing |
| `matplotlib` | >=3.8.0 | Plotting |
| `seaborn` | >=0.13.0 | Statistical visualization |
| `tensorboard` | >=2.15.0 | Training monitoring |
| `tqdm` | >=4.66.0 | Progress bars |
| `Pillow` | >=10.2.0 | Image I/O |
| `scikit-image` | >=0.22.0 | Image processing utilities |

### Verifying Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from ast_classifier.config import FullConfig; print('Config OK')"
```

---

## Data Layout

The system expects the following directory structure:

```
MainFolder/
├── Resistant/
│   ├── BacteriaID_AntibioticID_Dosage_ExperimentID/
│   │   └── images/
│   │       ├── 2024-03-01_12-00-00.bmp
│   │       ├── 2024-03-01_12-00-00_200.bmp
│   │       └── ...  (sequential datetime-named .bmp files)
│   ├── BacteriaID_AntibioticID_Dosage_ExperimentID/
│   │   └── images/
│   │       └── ...
│   └── ...
├── Susceptible/
│   ├── BacteriaID_AntibioticID_Dosage_ExperimentID/
│   │   └── images/
│   │       └── ...
│   └── ...
```

- **Top level:** Two directories `Resistant/` and `Susceptible/` providing the binary label
- **Experiment folders:** Named as `BacteriaID_AntibioticID_Dosage_ExperimentID` (underscore-separated)
- **Images:** Inside an `images/` subdirectory, BMP files named with datetime stamps. Lexicographic sort of filenames must equal temporal order
- **Image format:** 1280x1024 grayscale BMP at 5 frames per second, 1 hour per experiment (~18,000 frames)

### Before Running

Update two paths in `config.py`:

```python
# Line 8: Set your data directory
data_root: Path = Path("/path/to/MainFolder")

# Line 13: Set your YOLO weights path
yolo_weights: Path = Path("/path/to/yolo11-obb.pt")
```

Or pass them as CLI arguments (see Usage below).

---

## Usage

### Stage 1: Preprocessing

Detects bacteria with YOLOv11-OBB, extracts oriented bounding box crops, and tracks individuals across frames.

```bash
python -m ast_classifier.scripts.preprocess \
    --data-root /path/to/MainFolder \
    --output-dir ./preprocessed \
    --yolo-weights /path/to/yolo11-obb.pt
```

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-root` | *required* | Path to MainFolder with Resistant/ and Susceptible/ subdirectories |
| `--output-dir` | *required* | Output directory for HDF5 crop files and CSV tracking metadata |
| `--yolo-weights` | *required* | Path to YOLOv11-OBB `.pt` weights file |
| `--crop-size` | 96 | Output crop dimensions (square) |
| `--yolo-confidence` | 0.5 | Minimum detection confidence |
| `--yolo-batch-size` | 16 | Frames per YOLO inference batch |
| `--iou-threshold` | 0.3 | Minimum IoU for tracking match |
| `--max-track-age` | 15 | Frames before unmatched track is deleted (3s at 5fps) |
| `--min-track-hits` | 5 | Minimum detections to confirm a track (1s at 5fps) |
| `--min-track-length` | 150 | Minimum track duration in frames (30s at 5fps) |
| `--device` | cuda:0 | Inference device |

**Outputs per experiment:**

- `{experiment_id}.h5` -- HDF5 file containing:
  - `/crops` dataset: `(N, 96, 96)` uint8 grayscale crops
  - `/metadata` structured array: `frame_idx`, `detection_id`, `cx`, `cy`, `w`, `h`, `angle`, `confidence`, `track_id`
- `{experiment_id}.csv` -- Tracking results with all detection metadata plus `track_id`, `experiment_id`, and `label` columns

**Expected runtime:** ~4 hours for 500K images on RTX 3090 (~40 fps YOLO inference)

**Expected output size:** ~30-40 GB (with gzip compression)

---

### Stage 2: DINO Pretraining

Trains a ViT-Small backbone with self-supervised DINO on all bacteria crops, learning morphological representations without labels.

```bash
python -m ast_classifier.scripts.train \
    --stage dino \
    --preprocessed-dir ./preprocessed \
    --checkpoints-dir ./checkpoints
```

**What happens:**

1. Loads bacteria crops from all HDF5 files (capped at 5,000 per experiment for balance)
2. Applies multi-crop augmentation: 2 global crops (96x96) + 6 local crops (48x48)
3. Trains student-teacher ViT-Small pair with DINO loss for 100 epochs
4. After epoch 50, adds temporal contrastive loss using tracked bacteria pairs
5. Saves best backbone checkpoint to `checkpoints/dino/best_backbone.pt`

**Expected runtime:** ~2.5 days on RTX 3090

**VRAM usage:** ~10 GB

**Monitoring:** TensorBoard logs in `logs/dino/`

```bash
tensorboard --logdir ./logs/dino
```

---

### Stage 3: Feature Extraction

Extracts 384-dimensional CLS token features for every crop using the pretrained backbone.

```bash
python -m ast_classifier.scripts.train \
    --stage extract \
    --preprocessed-dir ./preprocessed \
    --features-dir ./features \
    --checkpoints-dir ./checkpoints
```

Or with a specific backbone checkpoint:

```bash
python -m ast_classifier.scripts.train \
    --stage extract \
    --backbone-path ./checkpoints/dino/best_backbone.pt
```

**Outputs:** One `.npz` file per experiment containing:
- `features`: `(N, 384)` float16 feature vectors
- `frame_idx`, `track_id`, `detection_id`: integer arrays
- `cx`, `cy`, `w`, `h`, `angle`, `confidence`: float arrays

**Expected runtime:** ~30 minutes on RTX 3090 (~5,000 crops/sec at batch_size=512)

**Expected output size:** ~7.3 GB

---

### Stage 4: Classifier Training

Trains the temporal MIL classifier on pre-extracted features.

```bash
python -m ast_classifier.scripts.train \
    --stage classifier \
    --data-root /path/to/MainFolder \
    --features-dir ./features \
    --checkpoints-dir ./checkpoints
```

**What happens:**

1. Builds experiment list from data directory, splits 70/15/15 (stratified by label and antibiotic)
2. Creates `TemporalMILDataset` with randomly sampled time windows (biased toward shorter windows)
3. For each sample:
   - Loads pre-extracted features for all tracked bacteria up to the selected time window
   - Computes multi-scale delta features
   - Encodes each bacterium's temporal trajectory with a 4-layer Transformer
   - Aggregates across bacteria with gated attention MIL
   - Computes population distribution statistics
   - Classifies with time-conditioned MLP head
4. Trains with time-aware loss (early correct predictions weighted 3x higher)
5. Early stopping on validation AUROC (patience=30 epochs)

**Expected runtime:** ~1-2 days on RTX 3090

**VRAM usage:** ~12-16 GB

**Monitoring:** TensorBoard logs in `logs/classifier/`

---

### Stage 5: Early-Exit Calibration

Calibrates the early-exit policy on the validation set.

```bash
python -m ast_classifier.scripts.train \
    --stage calibrate \
    --data-root /path/to/MainFolder \
    --features-dir ./features \
    --checkpoints-dir ./checkpoints
```

**What happens:**

1. Evaluates the trained classifier at every 30-second interval from 60s to 3600s
2. Fits a temperature scaler on validation logits (LBFGS optimization)
3. Applies temperature scaling to all predictions
4. Sweeps over patience x threshold grid:
   - Patience: {1, 2, 3, 5, 8} consecutive evaluations
   - Threshold: {0.70, 0.75, 0.80, 0.85, 0.90, 0.95} confidence
5. Finds Pareto-optimal operating points (best accuracy for each speed)
6. Reports optimal configurations for 90%, 95%, 99% accuracy targets

**Outputs:**
- `checkpoints/classifier/temperature_scaler.pt` -- calibrated temperature parameter
- `checkpoints/classifier/calibration_result.pkl` -- full calibration results

---

### Evaluation

Runs full evaluation on the test set with early-exit simulation.

```bash
python -m ast_classifier.scripts.evaluate \
    --data-root /path/to/MainFolder \
    --features-dir ./features \
    --checkpoints-dir ./checkpoints \
    --output-dir ./results
```

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-root` | *required* | Path to MainFolder |
| `--features-dir` | *required* | Extracted features directory |
| `--checkpoints-dir` | *required* | Model checkpoints directory |
| `--output-dir` | *required* | Results output directory |
| `--device` | cuda:0 | Inference device |

**Outputs:**

- `evaluation_results.json` -- Full metrics including:
  - Classification metrics at 60 minutes (accuracy, AUROC, AUPRC, sensitivity, specificity, F1, MCC)
  - Early-exit metrics (accuracy with halting policy)
  - Time-to-prediction analysis (mean/median exit time, time to reach 90/95/99% accuracy)
  - Accuracy at fixed time points (5, 10, 15, 30 minutes)
  - Per-antibiotic breakdown
- `accuracy_vs_time.png` -- Accuracy improvement curve over observation time
- `exit_time_distribution.png` -- Histogram of early-exit times
- `pareto_front.png` -- Accuracy vs speed trade-off

---

### Full Pipeline

Run all stages sequentially:

```bash
python -m ast_classifier.scripts.train \
    --stage all \
    --data-root /path/to/MainFolder \
    --preprocessed-dir ./preprocessed \
    --features-dir ./features \
    --checkpoints-dir ./checkpoints
```

This runs DINO pretraining, feature extraction, classifier training, and early-exit calibration in sequence.

---

## Configuration Reference

All configuration is centralized in `config.py` using Python dataclasses. Defaults are tuned for an RTX 3090 with ~500K images.

### PathConfig

| Field | Default | Description |
|-------|---------|-------------|
| `data_root` | `/path/to/MainFolder` | Root data directory (must contain Resistant/ and Susceptible/) |
| `preprocessed_dir` | `./preprocessed` | HDF5 and CSV output from Stage 1 |
| `features_dir` | `./features` | NPZ feature files from Stage 3 |
| `checkpoints_dir` | `./checkpoints` | Model checkpoints (all stages) |
| `logs_dir` | `./logs` | TensorBoard logs |
| `yolo_weights` | `/path/to/yolo11-obb.pt` | YOLOv11-OBB weights file |

### PreprocessingConfig

| Field | Default | Description |
|-------|---------|-------------|
| `yolo_confidence` | 0.5 | YOLO detection confidence threshold |
| `crop_size` | 96 | Output crop size (pixels, square) |
| `yolo_batch_size` | 16 | Frames per YOLO inference batch |
| `iou_threshold` | 0.3 | Minimum IoU for tracking match |
| `max_track_age` | 15 | Frames before unmatched track deletion |
| `min_track_hits` | 5 | Minimum detections to confirm track |
| `min_track_length` | 150 | Minimum track duration (frames) |
| `fps` | 5.0 | Camera frame rate |

### DINOConfig

| Field | Default | Description |
|-------|---------|-------------|
| `img_size` | 96 | Input image size |
| `patch_size` | 16 | ViT patch size (96/16 = 6x6 = 36 patches) |
| `embed_dim` | 384 | Transformer embedding dimension |
| `depth` | 12 | Number of transformer blocks |
| `num_heads` | 6 | Attention heads (384/6 = 64 dim/head) |
| `mlp_ratio` | 4.0 | FFN expansion ratio (hidden = 384 x 4 = 1536) |
| `drop_path_rate` | 0.1 | Stochastic depth rate (linearly increasing) |
| `batch_size` | 64 | Training batch size |
| `epochs` | 100 | Training epochs |
| `base_lr` | 5e-4 | Base learning rate (scaled by batch_size/256) |
| `warmup_epochs` | 10 | Linear LR warmup |
| `ema_momentum_start` | 0.996 | Teacher EMA start momentum |
| `ema_momentum_end` | 1.0 | Teacher EMA end momentum |
| `teacher_temp_start` | 0.04 | Teacher temperature start |
| `teacher_temp_end` | 0.07 | Teacher temperature end |
| `student_temp` | 0.1 | Student temperature |
| `temporal_loss_start_epoch` | 50 | Epoch to begin temporal contrastive loss |
| `temporal_loss_weight` | 0.5 | Weight of temporal contrastive loss |
| `max_crops_per_experiment` | 5000 | Cap per experiment for balanced sampling |

### ClassifierConfig

| Field | Default | Description |
|-------|---------|-------------|
| `feature_dim` | 384 | Input feature dimension (from backbone) |
| `temporal_hidden_dim` | 256 | Temporal encoder hidden dimension |
| `temporal_num_layers` | 4 | Temporal transformer depth |
| `temporal_num_heads` | 4 | Temporal attention heads |
| `temporal_ffn_dim` | 512 | Temporal FFN dimension |
| `mil_hidden_dim` | 128 | Gated attention hidden dimension |
| `population_feat_dim` | 64 | Population statistics output dimension |
| `classifier_hidden_dim` | 128 | Classifier MLP hidden dimension |
| `delta_scales` | [1, 5, 25, 125] | Temporal difference scales (frames) |
| `batch_size` | 16 | Training batch size (experiments) |
| `gradient_accumulation` | 2 | Effective batch = 32 |
| `max_tracks` | 64 | Maximum bacteria per experiment |
| `max_frames_per_track` | 512 | Maximum timesteps per track |
| `frame_subsample_rate` | 5 | Subsample factor (5fps -> 1fps effective) |
| `micro_batch_size` | 256 | Temporal encoder memory management |
| `epochs` | 200 | Maximum training epochs |
| `lr` | 1e-3 | Learning rate |
| `early_stopping_patience` | 30 | Epochs without improvement before stopping |
| `time_loss_alpha` | 2.0 | Early prediction reward strength |

### EarlyExitConfig

| Field | Default | Description |
|-------|---------|-------------|
| `patience` | 3 | Consecutive confident predictions required |
| `confidence_threshold` | 0.85 | Minimum softmax confidence |
| `eval_interval_sec` | 30 | Seconds between evaluations |
| `min_time_sec` | 60 | Minimum observation time before exit |
| `max_time_sec` | 3600 | Maximum observation time |
| `use_learned_halting` | False | Enable LSTM-based learned policy |

---

## Model Architecture Details

### ViT-Small Backbone

```
Input: (B, 1, 96, 96) grayscale image
  |
PatchEmbed: Conv2d(1, 384, kernel_size=16, stride=16)
  --> (B, 36, 384)  [6x6 grid of patches]
  |
Prepend CLS token + add positional embeddings
  --> (B, 37, 384)
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

### Temporal MIL Classifier

```
Input: track_features (B, N, T, 384), masks, time_fraction

DeltaFeatureComputer:
  For each scale k in {1, 5, 25, 125}:
    delta_k[t] = feature[t] - feature[t-k]  (zero-pad first k positions)
  Concatenate -> (B*N, T, 1536)
  Linear(1536, 384) -> (B*N, T, 384)

BacteriumTemporalEncoder:
  Linear(384, 256) [raw features]  +  Linear(384, 256) [deltas]
  -> (B*N, T, 256)  [element-wise sum]
  + Sinusoidal positional encoding
  -> 4-layer TransformerEncoder (d=256, 4 heads, ffn=512, GELU, pre-norm)
  -> Mean-pool over valid timesteps
  -> (B*N, 256)
  Reshape -> (B, N, 256)

GatedAttentionMIL:
  V = tanh(Linear(256, 128) @ h)
  U = sigmoid(Linear(256, 128) @ h)
  a = softmax(Linear(128, 1) @ (V * U))  [masked]
  bag = sum(a * h) -> (B, 256)

PopulationFeatureExtractor:
  Masked mean, std, skewness, kurtosis of h -> (B, 1024)
  Linear(1024, 256) -> GELU -> Linear(256, 64) -> (B, 64)

ClassifierHead:
  cat(bag, pop_feat, time_frac) -> (B, 321)
  Linear(321, 128) -> LN -> GELU -> Dropout
  Linear(128, 128) -> LN -> GELU -> Dropout
  Linear(128, 2) -> (B, 2) logits
```

**Parameters:** ~2.5M (operates on pre-extracted features, very fast to train)

---

## Augmentation Strategy

Augmentations are designed for brightfield microscopy physics:

| Augmentation | Parameters | Rationale |
|-------------|-----------|-----------|
| RandomResizedCrop | Global: scale (0.7, 1.0), Local: scale (0.3, 0.6) | DINO multi-crop strategy; local crops capture sub-cellular detail |
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

## Tracking Algorithm

### IoU-Based SORT Tracker

The tracker links bacteria detections across consecutive frames using a simplified SORT (Simple Online and Realtime Tracking) algorithm. Because bacteria at 100x magnification move very slowly (sub-pixel per frame at 5fps), pure IoU matching is sufficient without Kalman filtering or deep appearance features.

**Per-frame update cycle:**

1. **Compute cost matrix:** For each (active_track, new_detection) pair, compute `cost = 1 - IoU(track_obb, detection_obb)` using Shapely polygon intersection on the oriented bounding boxes
2. **Hungarian assignment:** Solve the optimal assignment using `lap.lapjv` with `cost_limit = 1 - iou_threshold = 0.7`
3. **Update matched tracks:** Reset age counter, update position, increment hit count
4. **Age unmatched tracks:** Increment age; delete if age exceeds `max_age` (15 frames = 3 seconds)
5. **Create new tracks:** Start a new track for each unmatched detection

**Post-processing filters:**
- Minimum hits: Track must have been detected in at least 5 frames (1 second)
- Minimum length: Track must span at least 150 frames (30 seconds)
- Detections belonging to filtered-out tracks are assigned `track_id = -1`

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

Prevents attention weights from collapsing to a single bacterium (too peaked) or spreading uniformly (uninformative). Target entropy ratio is 0.5, encouraging moderate concentration.

### Total Training Loss

```
L_total = L_time + 0.01 * L_entropy
```

### DINO Losses (Pretraining)

- **DINO Loss:** Cross-entropy between teacher softmax (centered, sharpened) and student log-softmax across all cross-view pairs
- **Temporal Contrastive (NT-Xent):** Normalized temperature-scaled cross-entropy pulling together embeddings of the same bacterium at different timepoints

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

### Per-Antibiotic Analysis

Breaks down accuracy and exit time by antibiotic type (parsed from experiment folder names), enabling identification of antibiotics where the model is most/least effective.

---

## Visualization

The `utils/visualization.py` module generates publication-quality figures:

| Function | Output | Purpose |
|----------|--------|---------|
| `plot_accuracy_vs_time` | Line plot | Shows how accuracy improves with observation duration; overlays 90% and 95% thresholds |
| `plot_exit_time_distribution` | Histogram | Distribution of when the model halts, with median line |
| `plot_attention_heatmap` | Bar chart | Which bacteria the model attends to most (interpretability) |
| `plot_tsne_embeddings` | Scatter plot | t-SNE of DINO features colored by resistance label; shows learned clustering |
| `plot_morphological_trajectory` | PCA trajectory | Single bacterium's feature evolution over time (start-to-end path) |
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
│   ├── preprocessing.py                # YOLO detection, OBB crop extraction, HDF5 storage
│   ├── tracking.py                     # IoU-based SORT tracker with OBB support
│   ├── augmentations.py                # Microscopy-specific augmentations for DINO
│   └── dataset.py                      # DINOCropDataset, TemporalPairDataset, TemporalMILDataset
│
├── models/                             # Neural network architectures
│   ├── __init__.py                     # Re-exports all model classes
│   ├── backbone.py                     # ViT-Small (96x96 grayscale, 1-channel)
│   ├── dino.py                         # DINO head, loss, temporal contrastive loss, wrapper
│   ├── temporal_encoder.py             # Delta features + per-bacterium temporal transformer
│   ├── mil_aggregator.py               # Gated attention MIL + population feature extractor
│   ├── classifier.py                   # Full TemporalMILClassifier end-to-end
│   └── early_exit.py                   # Confidence-based exit, temperature scaling, learned halting
│
├── training/                           # Training loops
│   ├── __init__.py
│   ├── train_dino.py                   # Stage 2: DINO self-supervised pretraining
│   ├── extract_features.py             # Stage 3: Batch feature extraction
│   ├── train_classifier.py             # Stage 4: Temporal MIL classifier training
│   └── calibrate_exit.py              # Stage 5: Early-exit threshold calibration
│
├── utils/                              # Evaluation and visualization
│   ├── __init__.py
│   ├── metrics.py                      # Classification, time-to-prediction, per-antibiotic metrics
│   └── visualization.py               # 7 publication-quality plotting functions
│
└── scripts/                            # CLI entry points
    ├── __init__.py
    ├── preprocess.py                   # Stage 1: YOLO + tracking pipeline
    ├── train.py                        # Stages 2-5: Training pipeline
    └── evaluate.py                     # Full evaluation with early-exit simulation
```

---

## Hardware Requirements

### Minimum (RTX 3090 / 24GB VRAM)

| Stage | VRAM Usage | Time Estimate |
|-------|-----------|---------------|
| Preprocessing | ~2 GB | ~4 hours |
| DINO Pretraining | ~10 GB | ~2.5 days |
| Feature Extraction | ~4 GB | ~30 minutes |
| Classifier Training | ~12-16 GB | ~1-2 days |
| Calibration | ~4 GB | ~1 hour |
| **Total** | | **~5-6 days** |

### Disk Space

| Data | Size |
|------|------|
| Raw images (input) | Varies (~500K BMPs) |
| Preprocessed HDF5 files | ~30-40 GB |
| Extracted features (NPZ) | ~7.3 GB |
| Checkpoints | ~1 GB |
| **Total additional** | **~40-50 GB** |

### Scaling

- **Multi-GPU:** Increase batch sizes proportionally. DINO supports `DistributedDataParallel` out of the box
- **More VRAM:** Increase `micro_batch_size` (256 -> 512), increase `max_tracks` (64 -> 128), increase `batch_size`
- **Less VRAM (16GB):** Reduce `batch_size` to 32 for DINO, increase `gradient_accumulation` for classifier, reduce `max_tracks` to 32

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

### Temporal and Contrastive Learning
- DynaCLR (2024). "Contrastive Learning of Cell Dynamics." *arXiv:2410.11281*
- Time Arrow Prediction (2024). "Self-supervised pretext task for cell event recognition." *arXiv:2411.03924*

### Multi-Instance Learning
- Ilse et al. (2018). "Attention-based Deep Multiple Instance Learning." *ICML*
- TransMIL (2021). "Transformer-based Correlated MIL." *NeurIPS*
- ACMIL (2024). "Attention-Challenging MIL." *ECCV*

### Early Classification of Time Series
- CALIMERA (2023). "Cost-aware early classification." *Information Processing and Management*
- EARLIEST (2019). "RL-based early classification." *NeurIPS*
- Early-Exit DNN Survey (2024). *ACM Computing Surveys*

### Bacterial Image Analysis
- DeepBacs (2022). "Multi-task bacterial image analysis." *Communications Biology*
- DeLTA 2.0 (2022). "Deep learning for bacterial cell tracking." *PLOS Computational Biology*

### Confidence Calibration
- Guo et al. (2017). "On Calibration of Modern Neural Networks." *ICML*
