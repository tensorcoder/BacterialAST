# Context Recovery Summary (March 2026)

This file captures the full state of the project as of 2026-03-20 for context recovery.

## What This Project Does

Rapid antimicrobial susceptibility testing (AST) of *E. coli* using time-lapse brightfield microscopy at 100x, 5 FPS, 1-hour experiments with 16 mg/L ampicillin. The pipeline detects bacteria with YOLO, extracts DINO self-supervised features, and classifies experiments as resistant or susceptible.

## Data

- **Source:** `/mnt/f/Data_second_protocol` with `Resistant/`, `Susceptible/`, `Test/` subfolders
- **42 experiments** total: 11R, 16S, 15 Test (labels inferred from strain EC number)
- **15 strains:** 7 resistant (EC35, EC40, EC48, EC58, EC60, EC65, EC87), 8 susceptible (EC126, EC33, EC36, EC39, EC42, EC67, EC79, EC89)
- Each experiment: ~14,500 BMP frames in `images/` subfolder
- YOLO weights: `/mnt/c/users/mkedz/Documents/PhD/PhD_code/yolo11/vertical_obb_100epo_best.pt`

## Preprocessing Versions

| Version | Dir | Crop size | Padding | Total crops | Status |
|---------|-----|-----------|---------|-------------|--------|
| v1 | `preprocessed/` | 128x128 | Reflection | 2,378,824 | Complete, **best features** |
| v2 | `preprocessed_v2/` | 96x96 | Zero | 2,099,023 | Complete, features broken |

## DINO Versions

| Version | Checkpoint dir | Crops | Time conditioning | Loss (best) | Features dir |
|---------|---------------|-------|-------------------|-------------|-------------|
| v1 | `checkpoints/dino/` | 128x128 reflect | Continuous 0.2s | 0.815 (ep44) | `features/` |
| v2 | (overwritten) | 96x96 zero-pad | Continuous 0.2s | 0.033 | `features_v2/` |
| v3 | `checkpoints_v3/dino/` | 96x96 zero-pad | 5-min quantized bins | 0.003 (ep26) | `features_v3/` |

**v1 is the only model that learned meaningful morphological features.** v2/v3 learned zero-padding silhouette shortcuts (85-90% of pixels are zero in 96x96 crops).

### Config change made for v3
- Added `time_quantize_sec: float = 300.0` to `DINOConfig` in `config.py`
- Added `time_quantize_sec` parameter to `ViTSmall.__init__()` in `models/backbone.py`
- Quantization happens in `ViTSmall._embed()`: `time = (time // self.time_quantize_sec) * self.time_quantize_sec`
- Passed through in `training/train_dino.py`, `training/extract_features.py`, `scripts/train_dino_holdout.py`, `scripts/train.py`

## Classifier Results (All on v1 features unless noted)

### Population Temporal Classifiers (strain-holdout CV, 5 folds)

| Variant | Results dir | AUROC@60min |
|---------|------------|-------------|
| Baseline (Transformer + Stats) | `results_strain_holdout/` | **0.802 +/- 0.092** |
| Delta features | `results_strain_holdout_delta/` | 0.744 +/- 0.081 |
| Sub-sequence sampling | `results_strain_holdout_subseq/` | 0.739 +/- 0.265 |
| BiLSTM | `results_strain_holdout_lstm/` | 0.656 +/- 0.202 |
| Stats + Auxiliary loss | `results_strain_holdout_stats_aux/` | 0.603 +/- 0.270 |
| Contextual auxiliary | `results_strain_holdout_ctx_aux/` | 0.559 +/- 0.148 |
| Attention bin encoder | `results_strain_holdout_attention/` | 0.500 (collapsed) |
| Attention + Auxiliary | `results_strain_holdout_attn_aux/` | 0.500 (collapsed) |
| Pop. temporal on v2 feats | `results_strain_holdout_v2/` | 0.564 +/- 0.200 |
| Pop. temporal on v3 feats | `results_strain_holdout_v3/` | 0.643 +/- 0.198 |

### Per-Crop MLP Classifiers (strain-holdout CV, 5 folds)

| Features | Results dir | AUROC@60min |
|----------|------------|-------------|
| v1 (128x128 reflect) | `results_crop_mlp/` | **0.764 +/- 0.141** |
| v2 (96x96 zero-pad) | `results_crop_mlp_v2/` | 0.562 +/- 0.277 |
| v3 (96x96 zero-pad + time quant) | `results_crop_mlp_v3/` | 0.661 +/- 0.232 |

### Per-Crop MLP v1 Per-Fold Breakdown

| Fold | Holdout R | Holdout S | AUROC |
|------|-----------|-----------|-------|
| 0 | EC58, EC87 | EC36, EC39 | 0.889 |
| 1 | EC58, EC60 | EC33, EC39 | 0.867 |
| 2 | EC35, EC87 | EC33, EC39 | 0.533 |
| 3 | EC35, EC87 | EC36, EC39 | 0.667 |
| 4 | EC40, EC48 | EC39, EC67 | 0.867 |

EC35 folds consistently underperform.

## Critical Insight: Trajectory Evaluation

**Aggregate accuracy is the WRONG metric.** The user identified that susceptible bacteria should initially look resistant (antibiotic hasn't acted yet) and progressively shift to susceptible over the experiment. The correct evaluation is tracking P(resistant) over time per experiment:

- **v1 crop MLP trajectories** (in `results_crop_mlp/plots/`):
  - Susceptible: P(R) drops from 0.65 → 0.17 over 60 minutes
  - Resistant: P(R) stays flat at ~0.70
  - Curves cross 0.5 at ~25-30 minutes
  - This is the clearest evidence the DINO features work

- **v2/v3 crop MLP trajectories**: Both classes stuck at ~0.50, completely overlapping, no signal

- **v3 population temporal trajectories** (in `plots_v3_trajectories/`): Classifiers output per-fold constants with no variation over time or between classes

Trajectory plots are in:
- `results_crop_mlp/plots/fold{0-4}_timeseries.png` — per-fold per-experiment P(R) over time (v1)
- `results_crop_mlp/plots/aggregate_timeseries_by_label.png` — mean P(R) by class (v1)
- `results_crop_mlp_v2/plots/` — same for v2 (no signal)
- `results_crop_mlp_v3/plots/` — same for v3 (no signal)
- `plots_v3_trajectories/` — v3 population temporal classifier trajectories (no signal)
- `plots_v3_quantized_time/` — aggregate v3 results and v2 vs v3 comparison

## DINO Training: What Worked and What Didn't

### dino_old: Mode Collapse (checkpoints/dino_old/)

The first 100-epoch DINO run collapsed — loss stuck at ln(65536) = 11.09 from epoch 2 onward. The model output a uniform distribution over all prototypes, meaning it could not commit any crop to any visual concept. Three problems contributed:

1. **Wrong normalisation.** Used ImageNet defaults (mean=0.5, std=0.25) instead of the actual data statistics. Raw microscopy crops have mean=57, std=10 in uint8 — after dividing by 255 and applying the wrong normalisation, all inputs were compressed to the narrow band [-1.25, -0.67] with std=0.16. Every crop looked nearly identical to the network.

2. **Too many prototypes.** `head_output_dim=65536` for ~207K training crops gives ~3 crops per prototype. Each prototype in DINO is a learnable visual cluster centre — the projection head outputs a temperature-sharpened softmax distribution over all prototypes. With so few examples per prototype, the centering and sharpening mechanisms cannot produce peaked distributions, and the loss settles at the entropy of a uniform distribution: ln(65536) = 11.09.

3. **Overly aggressive augmentation.** Brightness jitter of ±0.3 was ~4x the data's actual 0.145 dynamic range, and noise std of 0.05 was large relative to the signal. Augmentation dominated the real variation between crops.

### dino v1: Successful Training (checkpoints/dino/)

Fixed all three issues:

| Parameter | dino_old | dino v1 |
|-----------|----------|---------|
| CLAHE preprocessing | None | clipLimit=2.0, tileGrid=8x8 |
| Normalisation | mean=0.5, std=0.25 | mean=0.3387, std=0.1173 (post-CLAHE) |
| head_output_dim | 65536 | 4096 (~50 crops/prototype) |
| Brightness jitter | ±0.3 | ±0.03 |
| Noise std | 0.05 | 0.01 |

Adding CLAHE expanded the dynamic range from [40, 85] to [50, 156] before normalisation, giving the network more contrast to work with. Reducing prototypes to 4096 allowed each to represent a meaningful morphological pattern. Best loss reached 0.815 at epoch 44, well below ln(4096) = 8.32, confirming non-trivial structure was learned.

**On brightness jitter and noise:** even the reduced values (0.03 jitter, 0.01 noise) are questionable. The microscopy setup has controlled, fixed illumination — there is no real brightness variation between experiments to be invariant to. These augmentations force the model to discard subtle intensity features that may carry genuine morphological information (membrane texture, internal structure). The multi-crop strategy (global + local crops at different scales) plus geometric augmentations (flips, rotations) already provide sufficient view diversity for DINO's self-supervised objective. Setting jitter and noise to zero may improve feature quality.

### Potential Improvements

1. **Increase `max_crops_per_experiment`.** Currently set to 5000, which subsamples each experiment's ~57K crops down to 5000, giving only ~207K total training crops out of the 2.4M available. Raising this to 20K+ or removing the cap entirely would give DINO far more morphological diversity to learn from.

2. **Reduce prototypes.** 4096 prototypes is likely more than the number of meaningful morphological states bacteria exhibit (normal rod, elongated, blebbing, dividing, lysing — perhaps dozens of real categories). Reducing to 1024-2048 would concentrate the clusters and could produce sharper features. With more training crops (point 1), even 2048 prototypes would have hundreds of examples each.

3. **Remove brightness jitter and noise.** As discussed above, the controlled microscopy illumination means there is no real brightness variation to be invariant to. Setting both to zero may allow the model to retain subtle intensity features that carry morphological information.

4. **Train DINO only on late-time crops (t > 30 min).** After 30 minutes, susceptible bacteria show visible drug-induced morphological changes (elongation, blebbing, lysis) while resistant bacteria still look normal. Training on this subset exposes DINO to the full spectrum of morphologies that matter for classification. No data is lost because the `max_crops_per_experiment` cap (currently 5000) is well below the ~28.5K crops available per experiment after 30 minutes. The model still sees "normal" morphology in the resistant late-time crops, so early-time inference is not impaired.

5. **Use labels during DINO training on late-time crops.** If DINO is restricted to t > 30 min crops (point 4), experiment-level labels become meaningful at the crop level — susceptible crops genuinely look different from resistant crops at this point. Options include supervised contrastive learning (positive pairs from same class, negatives from different class) or adding a classification head alongside the DINO loss. This would directly guide the backbone toward features that distinguish drug-induced morphological change.

6. **Supervised fine-tuning after DINO pretraining.** As a less invasive alternative to point 5: keep DINO unsupervised, then fine-tune the frozen or unfrozen backbone with a classification head on late-time labeled crops as a second stage.

### dino v2/v3: Zero-Padding Shortcut

Trained on 96x96 zero-padded crops. Achieved paradoxically lower loss (0.033 and 0.003) than v1 (0.815) because the task was easier — DINO learned to match augmented views by the unique zero-padding silhouette boundary rather than bacterium morphology. See "Zero-Padding Failure Analysis" below for details.

## Zero-Padding Failure Analysis

96x96 crops with zero-padding: bacteria occupy ~10-15% of pixels, rest is zero. DINO learns to match augmented views by the unique zero-padding silhouette boundary. Evidence:
- Zero pixel fraction per crop: mean 84-90%, border zero fraction: ~100%
- DINO loss paradoxically much lower (0.003-0.033) than v1 (0.815) — easier shortcut task
- Downstream classifiers produce random-level predictions

Bacteria bbox size distribution (from HDF5 metadata `w`, `h` fields):
- P80 of max(w,h) = 66.5 pixels
- A crop size of 64 (4x4=16 patches) would fit ~78% of bacteria without padding
- A crop size of 80 (5x5=25 patches) would fit ~88%

## Outstanding Decisions / Next Steps (as of 2026-03-20)

1. **Crop size fix:** Either go back to reflection padding (known to work) or find optimal tight crop size (64 or 80 px discussed). User wants to preserve shape information, so tight-crop-and-resize was rejected. A crop size fitting 80% of bacteria with the rest center-cropped was the proposed approach.

2. **Re-run trajectory analysis on population temporal classifiers** — the baseline pop. temporal classifier (AUROC 0.802) was never evaluated with trajectory analysis. It may show even better separation than the crop MLP.

3. **Investigate EC35** — this strain drags down performance in every fold it appears in.

## Key Files

- `config.py` — all config dataclasses, `DINOConfig.time_quantize_sec` added here
- `models/backbone.py` — ViTSmall with time quantization in `_embed()`
- `data/preprocessing.py` — YOLO detection, crop extraction, HDF5 storage
- `data/augmentations.py` — CLAHE + microscopy augmentations
- `data/dataset.py` — `PopulationTemporalDataset`, `DINOCropDataset`
- `training/train_dino.py` — DINO pretraining loop
- `training/extract_features.py` — feature extraction with `extract_all_features()`
- `scripts/train.py` — CLI entry point (`--stage dino|extract|classifier|calibrate|all`)
- `scripts/strain_holdout_eval.py` — population temporal classifier strain-holdout CV
- `scripts/strain_holdout_crop_classifier.py` — per-crop MLP strain-holdout CV with trajectory plots
- `scripts/generate_result_plots.py` — comparison plots across variants
- `scripts/train_dino_holdout.py` — DINO training with strain exclusion

## User Preferences

- Prefers biological reasoning over pure ML metrics
- Wants trajectory-based evaluation, not aggregate accuracy
- Values preserving shape information in crops
- Prefers checking results before making architectural changes
- Does not want unnecessary changes or over-engineering
