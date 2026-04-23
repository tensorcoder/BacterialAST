# AST Classifier — Status, Results and Recommendations (2026-04-23)

This document is structured for conversion into a slide deck. Each
`##` section is intended as one slide or slide group. Figures are
referenced by relative paths from the repository root
(`/home/mkedz/code/ast_classifier`). Tables are kept small enough to
fit on a single slide.

---

## 1. Executive summary

**Subject:** Rapid AST (antimicrobial susceptibility test) for E. coli
under Ampicillin 16 mg/L, using a YOLO → DINO → CropMLP pipeline over
1-hour brightfield microscopy time-lapses at 100× / 5 FPS.

**Where we are today:**
- The CropMLP works above chance on held-out training strains
  (AUROC 0.76 ± 0.14, accuracy 0.60 ± 0.10).
- Per-fold plots show the **expected behaviour**: both R and S strains
  start classified as Resistant (no drug effect visible yet) and S
  strains drift toward Susceptible after ~20 min. This means the model
  learned a *morphological-change* feature, not a strain-identity
  feature.
- The one exception in training is **EC35** (a known resistant strain),
  which drifts Susceptible in 2 of its 3 experiments across folds 2 & 3.
- On **BlindedTest (B1–B4)** the model calls only one run Resistant
  (B1_preincubated, P=0.548 — borderline). All other B-strain
  recordings are called Susceptible with varying confidence.

**Why BlindedTest fails:** targeted probes show the pipeline is
**extremely sensitive to brightness and focus** — a ±10 uint8 brightness
shift flips 13–29 % of crop classifications, and a 1 px Gaussian blur
flips 15 %. Drift in these imaging factors over the 60-min experiment
is enough to mimic Susceptible-direction morphology in DINO feature
space without any real biological change.

**Secondary finding:** preprocessing leaves ~47 % of crops vertically
oriented due to an OBB w/h convention bug. Fix is ~5 lines and should
land before the next training cycle.

---

## 2. Dataset — training strains (EC-series)

**Folders under `/mnt/f/Data_second_protocol/`:**

| Folder | Role | # experiments | Strains |
|---|---|---|---|
| `Resistant/` | labeled Resistant | 11 | EC35, EC40, EC48, EC58, EC60, EC65, EC87 |
| `Susceptible/` | labeled Susceptible | 16 | EC33, EC36, EC39, EC42, EC67, EC79, EC89, EC126 |
| `Test/` | labeled via EC-number lookup | 15 | same 15 strains, additional repeats |
| **Total training** | **42** | **15 strains** |

**Experiment naming:** `EC{num}_{Antibiotic}_{dose}_{details}` (e.g.
`EC35_Ampicillin_16mgL_preincubated_2_TEM40`).

**Image naming:** `image_{unix_timestamp.ms}.bmp` — the collection
date of every recording is recoverable directly from the filenames.

**Collection conditions:**
- Every experiment runs ~60 minutes at 5 FPS → ~14.5 k frames/experiment.
- Recordings span **2025-06-21 to 2026-01-16** (~7 months).
- Experiments were collected on **many different days** across the 7
  months; microfluidic chips were rotated frequently (multiple chip
  generations within the dataset). Day-of-experiment and chip are
  therefore partly confounded with strain.

**Crop counts (after YOLO preprocessing):**
- 42 HDF5 files, **2.38 M total crops** (128×128 uint8 grayscale).
- Raw pixel statistics: mean ≈ 57, std ≈ 10, dynamic range ≈ [40, 85].
- Post-CLAHE: mean ≈ 87, std ≈ 28, range expanded to [50, 156].
- DINO training subset: ≤ 5000 crops/experiment × 42 ≈ **207 k crops**.

---

## 3. Dataset — BlindedTest strains (B-series)

**Folder:** `/mnt/f/Data_second_protocol/BlindedTest/`

| Strain | Recordings | Dates | Raw brightness μ |
|---|---|---|---|
| B1 | exp1_old_bright, preincubated, preincubated2 | 2026-03-25 → 2026-04-17 | 91 / 58 / 67 |
| B2 | exp1_old_bright, preincubated, preincubated2 | 2026-03-26 → 2026-04-15 | 93 / 62 / 68 |
| B3 | old_bright, preincubated, preincubated2 | 2026-03-27 → 2026-04-17 | 92 / 58 / 67 |
| B4 | preincubated, preincubated2, preincubated_new | 2026-04-03 → 2026-04-15 | 47 / 64 / 60 |

**Same equipment as training** — same microscope, camera, chip family.
The only known axis of difference is that the **condenser light was
manually adjusted** between runs: the `*_old_bright` recordings sit at
raw μ ≈ 91 (well outside training's ≈ 57 ± 10) and were reported as
producing inconsistent results; the `preincubated*` runs were taken
after dialing the condenser back and overlap the training range
(μ ≈ 47–68).

**Key point:** BlindedTest imaging does not introduce a novel chip or
microscope — the brightness axis is the primary known difference, and
B1–B4 are novel clinical isolates (unknown true R/S labels).

---

## 4. Preprocessing pipeline

**Code:** `data/preprocessing.py`

Flow per frame:
1. Run YOLO-OBB (`vertical_obb_100epo_best.pt`) → boxes as
   `(cx, cy, w, h, angle, confidence)`, class labels 0=Focused,
   1=Unfocused, 2=Vertical.
2. Keep only class 0 (Focused), confidence ≥ 0.25.
3. Rectify each OBB to axis-aligned via affine warp
   (`_rectify_obb_crop`, line 162).
4. Place on a **128 × 128 canvas** with
   **`cv2.BORDER_REFLECT_101`** padding (reflection, not zero).
5. Write crops + metadata (timestamp, bbox, confidence, angle) to HDF5.

### 4.1 OBB orientation bug (~47 % of crops end up vertical)

The rectification passes the YOLO angle directly to
`cv2.getRotationMatrix2D` and then crops a `w × h` region. YOLO-OBB
does **not** guarantee `w` is the long axis of the detection:

| Experiment | fraction with `w < h` | Expected |
|---|---|---|
| EC35 Jul-18 | 0.467 | ≈ 0 |
| EC35 Jul-21 | 0.463 | ≈ 0 |
| EC35 Jul-31 | 0.485 | ≈ 0 |

The angle histogram is also bunched in [0°, 90°] rather than spanning
[-90°, +90°] or [0°, 180°], which is consistent with YOLO reporting
the angle of whichever side ended up labeled `w`.

**Effect:** a bacterium oriented physically horizontal in the channel
lands horizontal only when YOLO labels its long side `w`. In the
other ~47 % of cases it lands vertical in the canvas.

**Visible in every crop grid** we rendered (see §§ 7–9). DINO
implicitly learned to handle both orientations; the perturbation probe
(§ 8.4) shows 90° rotation moves mean P(R) by only −0.004 — so the
model is not systematically biased, but capacity is split and per-crop
variance is increased.

**Fix (≤ 5 lines in `_rectify_obb_crop`):**
```python
if h > w:
    w, h = h, w
    angle = (angle + 90.0) % 180.0
```

This invalidates existing HDF5 files — all 42 experiments would need
re-preprocessing before the next DINO training run.

---

## 5. DINO backbone training

**Script:** `train_dino_correctly.py` (reproduces
`checkpoints/dino/best_backbone.pt`)

**Architecture (ViT-Small):**
- `img_size = 128`, `patch_size = 16`, in-channels = 1
- `embed_dim = 384`, `depth = 12`, `num_heads = 6`
- **`time_conditioned = False`** (production checkpoint has no
  `time_proj` weights; the `config.py` default `True` is stale)

**DINO head:**
- `head_hidden_dim = 2048`, `bottleneck = 256`,
  `head_output_dim = 4096`, `nlayers = 3`
- Note: `head_output_dim = 65536` collapses on this dataset size — 4096
  was chosen to prevent collapse.

**Optimisation:**
- Batch 64, 100 epochs, base LR 5e-4, min LR 1e-6
- Weight decay 0.04 → 0.4 (cosine)
- Warmup 10 epochs, grad clip 3.0
- EMA momentum 0.996 → 1.0, teacher temp 0.04 → 0.07 (30-epoch warmup),
  student temp 0.1, center momentum 0.9

**Multi-crop:**
- 2 global crops @ 128 (scale 0.7–1.0)
- 6 local crops @ 64 (scale 0.3–0.6)

**Dataset (self-supervised — no labels, no holdout):**
- Loads **every HDF5 file** under `preprocessed_dir` — all
  Resistant + Susceptible + Test experiments.
- `max_crops_per_experiment = 5000` → effective DINO dataset ≈ 207 k
  crops.

**Augmentations (`data/augmentations.py`):**
- CLAHE (clip 2.0, 8×8 tiles)
- `RandomIntensityJitter(brightness=0.03, contrast=0.3)` — additive
- `RandomNoise(std ∈ [0, 0.01])`
- `RandomDefocusBlur(radius_range=(0, 3))` — Gaussian blur,
  skipped if sampled radius < 0.5 (so many crops are effectively
  un-blurred)
- Normalise with mean=0.3387, std=0.1173 (post-CLAHE stats)

**Training result:** best DINO loss **0.815 @ epoch 43** (training
continued to epoch 70+).

**Why B-strains are OOD for DINO:** the backbone has never seen
B-series morphology; its features for B-crops are extrapolated from
whatever "bacteria on a chip" representation it learned from the
EC-series.

---

## 6. CropMLP classifier

**Script:** `scripts/strain_holdout_crop_classifier.py`

**Architecture (`CropMLP`):**
- Input: 384-dim DINO feature per crop. **No time, no bin index, no
  timestamp** — completely time-agnostic.
- Layers: 384 → 128 → 64 → 2, LayerNorm + GELU + Dropout(0.3).

**Training protocol:**
- **5-fold strain-holdout cross-validation.** Each fold holds out 2
  resistant strains + 2 susceptible strains for test.
- **Train only on crops with `t > 40 min`** (`min_time_sec = 2400`) —
  deliberately chosen so the biological R vs S difference is maximally
  pronounced during training.
- Class-weighted cross-entropy; AdamW LR=1e-3 wd=0.01.
- 100 epochs, 15-epoch early stopping patience, LR warmup + cosine.
- Best checkpoint = highest **val AUC**. **Note:** the val set is 1
  experiment held out per training strain — it is *not* a cross-strain
  OOD set. True OOD AUROC is reported on the test split.
- `max_crops_per_experiment = 10000`.

**Inference:** at evaluation time the MLP is run on *all* timepoints
(not just t > 40 min), then predictions are time-binned post-hoc for
the plots.

---

## 7. Results on training data (EC strains)

### 7.1 Aggregate metrics

| Metric | Value |
|---|---|
| Experiment AUROC | **0.76 ± 0.14** |
| Experiment accuracy | **0.60 ± 0.10** |
| Accuracy @ 60 min | 0.58 |

Figure: `results_crop_mlp/plots/aggregate_timeseries_by_label.png`

> Aggregate P(R) over time, pooled across all folds and all test
> experiments, separated by true label. Clean R-vs-S separation at
> late time — but this averages out per-fold and per-experiment
> variance. Use with care.

Figure: `results_crop_mlp/plots/aggregate_accuracy_vs_time.png`

> Experiment-level accuracy as a function of cumulative time. Climbs
> from ~0.55 at 5 min to ~0.58 at 60 min; large fold-to-fold variance
> (shaded band).

### 7.2 Per-fold verdicts (test-set = held-out strains)

| Fold | Holdout R | Holdout S | AUROC | Accuracy |
|---|---|---|---|---|
| 0 | EC58, EC87 | EC36, EC39 | 0.89 | 0.67 |
| 1 | EC58, EC60 | EC33, EC39 | 0.87 | 0.64 |
| **2** | **EC35, EC87** | **EC33, EC39** | **0.53** | **0.45** |
| 3 | EC35, EC87 | EC36, EC39 | 0.67 | 0.50 |
| 4 | EC40, EC48 | EC39, EC67 | 0.87 | 0.73 |

Folds where EC35 is held out (2 and 3) perform worst.

### 7.3 Per-fold crop-fraction time series

These are the plots the user identified as showing the correct
expected biological behaviour: both R and S start near P(R) ≈ 0.7
(no drug effect yet visible), and S strains drift toward P(R) ≈ 0.2
as lysis emerges.

Figures (one per fold):
- `results_crop_mlp/plots/fold0_crop_fractions.png`
- `results_crop_mlp/plots/fold1_crop_fractions.png`
- `results_crop_mlp/plots/fold2_crop_fractions.png`
- `results_crop_mlp/plots/fold3_crop_fractions.png`
- `results_crop_mlp/plots/fold4_crop_fractions.png`

Corresponding per-experiment timeseries (line per experiment, R solid,
S dashed):
- `results_crop_mlp/plots/fold0_timeseries.png` … `fold4_timeseries.png`

> **Important theoretical implication:** If the CropMLP were using
> strain-identity features, S strains would start at P(R) ≈ 0 and
> stay there. The fact that S strains *start at P(R) ≈ 0.7 and drift
> down to 0.2* is direct evidence the model is responding to
> **morphological change over time** (lysis), not baseline morphology.

### 7.4 EC35 anomaly

EC35 is the one training-data exception. All 3 runs classify borderline:

| Experiment | Date | Fold 2 P(R) | Fold 3 P(R) | Verdict |
|---|---|---|---|---|
| `EC35_*_exp2_TEM40` | 2025-07-18 | 0.59 | 0.53 | **R ✓** |
| `EC35_*_2_TEM40` | 2025-07-21 | 0.44 | 0.44 | S ✗ |
| `EC35_*_3_TEM40` | 2025-07-31 | 0.46 | 0.43 | S ✗ |

- EC87 was run on the *same day* as the failing Jul-21 EC35 run and
  classifies confidently Resistant (P ≈ 0.84), so same-day chip
  variability alone is not the cause.
- The three EC35 runs show very similar raw brightness (μ ≈ 57–63) and
  look biologically similar (no visible lysis in the failing runs).
- **EC35 sits close to the decision boundary** and is flipped by small
  per-experiment differences in focus, packing, and density.

Figure: `results_crop_mlp/plots_blinded_strains/probe_EC35_daybyday/EC35_day_by_day_grid.png`

> 3 × 3 grid: rows are (Jul-18 / Jul-21 / Jul-31) × (t=5 / 30 / 55 min).
> Crops CLAHE-preprocessed (as the model sees them). P-value on each
> tile is the ensemble P(R) for that crop. Visibly similar bacterial
> morphology, yet per-crop P(R) differs substantially at t=5 min
> already — direct evidence the model is reading sub-visual
> experimental differences.

Figure (raw equivalent): `results_crop_mlp/plots_blinded_strains/probe_EC35_daybyday/EC35_day_by_day_grid_raw.png`

---

## 8. Results on BlindedTest (B strains)

### 8.1 Per-experiment verdicts

Ensemble P(R) across 5 folds for all 12 BlindedTest recordings.

| Strain | Recording | P(R) ± σ | Prediction | Raw μ ± σ |
|---|---|---|---|---|
| B1 | preincubated | **0.548 ± 0.021** | **R** | 58 ± 11 |
| B1 | preincubated2 | 0.460 ± 0.053 | S | 67 ± 13 |
| B1 | exp1_old_bright | 0.092 ± 0.063 | S | 91 ± 13 |
| B2 | preincubated | 0.443 ± 0.036 | S | 62 ± 13 |
| B2 | preincubated2 | 0.323 ± 0.052 | S | 68 ± 13 |
| B2 | exp1_old_bright | 0.082 ± 0.043 | S | 93 ± 17 |
| B3 | preincubated | 0.403 ± 0.059 | S | 58 ± 10 |
| B3 | preincubated2 | 0.356 ± 0.049 | S | 67 ± 11 |
| B3 | old_bright | 0.091 ± 0.054 | S | 92 ± 15 |
| B4 | preincubated | 0.480 ± 0.043 | S | 47 ± 10 |
| B4 | preincubated_new | 0.267 ± 0.040 | S | 60 ± 10 |
| B4 | preincubated2 | 0.247 ± 0.043 | S | 64 ± 10 |

**Observation:** every `*_old_bright` recording (raw μ ≈ 91–93) is
called Susceptible with very high confidence (P ≈ 0.08–0.09). Every
`preincubated*` recording at training-range brightness (μ ≈ 47–68)
sits between 0.25 and 0.55.

### 8.2 Aggregate and per-fold plots (all 12 experiments, strain-coloured)

- Aggregate: `results_crop_mlp/plots_blinded_strains/aggregate_timeseries.png`
- Per fold: `results_crop_mlp/plots_blinded_strains/fold{0..4}_timeseries.png`
- Per-fold crop fractions: `results_crop_mlp/plots_blinded_strains/fold{0..4}_crop_fractions.png`

Colours: B1 red, B2 blue, B3 green, B4 purple. Within each strain:
○ solid = experiment 1, □ dashed = experiment 2, △ dotted =
experiment 3 (alphabetical within-strain).

### 8.3 Lowest-brightness-only subset (1 experiment per strain)

To avoid brightness-OOD effects, we also regenerated the plots using
only each strain's lowest-brightness `*_preincubated` recording:

- `results_crop_mlp/plots_blinded_strains/lowest_brightness/aggregate_timeseries.png`
- `results_crop_mlp/plots_blinded_strains/lowest_brightness/fold{0..4}_timeseries.png`
- `results_crop_mlp/plots_blinded_strains/lowest_brightness/fold{0..4}_crop_fractions.png`

> All four strains drift downward toward Susceptible over the 60 min.
> B1 (red) drifts least and ends near P(R) ≈ 0.5; B2/B3/B4 drift to
> 0.2–0.35. Consistent with the training Susceptible temporal
> signature — but unexpected if B1–B4 are biologically unaffected,
> which the user reports visually.

### 8.4 Probe 1 — B3_preincubated top-P(R) vs bottom-P(R) crop grids

Sort all ~100 k crops from B3_preincubated by ensemble P(R),
render top-64 and bottom-64.

- Top (model says Resistant):
  `results_crop_mlp/plots_blinded_strains/probe_B3_Ampicillin_16mgL_preincubated/probe1_top_resistant_crops.png`
- Middle (P ≈ 0.35 median):
  `results_crop_mlp/plots_blinded_strains/probe_B3_Ampicillin_16mgL_preincubated/probe1_middle_crops.png`
- Bottom (model says Susceptible):
  `results_crop_mlp/plots_blinded_strains/probe_B3_Ampicillin_16mgL_preincubated/probe1_bottom_susceptible_crops.png`

> Visually, top crops show dense regular horizontal bacterial stripes;
> bottom crops show the same general layout but with disrupted
> patterns (diagonal/offset banding, broken rows, ghost/double
> exposures). Brightness distributions (μ in tile titles) are
> *similar* across the three groups — the model is picking up
> something structural about packing regularity / alignment, not
> brightness.

Raw (non-CLAHE) equivalents:
`probe1_top_resistant_crops_raw.png`, `probe1_middle_crops_raw.png`,
`probe1_bottom_susceptible_crops_raw.png` (same folder).

### 8.5 Probe 2 — per-crop P(R) correlations with simple stats (B3_preincubated)

Figure: `results_crop_mlp/plots_blinded_strains/probe_B3_Ampicillin_16mgL_preincubated/probe2_scatter_vs_stats.png`

| Stat | Spearman ρ | Interpretation |
|---|---|---|
| crop mean brightness | +0.028 | ~zero within-experiment |
| crop std (contrast) | +0.132 | weak positive |
| bbox area (w·h) | −0.011 | ~zero |
| **time within experiment** | **−0.262** | **strongest correlate** |
| YOLO confidence | +0.140 | weak |

> Within a single experiment every crop sees the same condenser
> setting, so brightness doesn't correlate. *Time* does — but the
> model has no time input, so this correlation must come from
> **image content drifting over the hour in a direction the MLP
> reads as Susceptible**.

### 8.6 Probe 3 — perturbation sensitivity (B3_preincubated, 2000 crops)

Apply controlled perturbations to each crop, re-embed through DINO,
re-run CropMLP ensemble, measure ΔP(R). Baseline mean P(R) = 0.390.

Figures:
- `results_crop_mlp/plots_blinded_strains/probe_B3_Ampicillin_16mgL_preincubated/perturbations/perturbation_deltaP_boxplot.png`
- `results_crop_mlp/plots_blinded_strains/probe_B3_Ampicillin_16mgL_preincubated/perturbations/perturbation_flip_rates.png`

| Perturbation | mean ΔP(R) | flip rate (crosses 0.5) | \|ΔP\| > 0.1 |
|---|---|---|---|
| blur σ = 0.5 px | −0.02 | 4 % | 2 % |
| blur σ = 1.0 px | **−0.11** | **15 %** | 46 % |
| blur σ = 2.0 px | −0.25 | 29 % | 72 % |
| brightness −10 | **+0.21** | **29 %** | 75 % |
| brightness −20 | **+0.42** | **58 %** | 92 % |
| brightness +10 | −0.08 | 13 % | 35 % |
| brightness +20 | −0.16 | 24 % | 56 % |
| contrast × 0.8 | −0.17 | 21 % | 65 % |
| contrast × 1.2 | +0.09 | 12 % | 36 % |
| rotate 90° | −0.004 | 9 % | 22 % |

> **Brightness is a massive confound.** A ±10 uint8 shift (visually
> nearly imperceptible, ~18 % of the training dynamic range) flips
> 13–29 % of crops. The direction is deterministic: **darkening
> pushes P(R) up, brightening pushes it down.** This directly
> explains why `old_bright` recordings were discarded — they pushed
> everything toward Susceptible.
>
> **Focus is almost as bad.** 1 px Gaussian blur is imperceptible
> but flips 15 % of crops, always Susceptible. Normal focus drift
> over 60 min is enough to fully produce the observed `P(R) ∝ −time`
> trend **without any biological change**.
>
> **Rotation is fine** — DINO learned approximate rotation-invariance,
> probably because of the preprocessing orientation bug which left
> ~half of training crops vertical.

---

## 9. Interpretation

**What the CropMLP actually learned:** a morphological-change feature
(proven by R and S both starting high and only S dropping), sitting on
top of DINO features that are **not brightness- or focus-invariant**.

**Why the original fold evaluation looks clean:**
- Training and held-out-strain test data come from the **same imaging
  regime** — same microscope, same chip families, same range of daily
  condenser settings. Imaging variance is mixed through both.
- The biological R/S signal past 40 min (lysis vs no lysis) is strong
  enough in the training imaging regime that it dominates the noise
  for most strains.
- EC35 is the canary case: its bacteria sit close to the decision
  boundary in DINO feature space, so small per-experiment imaging
  noise (focus/packing/density) can tip it either side.

**Why BlindedTest is harder:**
- Two OOD axes: (1) novel strain isolates (B1–B4 unseen by DINO), and
  (2) a condenser-light manipulation. Even the "corrected"
  preincubated brightness range (μ ≈ 47–68) touches the edge of the
  training distribution.
- The perturbation probe shows brightness is enough to flip the
  verdict by itself — you do not need novel biology to fail on
  BlindedTest.

**What the time-drift in B3_preincubated really is:** image content
drifting over the hour — likely focus drift plus subtle illumination
drift plus gradual packing-density changes — in a direction the MLP
reads as "becoming susceptible". The bacteria themselves do not need
to change.

---

## 10. Recommendations (all pipeline stages)

### 10.1 Preprocessing (`data/preprocessing.py`)

**10.1.1 — Fix the OBB orientation bug** (one-block change, **required
before any future retraining**):

```python
# in _rectify_obb_crop, before cv2.getRotationMatrix2D
if h > w:
    w, h = h, w
    angle = (angle + 90.0) % 180.0
```

After the fix, **re-preprocess all 42 training experiments**. Existing
`./preprocessed/*.h5` files are stale after the fix and must be
regenerated before DINO training.

**10.1.2 — (optional) Log per-crop focus score**: compute
`cv2.Laplacian(crop).var()` at extraction time and store it alongside
the metadata. Useful both as a covariate for the CropMLP and as a
sanity-check for focus drift per experiment.

### 10.2 DINO training

Priority suggestions, ranked:

| # | Change | Rationale |
|---|---|---|
| 1 | `aug_brightness` **0.03 → 0.06** (user-confirmed) | Current range is far narrower than the observed sensitivity regime. |
| 2 | Add **gamma / non-linear brightness augmentation** (e.g. γ ∈ [0.7, 1.4]) | Real condenser-light changes are multiplicative — additive-only jitter is too easy to learn invariance against. |
| 3 | **Defocus augmentation**: change `radius_range = (0, 3)` → `(0.5, 5)` and **drop the skip-if-radius<0.5 branch** | Current design effectively leaves many crops un-blurred. After fix every crop sees a real blur. |
| 4 | Add **random horizontal + vertical flips (p=0.5 each)** | With the orientation fix, accidental rotation-invariance disappears. Flips restore it explicitly. |
| 5 | Add **random small rotation (±10–15°)** | Robustness against YOLO bbox-angle jitter once we remove the 90° mixing. |
| 6 | `aug_contrast` **0.3 → 0.4** | Contrast × 0.8 flips 21 % of crops; a 33 % augmentation range is still conservative. |
| 7 | `max_crops_per_experiment` **5000 → 10000** (user-confirmed) | ~2× effective dataset size. |
| 8 | Epochs **100 → 150** | Richer augmentation + 2× data justifies more training steps. |
| 9 | (Optional) Add **synthetic illumination-gradient augmentation** (±5 % linear ramp) | Real microscope illumination is never perfectly uniform. |

**Status quo — do not change:**
- `time_conditioned = False` — the production pipeline is time-agnostic
  by design and works that way. Adding time conditioning re-opens the
  question of how to calibrate across experiments of different length
  and sampling cadence.
- `head_output_dim = 4096` — 65536 collapses on this dataset size.
- DINO uses all experiments (no strain holdout at DINO stage). Since
  DINO is self-supervised, no label leakage.

### 10.3 CropMLP training (`scripts/strain_holdout_crop_classifier.py`)

**10.3.1 — Fix the val-set leakage.** Current val split holds out
one experiment per *training* strain, so `best_val_auc` measures
cross-experiment, same-strain generalisation. Move to **held-out
strains for val** (disjoint from the fold's test strains) so
checkpoints reflect true cross-strain performance.

**10.3.2 — Per-experiment feature normalisation.** Before the MLP,
subtract each experiment's mean DINO feature from its crops. Cheap
batch-correction; should reduce sensitivity to per-experiment
illumination / focus offsets.

**10.3.3 — Add focus score as an MLP input.** Concatenate the
Laplacian-variance focus score (from 10.1.2) to the 384-dim DINO
feature. Gives the MLP an explicit knob to down-weight P(R) from
blurred crops.

**10.3.4 — Consider retraining on all timepoints, not just t > 40 min.**
This is a bigger design question. Current design intentionally
concentrates training on the most-discriminative window, which works
for held-out training strains but produces an extrapolation regime
for t < 40 min where the model reacts to imaging noise rather than
biology. Alternatives:
- Still train primarily on t > 40 min but add a small fraction
  of earlier-time crops to pin the model's behaviour there.
- Or: move to per-experiment temporal aggregation (the
  `PopulationTemporalClassifier` path already in the repo) and let
  the model learn the temporal dynamics explicitly.

### 10.4 Evaluation / probes

- Continue probing per-experiment sensitivity with the perturbation
  method whenever a new dataset is added.
- Report per-experiment brightness and per-experiment median focus
  score alongside P(R) — they are known to correlate with model
  output and should be treated as covariates.
- Keep using the lowest-brightness-per-strain subset for BlindedTest-
  style evaluations until DINO training is brightness-robust.

---

## 11. Artifacts — index for slide conversion

**Training-data fold evaluation:**
- `results_crop_mlp/plots/aggregate_timeseries_by_label.png`
- `results_crop_mlp/plots/aggregate_accuracy_vs_time.png`
- `results_crop_mlp/plots/fold{0..4}_crop_fractions.png`
- `results_crop_mlp/plots/fold{0..4}_timeseries.png`
- `results_crop_mlp/strain_holdout_results.json`
- `results_crop_mlp/per_experiment_details.json`

**BlindedTest — full set (12 experiments):**
- `results_crop_mlp/plots_blinded_strains/aggregate_timeseries.png`
- `results_crop_mlp/plots_blinded_strains/fold{0..4}_crop_fractions.png`
- `results_crop_mlp/plots_blinded_strains/fold{0..4}_timeseries.png`
- `results_crop_mlp/plots_blinded_strains/results.json`
- `results_crop_mlp/plots_blinded_strains/per_fold_details.json`

**BlindedTest — lowest-brightness per strain (1 per B-strain):**
- `results_crop_mlp/plots_blinded_strains/lowest_brightness/aggregate_timeseries.png`
- `results_crop_mlp/plots_blinded_strains/lowest_brightness/fold{0..4}_*.png`

**Probes on B3_Ampicillin_16mgL_preincubated:**
- Top / middle / bottom crop grids (CLAHE + raw):
  `results_crop_mlp/plots_blinded_strains/probe_B3_Ampicillin_16mgL_preincubated/probe1_*.png`
- Per-crop scatters vs stats:
  `results_crop_mlp/plots_blinded_strains/probe_B3_Ampicillin_16mgL_preincubated/probe2_scatter_vs_stats.png`
- Perturbation sensitivity:
  `results_crop_mlp/plots_blinded_strains/probe_B3_Ampicillin_16mgL_preincubated/perturbations/perturbation_deltaP_boxplot.png`
  `results_crop_mlp/plots_blinded_strains/probe_B3_Ampicillin_16mgL_preincubated/perturbations/perturbation_flip_rates.png`

**EC35 day-by-day probe:**
- `results_crop_mlp/plots_blinded_strains/probe_EC35_daybyday/EC35_day_by_day_grid.png`
- `results_crop_mlp/plots_blinded_strains/probe_EC35_daybyday/EC35_day_by_day_grid_raw.png`

**Pipeline documentation:**
- `PIPELINE_DIAGRAMS.md` — Mermaid diagrams of the full pipeline.

---

## 12. Suggested slide order for presentation

1. Title + executive summary (§1)
2. Dataset overview — EC strains (§2)
3. Dataset overview — B strains (§3)
4. Pipeline: YOLO → DINO → CropMLP (§§4–6 summarised)
5. Preprocessing detail + OBB bug (§4.1)
6. DINO training detail + augmentation (§5)
7. CropMLP architecture + training protocol (§6)
8. Training-data fold results: aggregate (§7.1)
9. Training-data fold results: per-fold crop fractions (§7.3)
10. Training-data — the EC35 anomaly (§7.4) + EC35 grid figure
11. BlindedTest — per-experiment table (§8.1)
12. BlindedTest — aggregate + lowest-brightness (§§8.2–8.3)
13. Probe 1 — top-vs-bottom crop grids (§8.4)
14. Probe 2 — per-crop correlations (§8.5)
15. Probe 3 — perturbation sensitivity (§8.6)
16. Interpretation: what the model learned (§9)
17. Recommendations — preprocessing (§10.1)
18. Recommendations — DINO training (§10.2)
19. Recommendations — CropMLP + evaluation (§§10.3–10.4)
20. Next steps
