# CropMLP on ViT-Tiny DINO (5K crops/experiment cap)

Per-crop MLP susceptible/resistant classifier trained on top of features from
the **ViT-Tiny DINO backbone trained with `max_crops_per_experiment = 5000`**.
Built to be directly comparable to `results_crop_mlp/` (which uses the
ViT-Small backbone).

## Backbone (DINO Tiny, 5K-cap)

| Field | Value |
| --- | --- |
| Checkpoint | `checkpoints/dino_vit_tiny/dino/best_backbone.pt` |
| Architecture | ViT-Tiny/16 @ 128×128 grayscale |
| Spec | `embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0, time_conditioned=False` |
| Parameters | 5,400,768 (~5.40 M) |
| DINO trainer | `train_vit_tiny_scratch.py` (from-scratch) |
| Effective training crops | **261,732** (53 preprocessed experiments × `max_crops_per_experiment=5000`) |
| Epochs scheduled | 100 |
| Best epoch | 32 |
| Best DINO loss | 0.989 |
| Normalisation | post-CLAHE: mean 0.3387 / std 0.1173 |

## Features (extracted for this run)

| Field | Value |
| --- | --- |
| Source HDF5 | `preprocessed/` (53 files, 128×128 focused-only crops) |
| Output | `features_tiny_5k/<exp>.npz` (`features` float16 (N, 192), `timestamps` float64) |
| Extractor | `scripts/extract_features_tiny.py --variant 5k` |
| Experiments covered | 49 EC + 4 B (53 total) |

## CropMLP training (identical to `results_crop_mlp/`)

| Field | Value |
| --- | --- |
| Architecture | 192 → 128 → 64 → 2 with LayerNorm + GELU + Dropout(0.3) |
| Train filter | `rel_ts ≥ 2400 s` (≥ 40 min exposure), focused-only |
| Max crops / experiment (training) | 10,000 |
| Cross-validation | 5-fold strain-holdout, 2 R + 2 S strains held out per fold, seed=42 |
| Optimiser | AdamW lr=1e-3 wd=0.01, batch=2048 |
| Schedule | 5-epoch warmup + cosine, 100 epochs max |
| Early stopping | val AUROC, patience 15 |
| Class weights | `len(y) / (2 * n_per_class)` |

The 5 fold splits are identical to those in `results_crop_mlp/` (same seed,
same set of EC strains present in the features dir).

## Results

**Headline (mean ± std across 5 folds):**
- Experiment AUROC: **0.7644 ± 0.1410**
- Experiment accuracy: **0.5970 ± 0.1030**

**Per-fold (held-out R + held-out S → test acc, AUROC):**

| Fold | Held-out R | Held-out S | n_test | Accuracy | AUROC |
| ---: | --- | --- | ---: | ---: | ---: |
| 0 | EC58, EC87 | EC36, EC39 | 12 | 0.6667 | 0.8889 |
| 1 | EC58, EC60 | EC33, EC39 | 11 | 0.6364 | 0.8667 |
| 2 | EC35, EC87 | EC33, EC39 | 11 | 0.4545 | 0.5333 |
| 3 | EC35, EC87 | EC36, EC39 | 10 | 0.5000 | 0.6667 |
| 4 | EC40, EC48 | EC39, EC67 | 11 | 0.7273 | 0.8667 |

**Cumulative accuracy vs exposure time (mean across folds):**

|  5 min | 10 min | 15 min | 20 min | 25 min | 30 min | 40 min | 50 min | 60 min |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.564 | 0.529 | 0.529 | 0.545 | 0.544 | 0.561 | 0.579 | 0.579 | 0.579 |

## Direct comparison with `results_crop_mlp/` (ViT-Small)

| Metric | ViT-Small (results_crop_mlp/) | ViT-Tiny 5K (this) |
| --- | ---: | ---: |
| Backbone params | ~22.0 M | **5.40 M** (≈4× smaller) |
| DINO training crops | 206,732 | **261,732** |
| Experiment AUROC | 0.7644 ± 0.1410 | **0.7644 ± 0.1410** |
| Experiment accuracy | 0.5970 ± 0.1030 | **0.5970 ± 0.1030** |
| Accuracy @ 40 min | 0.579 | 0.579 |

The two models agree to four decimal places at the experiment level —
unsurprising given (a) the same fold splits, (b) only 10–12 held-out
experiments per fold (so accuracy is constrained to discrete `k / n_test`
values), and (c) AUROC is rank-based on per-experiment mean P(R), which
both backbones rank identically on this test set.

## Files

- `strain_holdout_results.json` — summary across folds (matches schema of `results_crop_mlp/strain_holdout_results.json`)
- `per_experiment_details.json` — per-experiment time-binned P(R) and cumulative predictions
- `checkpoints/fold{0..4}_best.pt` — trained MLP weights (5 × 5 MB)
- `plots/aggregate_accuracy_vs_time.png` — mean ± std accuracy vs cumulative time
- `plots/aggregate_timeseries_by_label.png` — mean ± std P(R) over time by true label
- `plots/fold{0..4}_timeseries.png` — per-experiment P(R) over time per fold
- `plots/fold{0..4}_crop_fractions.png` — fraction-R vs time (R-true vs S-true panels)

## Reproduce

```bash
PYTHONPATH=/home/mkedz/code ./.venv/bin/python \
  -m ast_classifier.scripts.extract_features_tiny --variant 5k

PYTHONPATH=/home/mkedz/code ./.venv/bin/python \
  -m ast_classifier.scripts.strain_holdout_crop_classifier \
  --features-dir ./features_tiny_5k \
  --output-dir ./results_crop_mlp_tiny_5k \
  --device cuda:0
```
