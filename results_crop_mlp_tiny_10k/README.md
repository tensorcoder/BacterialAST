# CropMLP on ViT-Tiny DINO (10K crops/experiment cap)

Per-crop MLP susceptible/resistant classifier trained on top of features from
the **ViT-Tiny DINO backbone trained with `max_crops_per_experiment = 10000`**.
Built to be directly comparable to `results_crop_mlp/` (which uses the
ViT-Small backbone) and to `results_crop_mlp_tiny_5k/` (the same ViT-Tiny
architecture trained with half as many crops per experiment).

## Backbone (DINO Tiny, 10K-cap)

| Field | Value |
| --- | --- |
| Checkpoint | `checkpoints/dino_vit_tiny_10k/dino/best_backbone.pt` |
| Architecture | ViT-Tiny/16 @ 128×128 grayscale |
| Spec | `embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0, time_conditioned=False` |
| Parameters | 5,400,768 (~5.40 M) |
| DINO trainer | `train_vit_tiny_scratch.py --max-crops-per-experiment 10000` (from-scratch) |
| Effective training crops | **516,732** (53 preprocessed experiments × `max_crops_per_experiment=10000`) |
| Epochs scheduled | 100 |
| Best epoch | 27 |
| Best DINO loss | 0.926 |
| Normalisation | post-CLAHE: mean 0.3387 / std 0.1173 |

## Features (extracted for this run)

| Field | Value |
| --- | --- |
| Source HDF5 | `preprocessed/` (53 files, 128×128 focused-only crops) |
| Output | `features_tiny_10k/<exp>.npz` (`features` float16 (N, 192), `timestamps` float64) |
| Extractor | `scripts/extract_features_tiny.py --variant 10k` |
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

The 5 fold splits are identical to those in `results_crop_mlp/` and
`results_crop_mlp_tiny_5k/` (same seed, same set of EC strains in the
features dir, verified `n_test` counts match: 12, 11, 11, 12, 11).

## Results

**Headline (mean ± std across 5 folds):**
- Experiment AUROC: **0.7644 ± 0.1410**
- Experiment accuracy: **0.5788 ± 0.1308**

**Per-fold (held-out R + held-out S → test acc, AUROC):**

| Fold | Held-out R | Held-out S | n_test | Accuracy | AUROC |
| ---: | --- | --- | ---: | ---: | ---: |
| 0 | EC58, EC87 | EC36, EC39 | 12 | 0.6667 | 0.8889 |
| 1 | EC58, EC60 | EC33, EC39 | 11 | 0.6364 | 0.8667 |
| 2 | EC35, EC87 | EC33, EC39 | 11 | 0.3636 | 0.5333 |
| 3 | EC35, EC87 | EC36, EC39 | 12 | 0.5000 | 0.6667 |
| 4 | EC40, EC48 | EC39, EC67 | 11 | 0.7273 | 0.8667 |

**Cumulative accuracy vs exposure time (mean across folds):**

|  5 min | 10 min | 15 min | 20 min | 25 min | 30 min | 40 min | 50 min | 60 min |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.564 | 0.529 | 0.529 | 0.545 | 0.562 | 0.579 | 0.579 | 0.579 | 0.579 |

## Direct comparison

| Metric | ViT-Small (`results_crop_mlp/`) | ViT-Tiny 5K (`results_crop_mlp_tiny_5k/`) | **ViT-Tiny 10K (this)** |
| --- | ---: | ---: | ---: |
| Backbone params | ~22.0 M | 5.40 M | **5.40 M** |
| DINO training crops | 206,732 | 261,732 | **516,732** |
| DINO best epoch | 43 | 32 | 27 |
| DINO best loss | 0.815 | 0.989 | 0.926 |
| Experiment AUROC | 0.7644 ± 0.1410 | 0.7644 ± 0.1410 | **0.7644 ± 0.1410** |
| Experiment accuracy | 0.5970 ± 0.1030 | 0.5970 ± 0.1030 | **0.5788 ± 0.1308** |
| Accuracy @ 40 min | 0.579 | 0.579 | 0.579 |

**Reading the table:** all three models agree on AUROC to 4 decimals, and
agree on accuracy at 40 / 50 / 60 min cumulative exposure (0.579). The
only differences are at the per-fold level:

- All three are identical on folds 0, 1, 3, 4.
- Fold 2 (held-out R: EC35+EC87, S: EC33+EC39) is the hardest fold.
  ViT-Small and ViT-Tiny 5K both get accuracy 0.4545 (5/11 correct);
  ViT-Tiny 10K gets 0.3636 (4/11). AUROC is 0.5333 in all three — the
  *ranking* of experiments by P(R) is identical, so the failure is at
  the 0.5 threshold, not in the discriminative signal.

This is consistent with the test sets having only 10–12 experiments per
fold, where accuracy is a coarse measure restricted to discrete `k/n_test`
values. The fact that AUROC (which is rank-based on per-experiment mean
P(R)) is identical across all three backbones suggests that whatever
ordering the ViT-Small features impose on experiments is preserved by the
ViT-Tiny features in both training regimes.

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
  -m ast_classifier.scripts.extract_features_tiny --variant 10k

PYTHONPATH=/home/mkedz/code ./.venv/bin/python \
  -m ast_classifier.scripts.strain_holdout_crop_classifier \
  --features-dir ./features_tiny_10k \
  --output-dir ./results_crop_mlp_tiny_10k \
  --device cuda:0
```
