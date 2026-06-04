# CropMLP on ViT-Tiny DINO (5K cap) — **EC35 excluded**

Same backbone and same training method as `results_crop_mlp_tiny_5k/`,
with the EC35 strain (3 R experiments) **dropped from train, val, and
test entirely** before fold generation.

## Why

In the standard CV (`results_crop_mlp_tiny_5k/`), fold 2 — which holds
out EC35 + EC87 (R) vs EC33 + EC39 (S) — was the only fold where AUROC
collapsed to 0.5333 across all three backbones (v1, Tiny 5K, Tiny 10K).
Fold 3, the other fold containing EC35 in the R holdout, also
underperformed (acc 0.500 / AUROC 0.667). This run isolates EC35 as a
potential confounder by removing it from the population entirely.

## What changed

| Field | Standard Tiny 5K | This run |
| --- | --- | --- |
| Features | `features_tiny_5k/` | same |
| Backbone | ViT-Tiny @ 5K crops/exp | same |
| CropMLP architecture | `192 → 128 → 64 → 2` | same |
| Train filter | `rel_ts ≥ 2400 s`, focused-only, max 10K crops/exp | same |
| Optimiser / schedule / CV protocol | AdamW 1e-3, 5-fold, seed 42, 2R+2S holdout | same |
| **Excluded strains** | none | **EC35** |

Implementation: `scripts/strain_holdout_crop_classifier.py` now accepts
`--exclude-strains EC35`, which drops the listed EC strains from the
`build_strain_grouped_experiments` output before `generate_folds` is
called. The fold rotation is therefore drawn from a strictly smaller
R-strain pool, so **the fold compositions differ from the standard run**.

## Folds (this run vs the standard Tiny 5K run)

Because EC35 is no longer in the R pool, the seeded `itertools.combinations`
shuffle picks five different `(R-combo, S-combo)` tuples:

| Fold | Standard Tiny 5K | **This run (no EC35)** |
| ---: | --- | --- |
| 0 | R=EC58,EC87 / S=EC36,EC39 | R=EC48,EC87 / S=EC33,EC42 |
| 1 | R=EC58,EC60 / S=EC33,EC39 | R=EC60,EC65 / S=EC39,EC79 |
| 2 | R=EC35,EC87 / S=EC33,EC39 | R=EC60,EC65 / S=EC36,EC79 |
| 3 | R=EC35,EC87 / S=EC36,EC39 | R=EC48,EC87 / S=EC42,EC89 |
| 4 | R=EC40,EC48 / S=EC39,EC67 | R=EC40,EC48 / S=EC33,EC79 |

This means the comparison is *EC35-included vs EC35-excluded* on different
test sets, not a clean A/B on the same test sets.

## Results

**Headline (mean ± std across 5 folds):**

| Metric | Standard Tiny 5K | **This run (no EC35)** | Δ |
| --- | ---: | ---: | ---: |
| Experiment AUROC | 0.7644 ± 0.1410 | **0.9720 ± 0.0254** | **+0.208** |
| Experiment accuracy | 0.5970 ± 0.1030 | **0.6418 ± 0.0660** | +0.045 |

**Per-fold:**

| Fold | Held-out R | Held-out S | n_test | Accuracy | AUROC |
| ---: | --- | --- | ---: | ---: | ---: |
| 0 | EC48, EC87 | EC33, EC42 | 10 | 0.6000 | 0.9600 |
| 1 | EC60, EC65 | EC39, EC79 | 11 | 0.5455 | 0.9667 |
| 2 | EC60, EC65 | EC36, EC79 | 11 | 0.7273 | **1.0000** |
| 3 | EC48, EC87 | EC42, EC89 | 11 | 0.6364 | 0.9333 |
| 4 | EC40, EC48 | EC33, EC79 | 10 | 0.7000 | **1.0000** |

**Cumulative accuracy vs cumulative exposure time (mean across folds):**

|  5 min | 10 min | 15 min | 20 min | 25 min | 30 min | 40 min | 50 min | 60 min |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.491 | 0.491 | 0.491 | 0.491 | 0.527 | 0.527 | 0.545 | 0.584 | 0.622 |

## Interpretation

1. **AUROC jumps from 0.76 → 0.97** when EC35 is removed.
   The std drops too (0.141 → 0.025) — folds are much more uniformly
   easy. Two of the five folds achieve perfect ranking (AUROC = 1.000).

2. **EC35 was a major confounder in the standard CV.**
   It accounts for ~all of the difficulty in the original fold 2
   (AUROC 0.533) and a large fraction of fold 3 (AUROC 0.667). The
   ranking the model produces on the remaining strains is excellent.

3. **Accuracy at 0.5 threshold is still only 0.642.**
   AUROC 0.972 with accuracy 0.642 means the model ranks experiments
   correctly but the per-experiment mean P(R) doesn't cleanly straddle
   0.5 — a calibration / threshold-tuning issue, not a representation
   issue. A small Platt scaling or even just `threshold = 0.4` would
   likely close most of the accuracy gap.

4. **Caveat: fold composition changed.**
   This is not a clean ablation of EC35 alone, because removing it from
   the R pool also reshuffles the seeded `(R-combo, S-combo)` rotation.
   What you're seeing is the result on a *different distribution of
   held-out strains* that happens to not include EC35. To do a clean
   ablation you'd need to fix the existing folds and either remove EC35
   only from train (still test it) or only from test (still train on it).

## Files

- `strain_holdout_results.json` — summary, schema identical to `results_crop_mlp/`
- `per_experiment_details.json` — per-experiment time-binned P(R)
- `checkpoints/fold{0..4}_best.pt` — 5 MLP weights
- `plots/aggregate_accuracy_vs_time.png` — aggregate accuracy curve
- `plots/aggregate_timeseries_by_label.png` — aggregate mean P(R) by label
- `plots/fold{0..4}_timeseries.png` — per-experiment P(R) over time
- `plots/fold{0..4}_crop_fractions.png` — fraction-R vs time (R vs S panels)

## Reproduce

```bash
PYTHONPATH=/home/mkedz/code ./.venv/bin/python \
  -m ast_classifier.scripts.strain_holdout_crop_classifier \
  --features-dir ./features_tiny_5k \
  --output-dir ./results_crop_mlp_tiny_5k_no_ec35 \
  --exclude-strains EC35 \
  --device cuda:0
```
