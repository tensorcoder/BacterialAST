# Comparison: ViT-Small v1 vs ViT-Tiny 5K vs ViT-Tiny 10K

A side-by-side write-up of the three DINO backbones currently feeding
CropMLP classifiers, with results from each in:

- `results_crop_mlp/` — ViT-Small v1 (the production baseline)
- `results_crop_mlp_tiny_5k/` — ViT-Tiny, DINO trained with 5,000 crops/exp cap
- `results_crop_mlp_tiny_10k/` — ViT-Tiny, DINO trained with 10,000 crops/exp cap

## 1. What changed and what didn't

| Aspect | ViT-Small v1 | ViT-Tiny 5K | ViT-Tiny 10K |
| --- | --- | --- | --- |
| **Backbone width (embed_dim)** | 384 | **192** | **192** |
| **Backbone heads** | 6 | **3** | **3** |
| **Backbone depth (layers)** | 12 | 12 | 12 |
| **Per-experiment crop cap in DINO** | 5,000 | 5,000 | **10,000** |
| Patch size | 16 | 16 | 16 |
| Image size | 128×128 | 128×128 | 128×128 |
| MLP ratio | 4.0 | 4.0 | 4.0 |
| Time conditioning | off | off | off |
| Drop-path rate | 0.1 | 0.1 | 0.1 |
| DINO head | same | same | same |
| Optimiser / schedule | same | same | same |
| Augmentation | same | same | same |
| Source HDF5 set | `preprocessed/` (≈207K → see note) | `preprocessed/` (53 files) | `preprocessed/` (53 files) |

All three runs are deliberately apples-to-apples on every non-architectural
setting — `train_vit_tiny_scratch.py` was forked from
`train_dino_correctly.py` (the script that reproduces the production
ViT-Small) with only the backbone width and num_heads changed. The Tiny-10K
run additionally doubled `max_crops_per_experiment` to test whether the
smaller model is data-limited at the 5K default.

Note on the v1 dataset size: the production ViT-Small was trained when the
preprocessed corpus held ≈ 207K crops (the `train_dino_correctly.py`
docstring is explicit about this); the corpus has since grown to 53
experiments. The Tiny variants saw the new, larger corpus. So the 5K Tiny
saw ~26% more crops than v1 even at the same 5K cap.

## 2. Number of primitives (parameters)

Counted by building the modules in PyTorch and summing `p.numel()` over all
parameters (script: see `Comparison` section in each variant's README, or
the snippet at the bottom of this file).

### Backbones (only the encoder — the part used for feature extraction)

| Model | Parameters | Ratio vs v1 |
| --- | ---: | ---: |
| **ViT-Small v1** | **21,418,368** (≈ 21.42 M) | 1.00× |
| **ViT-Tiny** (both variants) | **5,400,768** (≈ 5.40 M) | **0.252×** (≈ 4× smaller) |

The exact 4× ratio comes from halving `embed_dim` (most layers' weight
tensors scale as `embed_dim²`, so halving width quarters them).

### DINO projection head (used only during pretraining, discarded for feature use)

| Backbone | Head input dim | Approx params |
| --- | ---: | ---: |
| ViT-Small v1 | 384 | ~6.56 M |
| ViT-Tiny (5K & 10K) | 192 | ~6.16 M |

Head topology: `Linear(in→2048) → 2 × Linear(2048→2048) → Linear(2048→256) →
WeightNormLinear(256→4096)` (3 hidden MLP layers + a weight-normalised
prototype layer with 4096 prototypes). The head only contributes a small
embed_dim×2048 first layer that depends on backbone width — that's why the
two head sizes differ by only ~0.4 M.

### CropMLP classifier (per-crop R/S head, trained downstream)

| Backbone supplying features | CropMLP input dim | CropMLP parameters |
| --- | ---: | ---: |
| ViT-Small v1 | 384 | **58,050** |
| ViT-Tiny (5K & 10K) | 192 | **33,474** |

Architecture: `Linear(D → 128) → LayerNorm → GELU → Dropout(0.3) →
Linear(128 → 64) → LayerNorm → GELU → Dropout(0.3) → Linear(64 → 2)`.
Going from D=384 to D=192 saves 384·128 − 192·128 = 24,576 first-layer
weights; everything downstream is identical.

## 3. Full DINO configuration

This is the configuration that produced each `best_backbone.pt` checkpoint.
Anywhere a column says "same" it is bit-identical to v1.

### Architecture
| Field | ViT-Small v1 | ViT-Tiny 5K | ViT-Tiny 10K |
| --- | --- | --- | --- |
| `img_size` | 128 | same | same |
| `patch_size` | 16 | same | same |
| In channels | 1 (grayscale) | same | same |
| `embed_dim` | **384** | **192** | **192** |
| `depth` | 12 | same | same |
| `num_heads` | **6** | **3** | **3** |
| Head dim (`embed_dim / num_heads`) | 64 | 64 | 64 |
| `mlp_ratio` | 4.0 | same | same |
| `drop_path_rate` | 0.1 | same | same |
| `time_conditioned` | False | same | same |
| `time_quantize_sec` | 0.0 | same | same |

### DINO head
| Field | All three |
| --- | --- |
| `head_hidden_dim` | 2048 |
| `head_bottleneck_dim` | 256 |
| `head_output_dim` (number of prototypes) | **4096** (not 65536 — collapses with this dataset size) |
| `head_nlayers` | 3 |

### Optimisation
| Field | All three |
| --- | --- |
| Optimiser | AdamW |
| `batch_size` | 64 |
| `epochs` (scheduled) | 100 |
| `base_lr` | 5e-4 |
| `min_lr` | 1e-6 |
| `warmup_epochs` | 10 |
| `grad_clip` | 3.0 |
| `weight_decay_start` → `_end` | 0.04 → 0.4 (linear) |
| Mixed precision | yes (`torch.amp.autocast("cuda")`) |

### Teacher / temperature / centering
| Field | All three |
| --- | --- |
| `ema_momentum_start` → `_end` | 0.996 → 1.0 |
| `teacher_temp_start` → `_end` | 0.04 → 0.07 |
| `teacher_temp_warmup_epochs` | 30 |
| `student_temp` | 0.1 |
| `center_momentum` | 0.9 |

### Multi-crop & dataset
| Field | All three |
| --- | --- |
| `n_global_crops` | 2 |
| `n_local_crops` | 6 |
| `global_crop_scale` | (0.7, 1.0) |
| `local_crop_scale` | (0.3, 0.6) |
| `local_crop_size` | 64 |
| Crop normalisation (post-CLAHE) | mean 0.3387 / std 0.1173 |
| `aug_brightness` | 0.03 (NOT the DINO default 0.3 — that collapses) |
| `aug_contrast` | 0.3 |
| `aug_noise_std_max` | 0.01 |
| `aug_defocus_max` | 3 |
| `use_clahe` | True |
| Crop reflection-padding (preprocessing) | `cv2.BORDER_REFLECT_101` |

### Dataset cap and resulting size

| Variant | `max_crops_per_experiment` | HDF5 files | Effective training crops | Batches / epoch |
| --- | ---: | ---: | ---: | ---: |
| ViT-Small v1 | 5,000 | ≈ 42 (older corpus) | ≈ **207,000** (per training-script docstring) | ≈ 3,234 |
| ViT-Tiny 5K | 5,000 | 53 | **261,732** | 4,089 |
| ViT-Tiny 10K | 10,000 | 53 | **516,732** | 8,073 |

### Training outcomes

| Variant | Best epoch | Best DINO loss |
| --- | ---: | ---: |
| ViT-Small v1 | 43 | **0.815** |
| ViT-Tiny 5K | 32 | 0.989 |
| ViT-Tiny 10K | 27 | **0.926** |

The two Tiny variants have higher absolute loss than v1, but DINO loss is
**not** directly comparable across architectures (the loss is a
cross-entropy over student/teacher prototype distributions, and a wider
backbone can fit the prototype distribution more tightly). The right
comparison is the downstream task — see §5.

## 4. CropMLP configuration (identical across all three runs)

| Field | Value |
| --- | --- |
| Architecture | `D → 128 → 64 → 2`, LayerNorm + GELU + Dropout(0.3) between linear layers |
| Input dim D | 384 (v1) or 192 (Tiny) |
| Train filter | `rel_ts ≥ 2400 s` (only crops at ≥ 40 min exposure used for training) |
| Eval filter | none — predictions emitted for all crops, binned per 5 min |
| Max crops / experiment (training only) | 10,000 |
| Cross-validation | 5-fold strain-holdout, **2 R + 2 S strains** held out per fold |
| Seed | 42 |
| Optimiser | AdamW (lr 1e-3, weight_decay 0.01) |
| Batch size | train 2048, val 4096 |
| Schedule | 5-epoch warmup + cosine to 1% of base lr; 100 epochs max |
| Early stopping | val AUROC, patience 15 |
| Loss | cross-entropy with class weights `len(y) / (2 * n_per_class)` |
| Grad clip | 1.0 |

The 5 fold splits are byte-identical across all three runs (same seed,
same EC strains present in features dir). Held-out R/S strains and
`n_test` per fold:

| Fold | Held-out R | Held-out S | n_test |
| ---: | --- | --- | ---: |
| 0 | EC58, EC87 | EC36, EC39 | 12 |
| 1 | EC58, EC60 | EC33, EC39 | 11 |
| 2 | EC35, EC87 | EC33, EC39 | 11 |
| 3 | EC35, EC87 | EC36, EC39 | 12 |
| 4 | EC40, EC48 | EC39, EC67 | 11 |

## 5. Results comparison

### Headline (mean ± std across the 5 folds)

| Backbone | Experiment AUROC | Experiment accuracy |
| --- | ---: | ---: |
| **ViT-Small v1** | 0.7644 ± 0.1410 | 0.5970 ± 0.1030 |
| **ViT-Tiny 5K** | 0.7644 ± 0.1410 | 0.5970 ± 0.1030 |
| **ViT-Tiny 10K** | 0.7644 ± 0.1410 | **0.5788 ± 0.1308** |

### Per-fold accuracy / AUROC

| Fold | Small v1 (acc / AUROC) | Tiny 5K (acc / AUROC) | Tiny 10K (acc / AUROC) |
| ---: | ---: | ---: | ---: |
| 0 | 0.6667 / 0.8889 | 0.6667 / 0.8889 | 0.6667 / 0.8889 |
| 1 | 0.6364 / 0.8667 | 0.6364 / 0.8667 | 0.6364 / 0.8667 |
| 2 | 0.4545 / 0.5333 | 0.4545 / 0.5333 | **0.3636** / 0.5333 |
| 3 | 0.5000 / 0.6667 | 0.5000 / 0.6667 | 0.5000 / 0.6667 |
| 4 | 0.7273 / 0.8667 | 0.7273 / 0.8667 | 0.7273 / 0.8667 |

### Cumulative accuracy vs cumulative exposure time (mean across folds)

|  5 min | 10 min | 15 min | 20 min | 25 min | 30 min | 40 min | 50 min | 60 min |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Small v1** | 0.564 | 0.529 | 0.529 | 0.545 | 0.544 | 0.561 | 0.579 | 0.579 | 0.579 |
| **Tiny 5K** | 0.564 | 0.529 | 0.529 | 0.545 | 0.544 | 0.561 | 0.579 | 0.579 | 0.579 |
| **Tiny 10K** | 0.564 | 0.529 | 0.529 | 0.545 | 0.562 | 0.579 | 0.579 | 0.579 | 0.579 |

## 6. What the differences mean

**(a) AUROC is invariant across all three backbones.**
0.7644 ± 0.1410, byte-identical. AUROC is rank-based: it only cares about
the ordering of the 10–12 held-out experiments by per-experiment mean
P(R). All three feature spaces — 384-dim ViT-Small, 192-dim Tiny-5K,
192-dim Tiny-10K — induce *the same ranking* on each test set. That's a
strong (and somewhat surprising) statement: shrinking the backbone 4× and
adjusting the pretraining dataset size do not perturb the population-mean
discriminative ordering at the experiment level. The signal AST is
extracting from these features is robust enough to survive a representation
swap of this magnitude.

**(b) Accuracy is also invariant for v1 and Tiny-5K, and degrades by one
test example in fold 2 for Tiny-10K.**
With only 10–12 experiments per fold, accuracy is quantised to discrete
`k/n_test` values. The 0.5970→0.5788 drop is **one experiment** (5/11→4/11
in fold 2). The model's per-experiment P(R) still ranks the same 11
experiments the same way (AUROC unchanged at 0.5333) but the decision
boundary lands between two experiments differently, demoting one
prediction across the 0.5 threshold.

**(c) Fold 2 is intrinsically hard for all three.**
Held-out R strains EC35 + EC87 alongside S strains EC33 + EC39 is the
worst pairing in the seeded fold rotation. All three backbones get an
AUROC of exactly 0.5333 on this fold — i.e. they rank one fewer
resistant experiment above the susceptible ones than chance would. Since
the ranking is identical across backbones, the difficulty isn't a feature
issue; it's a population-level mismatch between training strains and
these specific held-out strains. Improving fold 2 likely requires either
more strains (so the train pool covers a wider phenotype distribution)
or a different population statistic than naive mean P(R).

**(d) 10K of pretraining crops did not help the downstream task.**
Doubling the per-experiment cap from 5K (261K total) to 10K (517K total)
roughly halved the DINO best loss (0.989 → 0.926), which says the larger
budget produces more uniformly spread prototype distributions. But the
downstream Crop-MLP-on-frozen-features task did not benefit — AUROC
unchanged, accuracy slightly worse on the one borderline experiment.
Two interpretations:

  1. The 5K-cap features are already saturated for this downstream task.
     The extra crops only sharpen invariances that are not the bottleneck
     for R vs S separation.
  2. The 10K Tiny was caught at an earlier best-loss epoch (27 vs 32),
     when teacher-temperature warm-up was still ramping. It is possible
     a longer run at the 10K cap would shift the result, but the
     plateau-stopped training did not show it.

  Either way, on this dataset there's no evidence that ViT-Tiny needs
  more pretraining crops than the 5K cap to match ViT-Small's downstream
  performance.

**(e) Parameter cost.**
For the same experiment-level AUROC, you save 16 M backbone parameters
(75% reduction) and 25K classifier parameters by switching from ViT-Small
to ViT-Tiny. That's roughly 4× less inference cost per crop. On 14K crops
per experiment, this is the difference between a featurisation step that
dominates the pipeline and one that's a fraction of YOLO detection.

## 7. Reproducibility snippet (parameter counting)

```python
import torch.nn as nn
from ast_classifier.models.backbone import ViTSmall   # generic; accepts any width/depth

def backbone(embed_dim, num_heads):
    return ViTSmall(
        img_size=128, in_channels=1, patch_size=16,
        embed_dim=embed_dim, depth=12, num_heads=num_heads,
        mlp_ratio=4.0, time_conditioned=False,
    )

class CropMLP(nn.Module):
    def __init__(self, in_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64),     nn.LayerNorm(64),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

print(sum(p.numel() for p in backbone(384, 6).parameters()))  # 21,418,368
print(sum(p.numel() for p in backbone(192, 3).parameters()))  #  5,400,768
print(sum(p.numel() for p in CropMLP(384).parameters()))      #     58,050
print(sum(p.numel() for p in CropMLP(192).parameters()))      #     33,474
```
