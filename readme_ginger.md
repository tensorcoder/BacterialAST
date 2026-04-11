# Reproducing results_crop_mlp Graphs for a New Experiment

Instructions for generating P(Resistant) trajectory plots for new experiment data using the existing trained models. No retraining is needed.

## Prerequisites

You need these files (not in the repo):

1. **YOLO weights** — `vertical_obb_100epo_best.pt`
2. **DINO v1 checkpoint** — `checkpoints/dino/best_backbone.pt` (trained on 128x128 reflected crops)
3. **MLP checkpoints** — `results_crop_mlp/checkpoints/fold0_best.pt` through `fold4_best.pt`

## Critical: Code Changes Before Running

The code currently produces 96x96 zero-padded crops. The DINO v1 model was trained on **128x128 reflection-padded** crops. Two changes are required:

### 1. Fix padding in `data/preprocessing.py`

Lines 221-224: change the padding from:
```python
borderType=cv2.BORDER_CONSTANT,
value=0,
```
to:
```python
borderType=cv2.BORDER_REFLECT_101,
```

There is no CLI flag for padding mode — this must be changed in the source.

### 2. Fix DINO input size in `config.py`

Change `DINOConfig.img_size` from `96` to `128`. This controls how the ViT-Small backbone is instantiated during feature extraction. A mismatch will cause a positional embedding shape error when loading the checkpoint.

## Step 1: Organise the New Experiment Folder

The preprocessing script expects this folder structure:

```
/path/to/new_data/
  Resistant/
    experiment_name/
      images/
        image_1741018345.67383.bmp
        ...
  Susceptible/
    experiment_name/
      images/
        ...
```

If you only have a single unlabelled experiment, place it inside either `Resistant/` or `Susceptible/` — the label only matters for colouring the trajectory plots.

## Step 2: Preprocess (YOLO detection → 128x128 HDF5 crops)

```bash
python -m ast_classifier.scripts.preprocess \
    --data-root /path/to/new_data \
    --output-dir ./preprocessed_new \
    --yolo-weights /path/to/vertical_obb_100epo_best.pt \
    --crop-size 128 \
    --device cuda:0
```

The `--crop-size 128` flag overrides the default. Combined with the padding code change above, this produces 128x128 reflection-padded crops matching what DINO v1 expects.

## Step 3: Extract DINO Features (HDF5 crops → 384-dim NPZ)

```bash
python -m ast_classifier.scripts.train \
    --stage extract \
    --preprocessed-dir ./preprocessed_new \
    --features-dir ./features_new \
    --backbone-path /path/to/checkpoints/dino/best_backbone.pt \
    --device cuda:0
```

This loads the DINO v1 ViT-Small backbone, applies CLAHE contrast enhancement and normalisation (mean=0.3387, std=0.1173), then extracts the 384-dim CLS token for every crop. Outputs one `.npz` per experiment containing `features` (N, 384) and `timestamps` (N,).

## Step 4: Run Inference with Existing MLP Checkpoints

Write an inference script modelled on `scripts/eval_no_amp_control.py`. That script contains the exact inference pattern needed:

- `run_crop_mlp_inference()` (line 238) loads a fold checkpoint, runs all crops through the MLP, and computes cumulative P(Resistant) at time windows
- `plot_variant_trajectories()` (line 291) plots the P(R) trajectory per experiment

The key adaptation needed: `eval_no_amp_control.py` hardcodes experiment discovery to look in `data_root / "Susceptible_no_amp"`. Your script must discover experiments from your new data folder instead. Use `ExperimentMeta` from `data/dataset.py` to represent each experiment, pointing `features_path` at the corresponding NPZ in `./features_new/`.

For each experiment, load all 5 fold checkpoints and average P(R) predictions across folds (as `eval_no_amp_control.py` does) to get robust trajectory estimates.

## Expected Output

In a working classifier:
- **Susceptible** experiments: P(R) starts ~0.65 (bacteria look normal early on) and drops to ~0.17 by 60 minutes
- **Resistant** experiments: P(R) stays flat at ~0.70
- The curves cross 0.5 at ~25-30 minutes

## Notes

- The MLP was trained only on crops from t > 40 minutes where R/S morphological differences are strongest.
- Predictions are binned into 5-minute windows.
- The DINO backbone is time-conditioned — it receives relative timestamps alongside crops.
