#!/bin/bash
# Full training pipeline — waits for preprocessing then runs stages 2-5
set -e

REPO_DIR="/home/mkedz/code/ast_classifier"
VENV="$REPO_DIR/.venv/bin/python3"
export PYTHONPATH="/home/mkedz/code"
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR"

DATA_ROOT="/mnt/f/Data_second_protocol"
PREPROCESSED_DIR="$REPO_DIR/preprocessed"
FEATURES_DIR="$REPO_DIR/features"
CHECKPOINTS_DIR="$REPO_DIR/checkpoints"

# ---------------------------------------------------------------
# Wait for preprocessing to finish (monitor the nohup process)
# ---------------------------------------------------------------
echo "$(date) | Waiting for preprocessing to complete..."
PREPROCESS_PID=$(pgrep -f "ast_classifier.scripts.preprocess" || true)
if [ -n "$PREPROCESS_PID" ]; then
    echo "$(date) | Preprocessing PID: $PREPROCESS_PID"
    while kill -0 "$PREPROCESS_PID" 2>/dev/null; do
        N_DONE=$(ls "$PREPROCESSED_DIR"/*.h5 2>/dev/null | wc -l)
        echo "$(date) | Preprocessing in progress... $N_DONE/42 experiments done"
        sleep 300  # check every 5 minutes
    done
fi

N_DONE=$(ls "$PREPROCESSED_DIR"/*.h5 2>/dev/null | wc -l)
echo "$(date) | Preprocessing complete. $N_DONE HDF5 files."

# ---------------------------------------------------------------
# Stage 2: DINO pretraining
# ---------------------------------------------------------------
echo ""
echo "$(date) | =============================================="
echo "$(date) | STAGE 2: DINO Self-Supervised Pretraining"
echo "$(date) | =============================================="
$VENV -m ast_classifier.scripts.train \
    --stage dino \
    --data-root "$DATA_ROOT" \
    --preprocessed-dir "$PREPROCESSED_DIR" \
    --features-dir "$FEATURES_DIR" \
    --checkpoints-dir "$CHECKPOINTS_DIR" \
    --device cuda:0 \
    2>&1 | tee "$LOG_DIR/train_dino.log"

echo "$(date) | DINO pretraining complete."

# ---------------------------------------------------------------
# Stage 3: Feature extraction
# ---------------------------------------------------------------
echo ""
echo "$(date) | =============================================="
echo "$(date) | STAGE 3: Feature Extraction"
echo "$(date) | =============================================="
$VENV -m ast_classifier.scripts.train \
    --stage extract \
    --data-root "$DATA_ROOT" \
    --preprocessed-dir "$PREPROCESSED_DIR" \
    --features-dir "$FEATURES_DIR" \
    --checkpoints-dir "$CHECKPOINTS_DIR" \
    --device cuda:0 \
    2>&1 | tee "$LOG_DIR/extract_features.log"

echo "$(date) | Feature extraction complete."

# ---------------------------------------------------------------
# Stage 4: Classifier training
# ---------------------------------------------------------------
echo ""
echo "$(date) | =============================================="
echo "$(date) | STAGE 4: Population Temporal Classifier"
echo "$(date) | =============================================="
$VENV -m ast_classifier.scripts.train \
    --stage classifier \
    --data-root "$DATA_ROOT" \
    --preprocessed-dir "$PREPROCESSED_DIR" \
    --features-dir "$FEATURES_DIR" \
    --checkpoints-dir "$CHECKPOINTS_DIR" \
    --device cuda:0 \
    2>&1 | tee "$LOG_DIR/train_classifier.log"

echo "$(date) | Classifier training complete."

# ---------------------------------------------------------------
# Stage 5: Early-exit calibration
# ---------------------------------------------------------------
echo ""
echo "$(date) | =============================================="
echo "$(date) | STAGE 5: Early-Exit Calibration"
echo "$(date) | =============================================="
$VENV -m ast_classifier.scripts.train \
    --stage calibrate \
    --data-root "$DATA_ROOT" \
    --preprocessed-dir "$PREPROCESSED_DIR" \
    --features-dir "$FEATURES_DIR" \
    --checkpoints-dir "$CHECKPOINTS_DIR" \
    --device cuda:0 \
    2>&1 | tee "$LOG_DIR/calibrate_exit.log"

echo "$(date) | Calibration complete."

# ---------------------------------------------------------------
# Stage 6: Evaluation + plots
# ---------------------------------------------------------------
echo ""
echo "$(date) | =============================================="
echo "$(date) | EVALUATION + PLOTS"
echo "$(date) | =============================================="
RESULTS_DIR="$REPO_DIR/results"
mkdir -p "$RESULTS_DIR"
$VENV -m ast_classifier.scripts.evaluate \
    --data-root "$DATA_ROOT" \
    --features-dir "$FEATURES_DIR" \
    --checkpoints-dir "$CHECKPOINTS_DIR" \
    --output-dir "$RESULTS_DIR" \
    --device cuda:0 \
    2>&1 | tee "$LOG_DIR/evaluate.log"

echo ""
echo "$(date) | =============================================="
echo "$(date) | ALL STAGES COMPLETE"
echo "$(date) | =============================================="
echo "$(date) | Results: $RESULTS_DIR"
echo "$(date) | Checkpoints: $CHECKPOINTS_DIR"
echo "$(date) | Logs: $LOG_DIR"
