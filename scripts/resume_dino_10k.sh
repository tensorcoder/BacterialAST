#!/usr/bin/env bash
# Resume the ViT-Tiny @ 10K crops/experiment DINO run from last.pt.
set -euo pipefail
cd /home/mkedz/code/ast_classifier
PYTHONPATH=/home/mkedz/code nohup ./.venv/bin/python -m ast_classifier.train_vit_tiny_scratch \
  --resume /home/mkedz/code/ast_classifier/checkpoints/dino_vit_tiny_10k/dino/last.pt \
  --max-crops-per-experiment 10000 \
  --output-dir /home/mkedz/code/ast_classifier/checkpoints/dino_vit_tiny_10k \
  --log-dir   /home/mkedz/code/ast_classifier/logs/dino_vit_tiny_10k \
  >> /home/mkedz/code/ast_classifier/logs/dino_vit_tiny_10k/train_stdout.log 2>&1 &
TRAIN_PID=$!
disown $TRAIN_PID 2>/dev/null || true
echo $TRAIN_PID > /tmp/dino_10k.pid
echo "resumed training PID $TRAIN_PID, log: logs/dino_vit_tiny_10k/train_stdout.log"
echo "to also restart the plateau watcher: nohup ./scripts/plateau_watch_dino_10k.sh > logs/dino_vit_tiny_10k/watcher_stdout.log 2>&1 &"
