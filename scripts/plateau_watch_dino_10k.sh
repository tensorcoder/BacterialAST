#!/usr/bin/env bash
set -u
LOG="/home/mkedz/code/ast_classifier/logs/dino_vit_tiny_10k/train_stdout.log"
PIDFILE="/tmp/dino_10k.pid"
SENTINEL="/home/mkedz/code/ast_classifier/logs/dino_vit_tiny_10k/PLATEAU_STOP.log"
PATIENCE=10
POLL_SEC=300

train_pid="$(cat "$PIDFILE" 2>/dev/null || true)"
[[ -z "$train_pid" ]] && { echo "no pid file"; exit 1; }

best_loss=""
best_epoch=-1
last_epoch=-1

while kill -0 "$train_pid" 2>/dev/null; do
  line="$(grep -aE 'Epoch [0-9]+/100 - Loss: [0-9.]+' "$LOG" 2>/dev/null | tail -1 || true)"
  if [[ -n "$line" ]]; then
    epoch="$(echo "$line" | sed -nE 's/.*Epoch ([0-9]+)\/100.*/\1/p')"
    loss="$(echo "$line" | sed -nE 's/.*Loss: ([0-9.]+).*/\1/p')"
    if [[ -n "$epoch" && -n "$loss" && "$epoch" != "$last_epoch" ]]; then
      last_epoch="$epoch"
      if [[ -z "$best_loss" ]] || awk -v a="$loss" -v b="$best_loss" 'BEGIN{exit !(a<b)}'; then
        best_loss="$loss"
        best_epoch="$epoch"
        echo "$(date -Iseconds) new best: epoch=$epoch loss=$loss" >> "${SENTINEL}.progress"
      else
        gap=$((epoch - best_epoch))
        echo "$(date -Iseconds) epoch=$epoch loss=$loss best=$best_loss@${best_epoch} gap=$gap" >> "${SENTINEL}.progress"
        if (( gap >= PATIENCE )); then
          echo "$(date -Iseconds) PLATEAU: $gap epochs since best ($best_loss @ $best_epoch). Killing pid $train_pid." >> "$SENTINEL"
          kill "$train_pid" 2>/dev/null
          sleep 5
          kill -0 "$train_pid" 2>/dev/null && kill -9 "$train_pid" 2>/dev/null
          exit 0
        fi
      fi
    fi
  fi
  sleep "$POLL_SEC"
done

echo "$(date -Iseconds) training process $train_pid exited on its own; watcher done." >> "$SENTINEL"
