#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
cd "$REPO_ROOT"

OUT="/scratch/$USER/sft_runs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SIZES=(128 256 512 1024 full)

for SIZE in "${SIZES[@]}"; do
  if [[ "$SIZE" == "full" ]]; then
    TRAIN_SIZE=-1
  else
    TRAIN_SIZE="$SIZE"
  fi

  echo "========================================"
  echo "Starting run for SIZE=$SIZE"
  echo "Repo root: $REPO_ROOT"
  echo "Output dir: $OUT/$SIZE"
  echo "========================================"

  CUDA_VISIBLE_DEVICES=0,1 uv run python student/run_sft.py \
    --output-dir "$OUT/$SIZE" \
    --run-name "sft_$SIZE" \
    --train-size "$TRAIN_SIZE" \
    --num-epochs 200 \
    --max-optimizer-steps 300 \
    --microbatch-size 1 \
    --gradient-accumulation-steps 32 \
    --learning-rate 2e-5 \
    --weight-decay 0.01 \
    --max-grad-norm 1.0 \
    --eval-every 25 \
    --save-every 100 \
    --policy-device cuda:0 \
    --vllm-device cuda:1 \
    --gradient-checkpointing \
    --wandb-mode online \
    --eval-before-train

  echo "Finished SIZE=$SIZE"
done

echo "All runs completed."
