#!/bin/bash
# run_eval.sh
# Evaluate all three ablation conditions (base SLM, control SFT, runtime-aware SFT)
# on the held-out test split and log results to WandB.
#
# Prerequisites:
#   - Checkpoints trained via training/train.py must exist at CONTROL_CKPT and
#     RUNTIME_AWARE_CKPT (default: Google Drive paths used in Colab).
#   - Test split must exist at data/curated/test/dataset_clean.json.
#     Run scripts/create_dataset_splits.py first if it does not.
#
# Usage:
#   bash scripts/run_eval.sh
#
# Override checkpoint paths if needed:
#   CONTROL_CKPT=/my/path RUNTIME_AWARE_CKPT=/my/path bash scripts/run_eval.sh
#
# To skip WandB logging, remove --use_wandb from each command below.

set -e
cd "$(dirname "$0")/.."

BASE_MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
CONTROL_CKPT="${CONTROL_CKPT:-/content/drive/MyDrive/efficient-codegen/checkpoints/control_full}"
RUNTIME_AWARE_CKPT="${RUNTIME_AWARE_CKPT:-/content/drive/MyDrive/efficient-codegen/checkpoints/runtime_aware_full}"
DATA_PATH="data/curated/test/dataset_clean.json"
NUM_CANDIDATES=5
TEMPERATURE=0.2

if [ ! -f "$DATA_PATH" ]; then
  echo "ERROR: Test split not found at $DATA_PATH"
  echo "Run: python3 scripts/create_dataset_splits.py"
  exit 1
fi

echo "=== Evaluating: base_slm ==="
python3 training/evaluate_model.py \
  --model_path "$BASE_MODEL" \
  --run_name base_slm \
  --data_path "$DATA_PATH" \
  --num_candidates $NUM_CANDIDATES \
  --temperature $TEMPERATURE \
  --use_wandb

echo ""
echo "=== Evaluating: control_sft ==="
python3 training/evaluate_model.py \
  --model_path "$CONTROL_CKPT" \
  --base_model_name "$BASE_MODEL" \
  --run_name control_sft \
  --data_path "$DATA_PATH" \
  --num_candidates $NUM_CANDIDATES \
  --temperature $TEMPERATURE \
  --use_wandb

echo ""
echo "=== Evaluating: runtime_aware_sft ==="
python3 training/evaluate_model.py \
  --model_path "$RUNTIME_AWARE_CKPT" \
  --base_model_name "$BASE_MODEL" \
  --run_name runtime_aware_sft \
  --data_path "$DATA_PATH" \
  --num_candidates $NUM_CANDIDATES \
  --temperature $TEMPERATURE \
  --use_wandb

echo ""
echo "=== All evaluations complete. Results logged to WandB. ==="
