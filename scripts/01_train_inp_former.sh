#!/usr/bin/env bash
set -euo pipefail

# Train INP-Former on normal training images.
# Replace DATA_ROOT with the official challenge dataset root.

# path to the dataset root
# which composed with OK_901/images/train, OK_901/images/val
# NG_1154/images/val, NG_1154/labels/val
DATA_ROOT="/path/to/official/track1"
# dir to save the trained model and logs
SAVE_DIR="./saved_results"

cd "$(dirname "$0")/../INP-Former"

CUDA_VISIBLE_DEVICES=0 python INP_Former_Multi_Class.py \
  --dataset ICME \
  --data_path "${DATA_ROOT}" \
  --phase train \
  --save_dir "${SAVE_DIR}" 
