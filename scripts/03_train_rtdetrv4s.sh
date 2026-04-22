#!/usr/bin/env bash
set -euo pipefail

# Train anomaly-guided RT-DETRv4s.
# Before running, update the dataset and heatmap paths in:
#   RT-DETRv4/configs/rtv4/rtv4_hgnetv2_s_7classes_anomaly_guided.yml

cd "$(dirname "$0")/../RT-DETRv4"

torchrun --nproc_per_node=2 --master_port=29500 train.py \
  -c configs/rtv4/rtv4_hgnetv2_s_7classes_anomaly_guided.yml \
  -t ../weights/rtdetrv4/RTv4-S-hgnet.pth \
  --use-amp \
  --anomaly-mode on
