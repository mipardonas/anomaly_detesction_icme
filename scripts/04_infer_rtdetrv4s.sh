#!/usr/bin/env bash
set -euo pipefail

# Final RT-DETRv4s inference with anomaly heatmaps, class-agnostic NMS,
# and anomaly proposal fusion. Replace paths before running.
# the path of class_id_map.json is the same as the one in RT-DETRv4
TEST_IMAGE_DIR="/path/to/official/track1/images/test"
TEST_HEATMAP_DIR="/path/to/output/heatmapscore_reorg/test"
OUTPUT_DIR="/path/to/output/rtv4_hgnetv2_s_7classes_anomaly_guided_test"
CLASS_ID_MAP="/path/to/class_id_map.json"

cd "$(dirname "$0")/../RT-DETRv4"

CUDA_VISIBLE_DEVICES=0 python tools/inference/batch_torch_inf.py \
  -c configs/rtv4/rtv4_hgnetv2_s_7classes_anomaly_guided.yml \
  -r ../weights/rtdetrv4/best_stg1.pth \
  --input-dir "${TEST_IMAGE_DIR}" \
  --anomaly-dir "${TEST_HEATMAP_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --anomaly-mode on \
  --require-anomaly \
  --conf-thres 0.03 \
  --pre-nms-topk 100 \
  --nms-iou-thres 0.30 \
  --max-det 20 \
  --class-agnostic-nms \
  --use-anomaly-proposals \
  --proposal-thresholds 0.70 0.82 0.90 \
  --proposal-min-area 80 \
  --proposal-topk 10 \
  --proposal-score-scale 0.60 \
  --proposal-append-iou 0.15 \
  --proposal-fusion-nms-iou 0.30 \
  --class-id-map "${CLASS_ID_MAP}"
