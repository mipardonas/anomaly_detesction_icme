#!/usr/bin/env bash
set -euo pipefail

# Reproduction wrapper for the historical `test_nms_tuned` experiment.
# This script does not modify the current inference code. It only freezes the
# command-line settings that were used for that run and adapts the paths for
# the local test split by default.
#
# Historical command found in shell history:
# CUDA_VISIBLE_DEVICES=0 python tools/inference/batch_torch_inf.py \
#   -c configs/rtv4/rtv4_hgnetv2_s_coco_anomaly_guided.yml \
#   -r outputs/rtv4_hgnetv2_s_coco_anomaly_guided/best_stg2.pth \
#   --input-dir /data/processed/ICME/Test \
#   --anomaly-dir /data/usrs/lnj/INP-Former/output/res_nocls/heatmapscore/Test \
#   --output-dir /data/usrs/lnj/icme/Track1/output/rtdetrv4_anomaly/test_nms_tuned \
#   --anomaly-mode on \
#   --require-anomaly \
#   --conf-thres 0.01 \
#   --pre-nms-topk 200 \
#   --nms-iou-thres 0.4 \
#   --max-det 50 \
#   --class-agnostic-nms \
#   --save-vis

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"

CONFIG="${CONFIG:-configs/rtv4/rtv4_hgnetv2_s_coco_anomaly_guided.yml}"
CHECKPOINT="${CHECKPOINT:-outputs/rtv4_hgnetv2_s_coco_anomaly_guided/best_stg2.pth}"

INPUT_DIR="${INPUT_DIR:-/data/processed/ICME/track1hybrid/test/images}"
ANOMALY_DIR="${ANOMALY_DIR:-/data/usrs/lnj/INP-Former/output/res_nocls/heatmapscore/test}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/usrs/lnj/icme/Track1/output/rtdetrv4_anomaly/test_nms_tuned_local}"

if [[ "${DEVICE}" == "cpu" ]]; then
  exec python tools/inference/batch_torch_inf.py \
    -c "${CONFIG}" \
    -r "${CHECKPOINT}" \
    --input-dir "${INPUT_DIR}" \
    --anomaly-dir "${ANOMALY_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --anomaly-mode on \
    --require-anomaly \
    --conf-thres 0.01 \
    --pre-nms-topk 200 \
    --nms-iou-thres 0.4 \
    --max-det 50 \
    --class-agnostic-nms \
    --save-vis \
    -d cpu
fi

exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" python tools/inference/batch_torch_inf.py \
  -c "${CONFIG}" \
  -r "${CHECKPOINT}" \
  --input-dir "${INPUT_DIR}" \
  --anomaly-dir "${ANOMALY_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --anomaly-mode on \
  --require-anomaly \
  --conf-thres 0.01 \
  --pre-nms-topk 200 \
  --nms-iou-thres 0.4 \
  --max-det 50 \
  --class-agnostic-nms \
  --save-vis \
  -d cuda
