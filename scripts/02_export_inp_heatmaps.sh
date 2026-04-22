#!/usr/bin/env bash
set -euo pipefail

# Export INP-Former anomaly heatmaps as .npy files.
# Run this script separately for train, val, and test by changing DATA_ROOT
# or by adapting INP_Former_Multi_Class.py to select the desired split.
# pay atttion here need to generate all the .npy files for all the images in the train, val, and test sets.
# because of finding overlapping between train and val sets which we generate
# so here we change the train, val, and test sets, different from the dataset in INP-Former
# but we do not use the image in the official test dataset in two stage

DATA_ROOT="/path/to/official/track1"
INP_CKPT="../weights/inp_former/icme_model.pth"
OUTPUT_DIR="/path/to/output/heatmapscore_reorg/test"

cd "$(dirname "$0")/../INP-Former"

CUDA_VISIBLE_DEVICES=0 python INP_Former_Multi_Class.py \
  --dataset ICME \
  --data_path "${DATA_ROOT}" \
  --phase test \
  --load_model "${INP_CKPT}" \
  --npy_dir "${OUTPUT_DIR}" 

# The RT-DETRv4 loader expects files named:
#   image_stem_anomaly_score.npy
