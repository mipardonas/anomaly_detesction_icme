# Reproduction Notes

## Pipeline Summary

The final submitted method uses two models:

1. INP-Former:
   - trained on normal training images;
   - exports dense anomaly heatmaps as `.npy` files;
   - heatmaps are named by image basename plus `_anomaly_score.npy`.

2. RT-DETRv4s:
   - initialized from COCO-pretrained RT-DETRv4s weights;
   - trained on 7-class COCO-format challenge annotations;
   - loads RGB images and matching INP-Former heatmaps;
   - fuses heatmaps into the HybridEncoder at feature level;
   - uses anomaly proposal fusion and class-agnostic NMS during final inference.

## Key Detector Config

The final detector config is:

RT-DETRv4/configs/rtv4/rtv4_hgnetv2_s_7classes_anomaly_guided.yml

Important fields:

num_classes: 7
remap_mscoco_category: False
HybridEncoder.anomaly_guided: True
HybridEncoder.anomaly_fusion_weight: 0.5
HybridEncoder.anomaly_residual_weight: 0.25
HybridEncoder.anomaly_map_minmax_norm: True

Before official reproduction, update the dataset and heatmap paths in the config.

## Final Inference Parameters

conf_thres: 0.03
pre_nms_topk: 100
nms_iou_thres: 0.30
max_det: 20
class_agnostic_nms: True
use_anomaly_proposals: True
proposal_thresholds: 0.70, 0.82, 0.90
proposal_min_area: 80
proposal_topk: 10
proposal_score_scale: 0.60
proposal_append_iou: 0.15
proposal_fusion_nms_iou: 0.30

## Official Metric

The Track 1 metric implementation is included at:

RT-DETRv4/caculate_metric2.py

## Files Not Included

The following are intentionally excluded and should be supplied by the reproducer:

- official challenge dataset;
- generated heatmaps, unless the reviewer wants to skip the INP-Former export stage.

## Dataset Manifest

The audit package includes a filename-only dataset manifest:

```text
docs/dataset_manifest.json
```

This JSON lists the two dataset organizations used in the pipeline:

- INP-Former stage dataset under `/data/processed/ICME/track1/OK_901` and `/data/processed/ICME/track1/NG_1154`;
- RT-DETRv4s stage dataset under `/data/processed/ICME/track1compose`.

The manifest includes relative file names, suffix counts, file sizes, and summary statistics. It does not include image or label contents.
