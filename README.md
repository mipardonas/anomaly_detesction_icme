# IDA 2026 Official Audit Package

This package collects the code and instructions needed to reproduce the final submitted pipeline:

1. train INP-Former on normal training images;
2. export anomaly heatmaps for train/val/test images as `.npy` files;
3. train anomaly-guided RT-DETRv4s on RGB images plus heatmaps;
4. run RT-DETRv4s inference with anomaly-map proposals and NMS;
5. generate the final challenge-format predictions.

## Directory Layout

IDA2026_official_audit_package/
  INP-Former/              # INP-Former training, heatmap generation, and model code
  RT-DETRv4/               # RT-DETRv4s training, inference, config, model code, and metric code
  environment/             # dependency files and setup notes
  scripts/                 # reproducible command templates
  weights/                 # placeholder directory for weights copied by the user
  docs/                    # optional extra notes

## What Is Included

- Training code:
  - `INP-Former/INP_Former_Multi_Class.py`
  - `RT-DETRv4/train.py`
- Testing / inference code:
  - `INP-Former/INP_Former_Multi_Class.py` with `--phase test`
  - `RT-DETRv4/tools/inference/batch_torch_inf.py`
  - `RT-DETRv4/generate_submission_json.py`
- Model definition / model code:
  - `INP-Former/models/`
  - `INP-Former/dinov2/`, `dinov1/`, `beit/`
  - `RT-DETRv4/engine/`
  - `RT-DETRv4/configs/`
- Evaluation:
  - `RT-DETRv4/caculate_metric2.py`
- Environment setup:
  - `environment/ENVIRONMENT.md`
  - `environment/requirements_inp_former.txt`
  - `environment/requirements_rt_detrv4_full.txt`
  - `environment/requirements_rt_detrv4_noflash.txt`
- Reproduction scripts:
  - `scripts/01_train_inp_former.sh`
  - `scripts/02_export_inp_heatmaps.sh`
  - `scripts/03_train_rtdetrv4s.sh`
  - `scripts/04_infer_rtdetrv4s.sh`

## Recommended Reproduction Order

1. Create the INP-Former environment using `environment/ENVIRONMENT.md`.
2. Copy the INP-Former checkpoint into `weights/inp_former/`.
3. Run or inspect `scripts/01_train_inp_former.sh` for the normal-image training stage.
4. Run `scripts/02_export_inp_heatmaps.sh` to export `.npy` heatmaps for train, val, and test splits.
5. Create the RT-DETRv4 environment using `environment/ENVIRONMENT.md`.
6. Copy COCO-pretrained RT-DETRv4s weights and the final trained RT-DETRv4s checkpoint into `weights/rtdetrv4/`.
7. Run `scripts/03_train_rtdetrv4s.sh` to reproduce detector training.
8. Run `scripts/04_infer_rtdetrv4s.sh` to reproduce final inference.

All dataset paths in scripts are placeholders. Replace them with the official dataset location used by the reviewer.
