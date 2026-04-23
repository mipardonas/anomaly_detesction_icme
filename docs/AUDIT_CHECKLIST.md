# Official Audit Checklist

## 1. Training Code

- INP-Former:
  - `INP-Former/INP_Former_Multi_Class.py`
  - `INP-Former/utils.py`
  - `INP-Former/dataset.py`
  - `INP-Former/track1_dataloader.py`
- RT-DETRv4s:
  - `RT-DETRv4/train.py`
  - `RT-DETRv4/configs/rtv4/rtv4_hgnetv2_s_7classes_anomaly_guided.yml`

## 2. Testing / Inference Code

- INP-Former heatmap export:
  - `INP-Former/INP_Former_Multi_Class.py`
  - `INP-Former/utils.py`, especially the heatmap saving logic in `evaluation_batch_with_npy`
- RT-DETRv4s final inference:
  - `RT-DETRv4/tools/inference/batch_torch_inf.py`
- Optional final JSON generation:
  - `RT-DETRv4/generate_submission_json.py`

## 3. Model Definition / Model Code

- INP-Former model code:
  - `INP-Former/models/`
  - `INP-Former/dinov2/`
  - `INP-Former/dinov1/`
  - `INP-Former/beit/`
  - `INP-Former/optimizers/`
- RT-DETRv4s model code:
  - `RT-DETRv4/engine/`
  - `RT-DETRv4/configs/`

## 4. Model Weights

- trained INP-Former checkpoint;
- COCO-pretrained RT-DETRv4s checkpoint;
- final trained anomaly-guided RT-DETRv4s checkpoint;

## 5. Environment Setup

- `environment/ENVIRONMENT.md`
- `environment/requirements_inp_former.txt`
- `environment/requirements_rt_detrv4.txt`

## 6. Run Commands

- `scripts/01_train_inp_former.sh`
- `scripts/02_export_inp_heatmaps.sh`
- `scripts/03_train_rtdetrv4s.sh`
- `scripts/04_infer_rtdetrv4s.sh`

Before running, update dataset paths, output paths, checkpoint paths, and class-id map path in the scripts and detector config.
