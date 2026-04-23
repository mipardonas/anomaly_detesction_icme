# Environment Setup

We recommend using two isolated environments because the two codebases were developed with different PyTorch/CUDA versions.

## System Used in Our Final Training

- OS: Linux
- GPU: 2 x GPU_A800_8KA 96G for RT-DETRv4s detector training
- Detector training time: about 1 hour
- CUDA: use a CUDA version compatible with the selected PyTorch build

## Environment A: INP-Former

Recommended for training INP-Former and exporting anomaly heatmaps.

```shell
conda create -n ida_inpformer python=3.8 -y
conda activate ida_inpformer

pip install --upgrade pip
pip install -r environment/requirements_inp_former.txt
```

```text
torch==2.0.0+cu118
torchvision==0.15.1+cu118
opencv_python_headless==4.6.0.66
Pillow==9.0.1
scikit_image==0.19.3
scikit_learn==0.22.2.post1
scipy==1.4.1
timm==0.9.12
kornia==0.7.3
```

If the official machine has a newer CUDA driver, the CUDA 11.8 PyTorch wheel should still run on most recent NVIDIA drivers. If installation from the PyTorch wheel index is blocked, install the matching PyTorch build manually first and then install the remaining packages.

## Environment B: RT-DETRv4s

Recommended for detector training and final inference.

```shell
conda create -n ida_rtdetrv4 python=3.12 -y
conda activate ida_rtdetrv4

pip install --upgrade pip
pip install -r environment/requirements_rt_detrv4.txt
```

```text
torch==2.5.1+cu124
torchvision==0.20.1+cu124
torchaudio==2.5.1+cu124
numpy==2.4.4
opencv_python==4.13.0.92
faster-coco-eval==1.7.2
transformers==5.5.0
tensorboard==2.20.0
```

## Dataset Preparation

Two dataset organizations were used during the final pipeline. The image files are not included in this audit package. Instead, a filename-only manifest is provided at:

```text
docs/dataset_manifest.json
```

The manifest records directory paths, file names, suffix counts, and file sizes only. It does not contain image or label contents.

### Dataset A: INP-Former Stage

This dataset is used for INP-Former training.

Original paths:

```text
/data/processed/ICME/track1/
  OK_901/
    images/
      train/   # 720 .bmp normal training images
      val/     # 181 .bmp normal validation images
  NG_1154/
    images/
      val/     # 222 .bmp abnormal validation images, plus 222 .npy files present in this folder
    labels/
      val/     # 222 .txt labels for abnormal validation images
```

Summary:

```text
OK train images: 720
OK val images: 181
NG val images: 222
NG val labels: 222
```

### Dataset B: RT-DETRv4s Stage

This dataset is the composed 7-class detection dataset used by anomaly-guided RT-DETRv4s.

Original path:

```text
/data/processed/ICME/track1compose/
  images/
    train/   # 1459 .bmp images
    val/     # 387 .bmp images
    test/    # 205 .bmp images
  labels/
    train/   # 828 .txt labels
    val/     # 207 .txt labels
    test/    # 115 .txt labels
  coco_annotations_7classes/
    *.json   # 3 COCO-format annotation files
```

Summary:

```text
Train images: 1459
Val images: 387
Test images: 205
Train labels: 828
Val labels: 207
Test labels: 115
COCO annotation files: 3
```

Expected heatmap organization:

```text
HEATMAP_ROOT/
  train/
    image_name_anomaly_score.npy
  val/
    image_name_anomaly_score.npy
  test/
    image_name_anomaly_score.npy
```

The detector data loader matches images and heatmaps by basename plus the suffix `_anomaly_score.npy`.

## Weight Files

Weights are not included in this package. See `weights/README.md` for expected locations.
