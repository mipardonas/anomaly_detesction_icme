# Weight Files
because of the limite of the memory, we put the weight files in the google drive.
https://drive.google.com/drive/folders/1D3GWJQvJNyVqr74KWfzzT7JfPDIfruot?usp=sharing

Recommended layout:

weights/
  inp_former/
    icme_model.pth              # trained INP-Former checkpoint
  rtdetrv4/
    dinov3_vitb16_pretrain_lvd1689m.pth # if RT-DETRv4 encoder loader requires a local DINOv3 weight
    RTv4-S-hgnet.pth                # COCO-pretrained RT-DETRv4s initialization
    s_best_stg1.pth                   # final anomaly-guided RT-DETRv4s checkpoint

The exact checkpoint filenames can be changed, but update the paths in `scripts/*.sh` accordingly.

Final submitted detector checkpoint:
weights/rtdetrv4/s_best_stg1.pth

Detector pretraining checkpoint:
weights/rtdetrv4/RTv4-S-hgnet.pth

INP-Former checkpoint:
weights/inp_former/icme_model.pth
