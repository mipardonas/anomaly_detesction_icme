import torch
from numpy.random import normal
import  random
import logging
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,  precision_recall_curve, average_precision_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
import os
from functools import partial
import math
from tqdm import tqdm
from aug_funcs import rot_img, translation_img, hflip_img, grey_img, rot90_img
import torch.backends.cudnn as cudnn
from adeval import  EvalAccumulatorCuda
from torchvision.ops import nms
from PIL import Image

def ader_evaluator(pr_px, pr_sp, gt_px, gt_sp, use_metrics = ['I-AUROC', 'I-AP', 'I-F1_max','P-AUROC', 'P-AP', 'P-F1_max', 'AUPRO']):
    if len(gt_px.shape) == 4:
        gt_px = gt_px.squeeze(1)
    if len(pr_px.shape) == 4:
        pr_px = pr_px.squeeze(1)
    
    # 检查图像级标签是否只有一类
    unique_sp = np.unique(gt_sp)
    if len(unique_sp) == 1:
        print(f"Warning: Only one class present in gt_sp: {unique_sp[0]}. Setting I-AUROC and I-AP to 0.5.")
        auroc_sp = 0.5
        ap_sp = 0.5
        best_f1_score_sp = 0.0
    else:
        auroc_sp = roc_auc_score(gt_sp, pr_sp)
        ap_sp = average_precision_score(gt_sp, pr_sp)
        best_f1_score_sp = f1_score_max(gt_sp, pr_sp)

    # 检查像素级标签是否只有一类
    unique_px = np.unique(gt_px)
    if len(unique_px) == 1:
        print(f"Warning: Only one class present in gt_px: {unique_px[0]}. Setting P-AUROC, P-AP, AUPRO to 0.5.")
        auroc_px = 0.5
        ap_px = 0.5
        best_f1_score_px = 0.0
        aupro_px = 0.0
    else:
        score_min = min(pr_sp)
        score_max = max(pr_sp)
        anomap_min = pr_px.min()
        anomap_max = pr_px.max()
        accum = EvalAccumulatorCuda(score_min, score_max, anomap_min, anomap_max, skip_pixel_aupro=False, nstrips=200)
        accum.add_anomap_batch(torch.tensor(pr_px).cuda(non_blocking=True),
                               torch.tensor(gt_px.astype(np.uint8)).cuda(non_blocking=True))
        metrics = accum.summary()
        auroc_px = metrics['p_auroc']
        ap_px = metrics['p_aupr']
        best_f1_score_px = f1_score_max(gt_px.ravel(), pr_px.ravel())
        aupro_px = metrics['p_aupro']

    metric_results = {}
    for metric in use_metrics:
        if metric.startswith('I-AUROC'):
            metric_results[metric] = auroc_sp
        elif metric.startswith('I-AP'):
            metric_results[metric] = ap_sp
        elif metric.startswith('I-F1_max'):
            metric_results[metric] = best_f1_score_sp
        elif metric.startswith('P-AUROC'):
            metric_results[metric] = auroc_px
        elif metric.startswith('P-AP'):
            metric_results[metric] = ap_px
        elif metric.startswith('P-F1_max'):
            metric_results[metric] = best_f1_score_px
        elif metric.startswith('AUPRO'):
            metric_results[metric] = aupro_px
    return list(metric_results.values())

def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)
    return logger

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def modify_grad(x, inds, factor=0.):
    inds = inds.expand_as(x)
    x[inds] *= factor
    return x

def modify_grad_v2(x, factor):
    factor = factor.expand_as(x)
    x *= factor
    return x

def global_cosine_hm_adaptive(a, b, y=3):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1).detach()
        mean_dist = point_dist.mean()
        factor = (point_dist/mean_dist)**(y)
        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))
        partial_func = partial(modify_grad_v2, factor=factor)
        b_.register_hook(partial_func)

    loss = loss / len(a)
    return loss

def cal_anomaly_maps(fs_list, ft_list, out_size=224):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bicubic', align_corners=True)
        a_map_list.append(a_map)
    anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
    return anomaly_map, a_map_list

def f1_score_max(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return f1s.max()

def evaluation_batch(model, dataloader, device, _class_=None, max_ratio=0, resize_mask=None):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    with torch.no_grad():
        for img, gt, label, img_path in tqdm(dataloader, ncols=80):
            img = img.to(device)
            output = model(img)
            en, de = output[0], output[1]
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            if resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')
            anomaly_map = gaussian_kernel(anomaly_map)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            # gt = gt.bool()
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]
            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)
            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)
            pr_list_sp.append(sp_score)
        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()
        
        # GPU acceleration
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = ader_evaluator(pr_list_px, pr_list_sp, gt_list_px, gt_list_sp)

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                      groups=channels,
                                      bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

class WarmCosineScheduler(_LRScheduler):

    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, ):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup_schedule, schedule))

        super(WarmCosineScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for base_lr in self.base_lrs]
        else:
            return [self.schedule[self.last_epoch] for base_lr in self.base_lrs]

def map_coords_to_orig(cnt, args, orig_w, orig_h):
    """
    将 crop 图上的坐标 cnt 映射回原图的归一化坐标 [0, 1]
    假设: 原图 -> Resize(input_size) -> CenterCrop(crop_size)
    """
    # 1. 计算在 Resize 空间下的偏移量
    left = (args.input_size - args.crop_size) // 2
    top = (args.input_size - args.crop_size) // 2
    
    # 2. 映射到 Resize 空间 (像素)
    cnt_res = cnt.squeeze().astype(np.float32) + np.array([left, top])
    
    # 3. 映射到原图像素空间
    # 注意：这里乘的是 (原图/Resize图) 的比例
    cnt_orig_px = cnt_res * np.array([orig_w / args.input_size, orig_h / args.input_size])
    
    # 4. 归一化到 [0, 1]
    cnt_norm = cnt_orig_px / np.array([orig_w, orig_h])
    return cnt_norm
        
def evaluation_batch_with_npy(model, dataloader, device, max_ratio=0, resize_mask=None, npy_dir=None, args=None):
    """
    与 evaluation_batch 相同，但额外保存异常分数矩阵（.npy 文件）。
    npy_dir: 保存异常分数文件的目录
    """
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    if npy_dir is not None:
        os.makedirs(npy_dir, exist_ok=True)

    with torch.no_grad():
        for img, gt, label, img_path in tqdm(dataloader, ncols=80):
            img = img.to(device)
            output = model(img)
            en, de = output[0], output[1]
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])

            # 保存异常分数矩阵（原始数值，未归一化）
            if npy_dir is not None:
                for i in range(img.shape[0]):
                    # 获取原始异常分数 (H, W)
                    anom_np = anomaly_map[i, 0].cpu().detach().numpy()
                    # 保存为 .npy 文件
                    basename = os.path.splitext(os.path.basename(img_path[i]))[0]
                    save_path = os.path.join(npy_dir, f"{basename}_anomaly_score.npy")
                    np.save(save_path, anom_np)  # 保存浮点数组

            # 指标计算（与之前相同）
            if resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')
            anomaly_map = gaussian_kernel(anomaly_map)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]
            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)

            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map_flat = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map_flat, dim=1, descending=True)[0][:, :int(anomaly_map_flat.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)
            pr_list_sp.append(sp_score)

        # 汇总指标
        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = ader_evaluator(pr_list_px, pr_list_sp, gt_list_px, gt_list_sp)

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]