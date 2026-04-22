import torch
import torch.nn as nn
import numpy as np
import os
from functools import partial
import warnings
from tqdm import tqdm
from torch.nn.init import trunc_normal_
import argparse
from optimizers import StableAdamW
from utils import evaluation_batch,WarmCosineScheduler, global_cosine_hm_adaptive, setup_seed, get_logger, evaluation_batch_with_npy
import random
import torch.nn.functional as F
import PIL.Image as Image
from pathlib import Path

# Dataset-Related Modules
from track1_dataloader import Track1Dataset
from dataset import create_icme_datasets, ImageOnlyTestDataset
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset

# Model-Related Modules
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block

warnings.filterwarnings("ignore")

def main(args):
    # Fixing the Random Seed
    setup_seed(1)
    # Data Preparation
    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)
    if args.dataset == 'ICME':
        class_file_path = os.path.join(args.data_path, 'class-name.txt')
        if args.phase == 'test' and not os.path.exists(class_file_path):
            print_fn(f'class-name.txt not found, using image-only test dataset: {args.data_path}')
            train_data = None
            train_dataloader = None
            # val_data = None
            train_ng_data = None
            test_data = ImageOnlyTestDataset(
                root=args.data_path,
                transform=data_transform,
                crop_size=args.crop_size,
            )
            test_data_list = [test_data]
        else:
            datasets = create_icme_datasets(
                root=args.data_path,
                transform=data_transform,
                gt_transform=gt_transform,
                resize_size=args.input_size,
                crop_size=args.crop_size,
                splits=['train', 'val', 'test']   # 需要哪些split
            )
            # 无监督训练数据集
            train_data = datasets['train']
            train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                        num_workers=4, drop_last=True)
            # 验证集和测试集
            val_data = datasets.get('val', None)
            val_data_list = [val_data] if val_data else []

    # Adopting a grouping-based reconstruction strategy similar to Dinomaly
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    # Encoder info
    encoder = vit_encoder.load(args.encoder)
    if 'small' in args.encoder:
        embed_dim, num_heads = 384, 6
    elif 'base' in args.encoder:
        embed_dim, num_heads = 768, 12
    elif 'large' in args.encoder:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise "Architecture not in small, base, large."

    # Model Preparation
    Bottleneck = []
    INP_Guided_Decoder = []
    INP_Extractor = []

    # bottleneck
    Bottleneck.append(Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.))
    Bottleneck = nn.ModuleList(Bottleneck)

    # INP
    INP = nn.ParameterList(
                    [nn.Parameter(torch.randn(args.INP_num, embed_dim))
                     for _ in range(1)])

    # INP Extractor
    for i in range(1):
        blk = Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        INP_Extractor.append(blk)
    INP_Extractor = nn.ModuleList(INP_Extractor)

    # INP_Guided_Decoder
    for i in range(8):
        blk = Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        INP_Guided_Decoder.append(blk)
    INP_Guided_Decoder = nn.ModuleList(INP_Guided_Decoder)

    model = INP_Former(encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor, decoder=INP_Guided_Decoder,
                             target_layers=target_layers,  remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
                             fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP, embed_dim=embed_dim)
    model = model.to(device)

    if args.phase == 'train':
        # Model Initialization
        trainable = nn.ModuleList([Bottleneck, INP_Guided_Decoder, INP_Extractor, INP])
        for m in trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        # define optimizer
        optimizer = StableAdamW([{'params': trainable.parameters()}],
                                lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
        lr_scheduler = WarmCosineScheduler(optimizer, base_value=1e-3, final_value=1e-4, total_iters=args.total_epochs*len(train_dataloader),
                                           warmup_iters=100)
        print_fn('train image number:{}'.format(len(train_data)))

        # Train
        for epoch in range(args.total_epochs):
            model.train()
            loss_list = []
            for img, _ in tqdm(train_dataloader, ncols=80):
                img = img.to(device)
                en, de, g_loss = model(img)
                loss = global_cosine_hm_adaptive(en, de, y=3)
                loss = loss + 0.2 * g_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                loss_list.append(loss.item())
                lr_scheduler.step()
            print_fn('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, args.total_epochs, np.mean(loss_list)))
            if (epoch + 1) % args.total_epochs == 0:
                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

                for item, val_data in zip(args.item_list, val_data_list):
                    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                                                                  num_workers=4)
                    results = evaluation_batch(model, val_dataloader, device, max_ratio=0.01, resize_mask=256)
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
                    auroc_sp_list.append(auroc_sp)
                    ap_sp_list.append(ap_sp)
                    f1_sp_list.append(f1_sp)
                    auroc_px_list.append(auroc_px)
                    ap_px_list.append(ap_px)
                    f1_px_list.append(f1_px)
                    aupro_px_list.append(aupro_px)
                    print_fn(
                        '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                            item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

                print_fn('Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                        np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                        np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))
                torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, 'model.pth'))
                model.train()
    elif args.phase == 'test':
        # 确定模型路径（优先使用 --load_model）
        model_path = args.load_model if args.load_model else os.path.join(args.save_dir, args.save_name, 'model.pth')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint, strict=False)
        model = model.to(device)
        auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
        auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []
        model.eval()
        for item, test_data in zip(args.item_list, test_data_list):
            test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                                          num_workers=0)
            
            # 如果指定了 npy_dir，使用带可视化的评估函数
            if args.npy_dir is not None:
                npy_item_dir = args.npy_dir
                results = evaluation_batch_with_npy(
                    model, test_dataloader, device,
                    max_ratio=0.01, resize_mask=256,
                    npy_dir=npy_item_dir,
                    args=args,
                )
            else:
                results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256, args=args, alpha=args.alpha)
                
            auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
            auroc_sp_list.append(auroc_sp)
            ap_sp_list.append(ap_sp)
            f1_sp_list.append(f1_sp)
            auroc_px_list.append(auroc_px)
            ap_px_list.append(ap_px)
            f1_px_list.append(f1_px)
            aupro_px_list.append(aupro_px)
            print_fn(
                '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                    item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

        print_fn(
            'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))
    else:
        raise ValueError(f'Unsupported phase: {args.phase}')


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
    parser = argparse.ArgumentParser(description='')

    # dataset info
    parser.add_argument('--dataset', type=str, default=r'MVTec-AD') # 'MVTec-AD' or 'VisA' or 'Real-IAD'
    parser.add_argument('--data_path', type=str, default=r'E:\IMSN-LW\dataset\mvtec_anomaly_detection') # Replace it with your path.

    # save info
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='INP-Former-Multi-Class')

    # model info
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14') # 'dinov2reg_vit_small_14' or 'dinov2reg_vit_base_14' or 'dinov2reg_vit_large_14'
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=392)
    parser.add_argument('--INP_num', type=int, default=6)

    # training info
    parser.add_argument('--total_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--phase', type=str, default='train')

    parser.add_argument('--npy_dir', type=str, default=None, help='Directory to save visualization images')
    parser.add_argument('--load_model', type=str, default=None, help='Path to model .pth file for inference.')

    args = parser.parse_args()
    args.save_name = args.save_name + f'_dataset={args.dataset}_Encoder={args.encoder}_Resize={args.input_size}_Crop={args.crop_size}_INP_num={args.INP_num}'
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'ICME':
        args.item_list = ['ICME']
    main(args)
