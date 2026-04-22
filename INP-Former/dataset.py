import random
from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import torch.multiprocessing
import json
from track1_dataloader import Track1Dataset
import torch
import cv2
from pathlib import Path

torch.multiprocessing.set_sharing_strategy('file_system')

def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms

def create_icme_datasets(
    root,
    transform,
    gt_transform,
    resize_size,
    crop_size,
    splits=['train', 'val', 'test']
):
    """
    创建ICME数据集，返回字典，键为split名称，值为对应的Dataset对象。
    各split返回格式：
        - 'train'      : 用于无监督训练，返回 (img, 0)
        - 'train_ng'   : 用于分类头训练，返回 (img, polygons, labels)
        - 'val'/'test' : 用于评估，返回 (img, gt_mask, label, img_path)
    """
    datasets = {}

    base_datasets = {}
    if 'train' in splits:
        pass
    if 'val' in splits:
        base_datasets['val'] = Track1Dataset(root, include_ok=True, load_images=False, image_transform=None, split='val')

    class OKOnlyTrainDataset(torch.utils.data.Dataset):
        def __init__(self, root, transform):
            self.transform = transform
            root_path = Path(root)
            if root_path.name == 'images' and root_path.parent.name == 'OK_901':
                self.img_root = root_path
            else:
                self.img_root = root_path / 'OK_901' / 'images'

            exts = {'.bmp', '.png', '.jpg', '.jpeg', '.BMP', '.PNG', '.JPG', '.JPEG'}
            self.img_paths = sorted(
                [p for p in self.img_root.rglob('*') if p.is_file() and p.suffix in exts]
            )
            if not self.img_paths:
                raise ValueError(f'No OK image files found under: {self.img_root}')

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx):
            img_path = self.img_paths[idx]
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img, 0

    class EvalDataset(torch.utils.data.Dataset):
        def __init__(self, base, transform, resize_size, crop_size):
            self.base = base
            self.transform = transform
            self.resize_size = resize_size
            self.crop_size = crop_size
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            image_path, target = self.base[idx]
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img)
            polygons = target['polygons']
            image_size = target['image_size'].tolist()
            gt_mask = self._polygons_to_mask(polygons, image_size)
            # 将 Path 对象转为字符串，避免 DataLoader 报错
            return img_tensor, gt_mask, target['is_defective'].item(), str(image_path)
        def _polygons_to_mask(self, polygons, image_size):
            # 保持原有实现不变
            import cv2
            import numpy as np
            H, W = image_size
            mask_orig = np.zeros((H, W), dtype=np.uint8)
            for poly in polygons:
                pts = poly.numpy().copy()
                pts[:, 0] *= W
                pts[:, 1] *= H
                pts = pts.astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask_orig, [pts], 1)
            mask_resized = cv2.resize(mask_orig, (self.resize_size, self.resize_size), interpolation=cv2.INTER_NEAREST)
            Hr, Wr = mask_resized.shape
            top = (Hr - self.crop_size) // 2
            left = (Wr - self.crop_size) // 2
            mask_cropped = mask_resized[top:top+self.crop_size, left:left+self.crop_size]
            return torch.from_numpy(mask_cropped).float().unsqueeze(0)

    # 构建数据集对象
    if 'train' in splits:
        # 原始逻辑：使用 Track1Dataset 元数据，再过滤正常样本。
        datasets['train'] = OKOnlyTrainDataset(root, transform)
    if 'val' in splits:
        datasets['val'] = EvalDataset(base_datasets['val'], transform, resize_size, crop_size)

    return datasets

class ImageOnlyTestDataset(torch.utils.data.Dataset):
    """
    用于只有图片、没有标注文件的测试目录。
    返回格式与 evaluation_batch 保持一致：
    (img, gt_mask, label, img_path)
    """
    def __init__(self, root, transform, crop_size):
        self.root = Path(root)
        self.transform = transform
        self.crop_size = crop_size

        exts = {'.bmp', '.png', '.jpg', '.jpeg', '.BMP', '.PNG', '.JPG', '.JPEG'}
        self.img_paths = sorted(
            [p for p in self.root.rglob('*') if p.is_file() and p.suffix in exts]
        )

        if not self.img_paths:
            raise ValueError(f'No image files found under: {self.root}')

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        gt = torch.zeros([1, self.crop_size, self.crop_size], dtype=torch.float32)
        label = torch.tensor(0, dtype=torch.int64)
        return img, gt, label, str(img_path)
