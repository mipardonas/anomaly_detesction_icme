from pathlib import Path
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from typing import Union, Optional, Callable, List, Tuple, Dict

# from app_logging import get_logger
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# logger = get_logger(__name__)

class Track1Dataset(Dataset):
    def __init__(
        self,
        # root_dir: Path | str,
        root_dir: Union[Path, str],
        *,
        include_ok: bool = True,
        load_images: bool = True,
        image_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        split: Optional[str] = None,   # 新增参数：'train' / 'val' / 'test' / None
    ):
        super().__init__()
        self._root = root_dir if isinstance(root_dir, Path) else Path(root_dir)
        self._include_ok = include_ok
        self._load_images = load_images
        self._image_transform = image_transform
        self._split = split   # 保存 split

        self._class_names = self._load_class_names()
        self._samples = self._build_samples()

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Tuple[Union[torch.Tensor, Path], dict]:
        sample = self._samples[index]
        image_path: Path = sample["image_path"]
        # label_path: Path | None = sample["label_path"]
        label_path: Union[Path, str] = sample["label_path"]

        if self._load_images:
            image = self._load_image_tensor(image_path)
            image_size = (int(image.shape[-2]), int(image.shape[-1]))
        else:
            image = image_path
            image_size = self._get_image_size(image_path)

        labels, polygons = self._parse_label_file(label_path)
        target = {
            "image_id": torch.tensor(index, dtype=torch.int64),
            "image_path": str(image_path),
            "is_defective": torch.tensor(int(sample["is_defective"]), dtype=torch.int64),
            "labels": labels,
            "polygons": polygons,
            "image_size": torch.tensor(image_size, dtype=torch.int64),
        }

        if self._image_transform is not None:
            if isinstance(image, Path):
                raise ValueError("image_transform requires load_images=True.")
            image = self._image_transform(image)

        return image, target

    def _load_class_names(self) -> List[str]:
        class_file = self._root / "class-name.txt"
        if not class_file.exists():
            raise FileNotFoundError(f"Class file not found: {class_file}")

        class_names: List[str] = []
        with class_file.open("r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name:
                    class_names.append(name)

        if not class_names:
            raise ValueError(f"No classes found in: {class_file}")
        return class_names

    def _build_samples(self) -> List[dict]:
        samples: List[dict] = []

        # 根据 split 决定子目录
        if self._split is None:
            # 原行为：直接从 NG_1154/images 和 OK_901/images 根目录取
            ng_image_dir = self._root / "NG_1154" / "images"
            ng_label_dir = self._root / "NG_1154" / "labels"
            ok_image_dir = self._root / "OK_901" / "images"
        else:
            # 使用 split 子目录：例如 NG_1154/images/train
            ng_image_dir = self._root / "NG_1154" / "images" / self._split
            ng_label_dir = self._root / "NG_1154" / "labels" / self._split
            ok_image_dir = self._root / "OK_901" / "images" / self._split

        if not ng_image_dir.exists() or not ng_label_dir.exists():
            # raise FileNotFoundError(f"Expected NG folders are missing: {ng_image_dir} and/or {ng_label_dir}")
            pass

        # 加载 NG 样本（异常）
        for image_path in sorted(ng_image_dir.glob("*.bmp")):
            label_path = ng_label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for NG image: {image_path.name}")

            samples.append(
                {
                    "image_path": image_path,
                    "label_path": label_path,
                    "is_defective": True,
                }
            )

        if self._include_ok:
            if not ok_image_dir.exists():
                # 如果是 val/test 且没有 OK 图片（例如测试集可能不含 OK），则跳过警告
                # 这里简单检查，如果目录不存在则跳过
                pass
            else:
                for image_path in sorted(ok_image_dir.glob("*.bmp")):
                    samples.append({
                        "image_path": image_path,
                        "label_path": None,
                        "is_defective": False,
                    })

        if not samples:
            raise ValueError(f"No samples found under: {self._root}")

        return samples

    def _parse_label_file(self, label_path: Optional[Path]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if label_path is None:
            return torch.empty((0,), dtype=torch.int64), []

        labels: List[int] = []
        polygons: List[torch.Tensor] = []

        with label_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue

                parts = raw.split()
                if len(parts) < 7:
                    raise ValueError(f"Invalid label format in {label_path} at line {line_number}: {raw}")

                class_id = int(parts[0])
                coords = [float(v) for v in parts[1:]]
                if len(coords) % 2 != 0:
                    raise ValueError(f"Polygon coordinates must be x/y pairs in {label_path} at line {line_number}")

                labels.append(class_id)
                polygons.append(torch.tensor(coords, dtype=torch.float32).view(-1, 2))

        return torch.tensor(labels, dtype=torch.int64), polygons

    def _load_image_tensor(self, image_path: Path) -> torch.Tensor:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read image with cv2: {image_path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        return image

    def _get_image_size(self, image_path: Path) -> Tuple[int, int]:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read image with cv2: {image_path}")
        height, width = bgr.shape[:2]
        return int(height), int(width)


def track1_collate_fn(batch: List[Tuple[Union[torch.Tensor, Path], dict]]) -> Tuple[List[Union[torch.Tensor, Path]], List[dict]]:
    images, targets = zip(*batch)
    return List(images), List(targets)


def create_track1_dataloader(
    # root_dir: Path | str,
    root_dir: Union[Path, str],
    *,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    include_ok: bool = True,
    load_images: bool = True,
    image_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[Track1Dataset, DataLoader]:
    dataset = Track1Dataset(
        root_dir,
        include_ok=include_ok,
        load_images=load_images,
        image_transform=image_transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=track1_collate_fn,
    )
    return dataset, dataloader


def overlay_polygons_on_image(
    image: torch.Tensor,
    target: dict,
    # class_names: list[str] | None = None,
    class_names: Union[List[str], None] = None,
    *,
    line_thickness: int = 1,
) -> Tuple[Figure, List[Axes]]:
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor shape [C, H, W], got: {Tuple(image.shape)}")

    h = int(image.shape[-2])
    w = int(image.shape[-1])

    image_vis = image.detach().cpu()
    if image_vis.dtype != torch.uint8:
        image_vis = (image_vis.clamp(0, 1) * 255).to(torch.uint8)
    image_rgb = image_vis.permute(1, 2, 0).numpy()

    labels: torch.Tensor = target["labels"]
    polygons: list[torch.Tensor] = target["polygons"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(image_rgb)
    axes[1].set_title("Overlay")
    axes[1].axis("off")

    for label, polygon in zip(labels.tolist(), polygons, strict=False):
        points = polygon.detach().cpu().numpy().copy()
        points[:, 0] = points[:, 0] * w
        points[:, 1] = points[:, 1] * h

        if len(points) > 0:
            patch = Polygon(points, closed=True, fill=False, edgecolor="lime", linewidth=max(line_thickness, 1))
            axes[1].add_patch(patch)

            text = str(label)
            if class_names is not None and 0 <= label < len(class_names):
                text = f"{label}:{class_names[label]}"
            p0 = points[0]
            axes[1].text(
                float(p0[0]),
                float(p0[1]) - 3,
                text,
                color="yellow",
                fontsize=7,
                bbox={"facecolor": "black", "alpha": 0.35, "pad": 1, "edgecolor": "none"},
            )

    fig.tight_layout()
    return fig, [axes[0], axes[1]]


if __name__ == "__main__":
    main()