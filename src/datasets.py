"""Dataset definitions for underwater DDPM training.

该模块提供 `UnderwaterImageDataset`，用于加载输入与 GT 图像。默认只使用 GT
图像进行无条件扩散模型训练，但保留 input 图像路径，方便未来扩展成有
条件的模型或可视化对比。
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class UnderwaterImageDataset(Dataset):
    """自定义数据集，用于加载水下图像及对应 GT。

    假设 `gt_dir` 下存放干净的 PNG/JPEG 图像。若提供 `input_dir`，将尝试按
    文件名匹配，但目前训练直接使用 GT 图像；匹配逻辑主要用于未来可视化。
    """

    def __init__(
        self,
        gt_dir: str,
        input_dir: Optional[str],
        image_size: int,
        channels: int,
        augmentation_cfg: Dict[str, bool],
    ) -> None:
        super().__init__()
        self.gt_dir = Path(gt_dir)
        self.input_dir = Path(input_dir) if input_dir else None

        if not self.gt_dir.exists():
            raise FileNotFoundError(f"GT 路径不存在: {self.gt_dir}")

        # 支持的图像扩展名
        self.extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        self.gt_paths: List[Path] = [
            p for p in self.gt_dir.iterdir() if p.suffix.lower() in self.extensions
        ]
        if not self.gt_paths:
            raise ValueError(f"在 {self.gt_dir} 下未找到任何图像文件")

        # 定义数据增强与预处理
        aug_transforms: List[Callable] = []
        if augmentation_cfg.get("horizontal_flip", False):
            aug_transforms.append(transforms.RandomHorizontalFlip())
        if augmentation_cfg.get("color_jitter", False):
            aug_transforms.append(
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                )
            )

        # 统一的尺寸与归一化，将像素值映射到 [-1, 1]
        self.transform = transforms.Compose(
            [
                *aug_transforms,
                transforms.Resize(image_size, interpolation=Image.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x[:channels, ...]),
                transforms.Normalize((0.5,) * channels, (0.5,) * channels),
            ]
        )

    def __len__(self) -> int:
        return len(self.gt_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        gt_path = self.gt_paths[idx]
        image = Image.open(gt_path).convert("RGB")
        tensor = self.transform(image)

        sample = {"image": tensor, "gt_path": str(gt_path)}

        # 如果存在 input_dir，加载并返回输入图像，方便可视化对比
        if self.input_dir:
            candidate = self.input_dir / gt_path.name
            if candidate.exists():
                input_img = Image.open(candidate).convert("RGB")
                sample["input"] = self.transform(input_img)
                sample["input_path"] = str(candidate)
            else:
                sample["input_path"] = None

        return sample
