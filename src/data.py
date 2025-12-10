from pathlib import Path
from typing import Dict, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def build_transforms(image_size: int, channels: int, augmentation: Optional[Dict]) -> transforms.Compose:
    aug = []
    if augmentation:
        if augmentation.get("horizontal_flip", False):
            aug.append(transforms.RandomHorizontalFlip())
        if augmentation.get("vertical_flip", False):
            aug.append(transforms.RandomVerticalFlip())
        if augmentation.get("color_jitter", False):
            aug.append(
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                )
            )
    aug.extend(
        [
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:channels, ...]),
            transforms.Normalize(mean=[0.5] * channels, std=[0.5] * channels),
        ]
    )
    return transforms.Compose(aug)


class UnderwaterPairDataset(Dataset):
    """Dataset for paired underwater images (input) and ground-truth targets."""

    def __init__(
        self,
        input_dir: str,
        target_dir: str,
        image_size: int,
        channels: int,
        augmentation: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        if not self.target_dir.exists():
            raise FileNotFoundError(f"Target directory not found: {self.target_dir}")

        self.extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        self.target_paths = sorted(
            [p for p in self.target_dir.iterdir() if p.suffix.lower() in self.extensions]
        )
        if not self.target_paths:
            raise ValueError(f"No images found under {self.target_dir}")

        self.transforms = build_transforms(image_size, channels, augmentation)

    def __len__(self) -> int:
        return len(self.target_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        target_path = self.target_paths[idx]
        input_path = self.input_dir / target_path.name
        if not input_path.exists():
            raise FileNotFoundError(
                f"Missing paired input image for {target_path.name} under {self.input_dir}"
            )

        input_img = Image.open(input_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")

        return {
            "input": self.transforms(input_img),
            "target": self.transforms(target_img),
            "input_path": str(input_path),
            "target_path": str(target_path),
        }


def create_dataloader(
    split_cfg: Dict,
    data_cfg: Dict,
    shuffle: bool = True,
) -> DataLoader:
    dataset = UnderwaterPairDataset(
        input_dir=split_cfg["input_dir"],
        target_dir=split_cfg["target_dir"],
        image_size=data_cfg["image_size"],
        channels=data_cfg["channels"],
        augmentation=data_cfg.get("augmentation") if shuffle else None,
    )
    batch_size = split_cfg.get(
        "batch_size", data_cfg.get("batch_size") if shuffle else data_cfg.get("val_batch_size", data_cfg.get("batch_size"))
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=False,
    )
