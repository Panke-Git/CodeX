import json
from pathlib import Path
from typing import Dict, List

import torch
from torchvision.utils import make_grid, save_image


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_output_dirs(cfg: Dict) -> Dict[str, Path]:
    name = cfg["experiment"].get("name", "run")
    root = Path(cfg["experiment"].get("output_root", "runs"))
    output_dir = root / name
    checkpoints = output_dir / "checkpoints"
    samples = output_dir / "samples"
    inference = output_dir / "inference"

    checkpoints.mkdir(parents=True, exist_ok=True)
    samples.mkdir(parents=True, exist_ok=True)
    inference.mkdir(parents=True, exist_ok=True)

    return {
        "root": output_dir,
        "checkpoints": checkpoints,
        "samples": samples,
        "inference": inference,
        "metrics": output_dir / "metrics.json",
    }


def save_tensor_images(tensor: torch.Tensor, path: Path) -> None:
    tensor = tensor.clamp(-1, 1)
    tensor = (tensor + 1) * 0.5
    save_image(tensor, path)


def save_grid(tensor: torch.Tensor, path: Path, nrow: int = 4) -> None:
    tensor = tensor.clamp(-1, 1)
    tensor = (tensor + 1) * 0.5
    grid = make_grid(tensor, nrow=nrow)
    save_image(grid, path)


def write_metrics(history: List[Dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
