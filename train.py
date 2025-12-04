"""训练入口：从 YAML 读取配置，构建数据集、模型与扩散过程并启动训练。

使用方式：
    python train.py --config configs/default.yaml

训练过程中会：
- 在日志目录下保存配置备份、采样图、模型检查点
- 支持混合精度与梯度裁剪
- 每个 epoch 结束后可生成若干采样结果，方便观察收敛
"""

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import yaml

from src.datasets import UnderwaterImageDataset
from src.diffusion import GaussianDiffusion
from src.model import UNetModel


def load_config(path: str) -> Dict[str, Any]:
    """读取 YAML 配置文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """设置随机种子，确保实验可复现。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_dataloader(
    cfg: Dict[str, Any], split: str, shuffle: bool = True
) -> DataLoader:
    """根据 split 构建训练或验证 DataLoader。"""

    split_cfg = cfg["data"].get(split, {})
    if not split_cfg:
        raise ValueError(f"配置中缺少 {split} 数据集信息。")

    dataset = UnderwaterImageDataset(
        gt_dir=split_cfg["gt_dir"],
        input_dir=split_cfg.get("input_dir"),
        image_size=cfg["data"]["image_size"],
        channels=cfg["data"]["channels"],
        augmentation_cfg=cfg["data"].get("augmentation", {}) if split == "train" else {},
    )
    batch_size = (
        cfg["data"]["batch_size"] if split == "train" else cfg["data"].get("val_batch_size", cfg["data"]["batch_size"])
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=True,
    )


def save_samples(images: torch.Tensor, save_dir: Path, epoch: int, save_grid: bool) -> None:
    """保存采样结果；同时保存单张和网格。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    # 反归一化到 [0,1]
    images = (images.clamp(-1, 1) + 1) * 0.5

    for idx, img in enumerate(images):
        save_image(img, save_dir / f"epoch{epoch:04d}_sample{idx}.png")

    if save_grid:
        grid = make_grid(images, nrow=max(1, int(len(images) ** 0.5)))
        save_image(grid, save_dir / f"epoch{epoch:04d}_grid.png")


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(state, save_dir / f"epoch{epoch:04d}.pt")


def build_model(cfg: Dict[str, Any]) -> UNetModel:
    return UNetModel(
        in_channels=cfg["data"]["channels"],
        base_channels=cfg["model"]["base_channels"],
        channel_mults=cfg["model"]["channel_mults"],
        num_res_blocks=cfg["model"]["num_res_blocks"],
        dropout=cfg["model"].get("dropout", 0.0),
    )


def train(cfg: Dict[str, Any]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["experiment"].get("seed", 42))

    # 日志/输出目录
    output_dir = Path(cfg["experiment"]["output_dir"])
    samples_dir = output_dir / "samples"
    ckpt_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置备份，方便溯源
    with open(output_dir / "config_saved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    train_loader = prepare_dataloader(cfg, split="train", shuffle=True)
    val_loader = None
    if cfg["data"].get("val"):
        val_loader = prepare_dataloader(cfg, split="val", shuffle=False)
    model = build_model(cfg).to(device)
    diffusion = GaussianDiffusion(
        model=model,
        image_size=cfg["data"]["image_size"],
        channels=cfg["data"]["channels"],
        timesteps=cfg["diffusion"]["num_steps"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
    ).to(device)

    optimizer = optim.AdamW(
        diffusion.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    scaler = amp.GradScaler(enabled=cfg["train"].get("mixed_precision", True))

    ema_decay = cfg["train"].get("ema_decay", 1.0)
    ema_model = None
    if ema_decay < 1.0:
        ema_model = build_model(cfg).to(device)
        ema_model.load_state_dict(model.state_dict())

    num_epochs = cfg["train"]["num_epochs"]
    log_interval = cfg["train"].get("log_interval", 50)

    global_step = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            images = batch["image"].to(device)
            noise = torch.randn_like(images)
            t = torch.randint(0, cfg["diffusion"]["num_steps"], (images.shape[0],), device=device).long()

            with amp.autocast(enabled=scaler.is_enabled()):
                loss = diffusion.p_losses(images, t, noise)

            scaler.scale(loss).backward()
            if cfg["train"].get("grad_clip", 0.0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(diffusion.parameters(), cfg["train"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            if ema_model is not None:
                with torch.no_grad():
                    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                        p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

            global_step += 1
            if global_step % log_interval == 0:
                print(f"Epoch {epoch} Step {global_step}: loss={loss.item():.6f}")

        # 验证集评估：使用 EMA 模型（若存在），仅计算噪声预测 MSE
        val_loss = None
        if val_loader is not None:
            model_to_eval = ema_model if ema_model is not None else model
            model_to_eval.eval()
            total_loss = 0.0
            total_samples = 0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    noise = torch.randn_like(images)
                    t = torch.randint(
                        0, cfg["diffusion"]["num_steps"], (images.shape[0],), device=device
                    ).long()
                    loss_val = diffusion.p_losses(images, t, noise)
                    total_loss += loss_val.item() * images.size(0)
                    total_samples += images.size(0)
            val_loss = total_loss / max(total_samples, 1)
            print(f"Epoch {epoch}: validation loss={val_loss:.6f}")

        # 采样与保存
        if epoch % cfg["train"].get("sample_interval", 1) == 0:
            diffusion.eval()
            sampler = ema_model if ema_model is not None else model
            diffusion.model = sampler  # 暂时替换，便于采样
            with torch.no_grad():
                if cfg["sampling"].get("ddim_steps", 0) > 0:
                    samples = diffusion.ddim_sample(
                        batch_size=cfg["sampling"].get("num_samples", 4),
                        device=device,
                        num_steps=cfg["sampling"]["ddim_steps"],
                    )
                else:
                    samples = diffusion.sample(batch_size=cfg["sampling"].get("num_samples", 4), device=device)
            save_samples(samples, samples_dir, epoch, cfg["sampling"].get("save_grid", True))
            diffusion.model = model  # 采样后换回训练模型

        if epoch % cfg["train"].get("checkpoint_interval", 0) == 0:
            save_checkpoint(model, optimizer, epoch, ckpt_dir)

    print("训练完成！模型与采样结果已保存。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDPM for underwater images")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="路径指向 YAML 配置文件",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)
