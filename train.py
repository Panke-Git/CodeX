import argparse
import yaml
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import create_dataloader
from src.diffusion import GaussianDiffusion
from src.metrics import psnr, ssim
from src.models.unet import ConditionalUNet
from src.utils import prepare_output_dirs, save_grid, set_seed, write_metrics


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1) * 0.5


def evaluate(
    diffusion: GaussianDiffusion,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    total = 0
    with torch.no_grad():
        for batch in loader:
            cond = batch["input"].to(device)
            target = batch["target"].to(device)
            t = torch.randint(0, diffusion.timesteps, (target.size(0),), device=device)
            noise = torch.randn_like(target)
            noisy = diffusion.q_sample(target, t, noise)
            pred_noise = model(noisy, t, cond)
            loss = torch.nn.functional.mse_loss(pred_noise, noise)
            recon = diffusion.predict_start_from_noise(noisy, t, pred_noise)

            target_01 = to_01(target)
            recon_01 = to_01(recon)
            total_ssim += ssim(recon_01, target_01).sum().item()
            total_psnr += psnr(recon_01, target_01).sum().item()
            total_loss += loss.item() * target.size(0)
            total += target.size(0)
    return {
        "loss": total_loss / max(total, 1),
        "ssim": total_ssim / max(total, 1),
        "psnr": total_psnr / max(total, 1),
    }


def sample_batch(
    diffusion: GaussianDiffusion,
    model: nn.Module,
    cond: torch.Tensor,
    device: torch.device,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    model.eval()
    original_model = diffusion.model
    diffusion.model = model
    sampler = diffusion.ddim_sample if cfg["sampling"].get("ddim_steps", 0) else diffusion.sample
    if sampler == diffusion.ddim_sample:
        out = sampler(cond, device=device, num_steps=cfg["sampling"]["ddim_steps"])
    else:
        out = sampler(cond, device=device)
    diffusion.model = original_model
    return out


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    ema_state: Dict[str, torch.Tensor] | None,
) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    if ema_state is not None:
        state["ema_model"] = ema_state
    torch.save(state, path)


def load_checkpoint(model: nn.Module, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])


def run_inference(
    diffusion: GaussianDiffusion,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    cfg: Dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Infer {out_dir.name}")):
            cond = batch["input"].to(device)
            enhanced = sample_batch(diffusion, model, cond, device, cfg)
            save_grid(enhanced, out_dir / f"batch_{batch_idx:04d}.png", nrow=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train conditional DDPM for underwater enhancement")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="YAML config path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"].get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = create_dataloader(cfg["data"]["train"], cfg["data"], shuffle=True)
    val_loader = create_dataloader(cfg["data"]["val"], cfg["data"], shuffle=False)

    channels = cfg["data"]["channels"]
    model = ConditionalUNet(
        in_channels=channels,
        cond_channels=channels,
        base_channels=cfg["model"]["base_channels"],
        channel_mults=cfg["model"]["channel_mults"],
        num_res_blocks=cfg["model"]["num_res_blocks"],
        dropout=cfg["model"].get("dropout", 0.0),
    ).to(device)

    diffusion = GaussianDiffusion(
        model=model,
        image_size=cfg["data"]["image_size"],
        channels=channels,
        timesteps=cfg["diffusion"]["timesteps"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )

    scaler = GradScaler(enabled=cfg["train"].get("mixed_precision", False))

    ema_decay = cfg["train"].get("ema_decay", 0.0)
    ema_model = None
    if ema_decay > 0:
        ema_model = ConditionalUNet(
            in_channels=channels,
            cond_channels=channels,
            base_channels=cfg["model"]["base_channels"],
            channel_mults=cfg["model"]["channel_mults"],
            num_res_blocks=cfg["model"]["num_res_blocks"],
            dropout=cfg["model"].get("dropout", 0.0),
        ).to(device)
        ema_model.load_state_dict(model.state_dict())

    paths = prepare_output_dirs(cfg)

    best = {
        "loss": {"value": float("inf"), "path": None},
        "ssim": {"value": float("-inf"), "path": None},
        "psnr": {"value": float("-inf"), "path": None},
    }
    history: list[Dict[str, float]] = []

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        total = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['train']['epochs']}")
        for batch in progress:
            cond = batch["input"].to(device)
            target = batch["target"].to(device)
            t = torch.randint(0, diffusion.timesteps, (target.size(0),), device=device)
            noise = torch.randn_like(target)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg["train"].get("mixed_precision", False)):
                loss = diffusion.p_losses(target, cond, t, noise)
            scaler.scale(loss).backward()
            if cfg["train"].get("grad_clip", 0.0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            if ema_model is not None:
                with torch.no_grad():
                    for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                        ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

            epoch_loss += loss.item() * target.size(0)
            total += target.size(0)
            progress.set_postfix(loss=loss.item())

        train_loss = epoch_loss / max(total, 1)
        metrics = evaluate(diffusion, ema_model or model, val_loader, device)
        metrics["train_loss"] = train_loss
        metrics["epoch"] = epoch
        history.append(metrics)
        write_metrics(history, paths["metrics"])

        print(
            f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={metrics['loss']:.6f}, "
            f"SSIM={metrics['ssim']:.4f}, PSNR={metrics['psnr']:.2f}"
        )

        def _save_if_best(metric: str, value: float) -> None:
            better = value < best[metric]["value"] if metric == "loss" else value > best[metric]["value"]
            if better:
                best[metric]["value"] = value
                ckpt_path = paths["checkpoints"] / f"best_{metric}.pt"
                save_checkpoint(
                    ckpt_path,
                    model,
                    optimizer,
                    epoch,
                    ema_model.state_dict() if ema_model is not None else None,
                )
                best[metric]["path"] = ckpt_path

        _save_if_best("loss", metrics["loss"])
        _save_if_best("ssim", metrics["ssim"])
        _save_if_best("psnr", metrics["psnr"])

        if epoch % cfg["train"].get("sample_every", 1) == 0:
            val_batch = next(iter(val_loader))
            cond = val_batch["input"].to(device)[: cfg["sampling"].get("num_samples", 4)]
            sampler_model = ema_model or model
            enhanced = sample_batch(diffusion, sampler_model, cond, device, cfg)
            save_grid(enhanced, paths["samples"] / f"epoch_{epoch:04d}.png")

    print("Training complete. Running inference for best checkpoints...")

    for metric, record in best.items():
        if record["path"] is None:
            continue
        load_checkpoint(model, Path(record["path"]))
        if ema_model is not None and "ema_model" in torch.load(record["path"], map_location="cpu"):
            ema_state = torch.load(record["path"], map_location="cpu").get("ema_model")
            if ema_state:
                ema_model.load_state_dict(ema_state)
                model_to_use = ema_model
            else:
                model_to_use = model
        else:
            model_to_use = model
        run_inference(
            diffusion,
            model_to_use,
            val_loader,
            device,
            paths["inference"] / f"best_{metric}",
            cfg,
        )


if __name__ == "__main__":
    main()
