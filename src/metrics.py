import math
from typing import Tuple

import torch
import torch.nn.functional as F


def _gaussian_kernel(kernel_size: int = 11, sigma: float = 1.5, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device) - kernel_size // 2
    grid = coords ** 2
    kernel_1d = torch.exp(-(grid) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Compute Structural Similarity (SSIM) index for batches."""
    if pred.ndim != 4:
        raise ValueError("SSIM expects NCHW tensors")

    device = pred.device
    kernel = _gaussian_kernel(device=device).unsqueeze(0).unsqueeze(0)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    padding = kernel.shape[-1] // 2

    def _filter(x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        kernel_expanded = kernel.expand(channels, 1, -1, -1)
        return F.conv2d(x, kernel_expanded, padding=padding, groups=channels)

    mu1 = _filter(pred)
    mu2 = _filter(target)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _filter(pred * pred) - mu1_sq
    sigma2_sq = _filter(target * target) - mu2_sq
    sigma12 = _filter(pred * target) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean(dim=[1, 2, 3])


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction="none")
    mse_per = mse.flatten(1).mean(dim=1)
    return 10 * torch.log10((data_range**2) / torch.clamp(mse_per, min=1e-10))
