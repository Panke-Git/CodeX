"""Core diffusion process utilities for DDPM training and sampling.

包含：
- Beta 调度、前向扩散 q(x_t | x_0) 的计算
- 训练时的损失计算（预测噪声的 MSE）
- 采样函数，支持 DDPM（逐步逆扩散）和 DDIM 加速采样
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """可选的余弦调度，返回 beta 序列。"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0001, 0.9999)


class GaussianDiffusion(nn.Module):
    """封装前向/反向扩散计算逻辑。"""

    def __init__(
        self,
        model: nn.Module,
        image_size: int,
        channels: int,
        timesteps: int,
        beta_start: float,
        beta_end: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps

        # 线性 beta 调度，可替换为 cosine_beta_schedule 等
        betas = torch.linspace(beta_start, beta_end, timesteps)
        self.register_buffer("betas", betas)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)

        # 预先缓存训练/采样需要的各类系数
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))
        self.register_buffer("posterior_variance", betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """根据 q(x_t | x_0) 对干净图像加入噪声。"""
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """预测噪声的 MSE 损失。"""
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, t_index: int) -> torch.Tensor:
        """单步逆扩散：p(x_{t-1} | x_t)。"""
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / (1.0 - betas_t))

        # 预测噪声并计算均值
        model_mean = sqrt_recip_alphas_t * (x - betas_t / sqrt_one_minus_alphas_cumprod_t * self.model(x, t))

        if t_index == 0:
            return model_mean

        posterior_variance_t = extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """使用 DDPM 逆扩散生成样本。"""
        x = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=device)
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, i)
        return x

    @torch.no_grad()
    def ddim_sample(self, batch_size: int, device: torch.device, num_steps: int) -> torch.Tensor:
        """DDIM 加速采样，步数可小于训练步数。"""
        step_ratio = max(self.timesteps // num_steps, 1)
        x = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=device)
        for i in reversed(range(0, self.timesteps, step_ratio)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            betas_t = extract(self.betas, t, x.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            sqrt_recip_alphas_t = torch.sqrt(1.0 / (1.0 - betas_t))
            pred_noise = self.model(x, t)
            model_mean = sqrt_recip_alphas_t * (x - betas_t / sqrt_one_minus_alphas_cumprod_t * pred_noise)

            if i == 0:
                x = model_mean
            else:
                # DDIM 无随机项，直接使用上一时刻均值
                x = model_mean
        return x


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """从预存向量中按时间步提取对应系数，并 reshape 以便广播。"""
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu()).to(t.device)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
