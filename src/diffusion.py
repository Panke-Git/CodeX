from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Extract coefficients for a batch of time steps and reshape for broadcasting."""
    b = t.shape[0]
    out = a.gather(-1, t.to(a.device))
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


class GaussianDiffusion(nn.Module):
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

        betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        ) * noise

    def predict_start_from_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        sqrt_recip_alphas_cumprod = torch.sqrt(
            1.0 / extract(self.alphas_cumprod, t, x_t.shape)
        )
        sqrt_recipm1_alphas_cumprod = torch.sqrt(
            1.0 / extract(self.alphas_cumprod, t, x_t.shape) - 1.0
        )
        return sqrt_recip_alphas_cumprod * x_t - sqrt_recipm1_alphas_cumprod * noise

    def p_losses(
        self, x_start: torch.Tensor, cond: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(noisy, t, cond)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample(
        self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor, t_index: int
    ) -> torch.Tensor:
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = torch.sqrt(1.0 / (1.0 - betas_t))

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t / sqrt_one_minus_alphas_cumprod_t * self.model(x, t, cond)
        )

        if t_index == 0:
            return model_mean

        posterior_variance_t = extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, device: torch.device) -> torch.Tensor:
        batch_size = cond.shape[0]
        x = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=device)
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, t, i)
        return x

    @torch.no_grad()
    def ddim_sample(self, cond: torch.Tensor, device: torch.device, num_steps: int) -> torch.Tensor:
        step_ratio = max(self.timesteps // num_steps, 1)
        batch_size = cond.shape[0]
        x = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=device)
        for i in reversed(range(0, self.timesteps, step_ratio)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            betas_t = extract(self.betas, t, x.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(
                self.sqrt_one_minus_alphas_cumprod, t, x.shape
            )
            sqrt_recip_alphas_t = torch.sqrt(1.0 / (1.0 - betas_t))
            pred_noise = self.model(x, t, cond)
            model_mean = sqrt_recip_alphas_t * (x - betas_t / sqrt_one_minus_alphas_cumprod_t * pred_noise)
            x = model_mean
        return x

    @torch.no_grad()
    def reconstruct(self, x_start: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x_start)
        noisy = self.q_sample(x_start, t, noise)
        pred_noise = self.model(noisy, t, cond)
        return self.predict_start_from_noise(noisy, t, pred_noise)
