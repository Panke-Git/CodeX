from typing import List

import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


def _group_norm(num_channels: int, num_groups: int = 8) -> nn.GroupNorm:
    groups = min(num_groups, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.block1 = nn.Sequential(
            _group_norm(in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            _group_norm(out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )
        self.res_conv = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        time_emb = self.time_mlp(t)[:, :, None, None]
        h = h + time_emb
        h = self.block2(h)
        return h + self.res_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.group_norm = _group_norm(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.group_norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        b, c, hgt, wid = q.shape
        q = q.reshape(b, c, hgt * wid).permute(0, 2, 1)
        k = k.reshape(b, c, hgt * wid)
        attn = torch.bmm(q, k) * (c ** -0.5)
        attn = attn.softmax(dim=-1)
        v = v.reshape(b, c, hgt * wid).permute(0, 2, 1)
        out = torch.bmm(attn, v)
        out = out.permute(0, 2, 1).reshape(b, c, hgt, wid)
        out = self.proj(out)
        return out + x


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConditionalUNet(nn.Module):
    """U-Net backbone that conditions on the input underwater image via channel concatenation."""

    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        base_channels: int,
        channel_mults: List[int],
        num_res_blocks: int,
        dropout: float,
        use_attention: bool = True,
    ) -> None:
        super().__init__()

        time_dim = base_channels * 4
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv2d(in_channels + cond_channels, base_channels, kernel_size=3, padding=1)

        self.downs = nn.ModuleList()
        self.attn_down = nn.ModuleList()
        ch = base_channels
        skip_channels: List[int] = []
        for mult in channel_mults:
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlock(ch, out_ch, time_dim, dropout))
                ch = out_ch
                skip_channels.append(ch)
                self.attn_down.append(AttentionBlock(ch) if use_attention else nn.Identity())
            self.downs.append(Downsample(ch))
            self.attn_down.append(nn.Identity())

        self.mid = nn.ModuleList(
            [
                ResidualBlock(ch, ch, time_dim, dropout),
                AttentionBlock(ch) if use_attention else nn.Identity(),
                ResidualBlock(ch, ch, time_dim, dropout),
            ]
        )

        self.ups = nn.ModuleList()
        self.attn_up = nn.ModuleList()
        skip_stack = list(reversed(skip_channels))
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                skip_ch = skip_stack.pop()
                self.ups.append(ResidualBlock(ch + skip_ch, out_ch, time_dim, dropout))
                self.attn_up.append(AttentionBlock(out_ch) if use_attention else nn.Identity())
                ch = out_ch
            self.ups.append(Upsample(ch))
            self.attn_up.append(nn.Identity())

        self.final_norm = _group_norm(ch)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(ch, in_channels, kernel_size=3, padding=1)

    def forward(self, noisy: torch.Tensor, timesteps: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(timesteps)
        x = self.init_conv(torch.cat([noisy, cond], dim=1))

        skips = []
        for layer, attn in zip(self.downs, self.attn_down):
            if isinstance(layer, ResidualBlock):
                x = layer(x, t_emb)
                x = attn(x)
                skips.append(x)
            else:
                x = layer(x)

        for layer in self.mid:
            if isinstance(layer, ResidualBlock):
                x = layer(x, t_emb)
            else:
                x = layer(x)

        for layer, attn in zip(self.ups, self.attn_up):
            if isinstance(layer, ResidualBlock):
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = layer(x, t_emb)
                x = attn(x)
            else:
                x = layer(x)

        x = self.final_norm(x)
        x = self.final_act(x)
        return self.final_conv(x)
