"""
U-Net 与 EMA — 架构与 CRN 项目完全一致，仅语义不同（预测噪声 ε）。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import config


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freq = math.log(10000) / (half - 1)
        freq = torch.exp(torch.arange(half, device=device) * -freq)
        emb  = t[:, None] * freq[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1    = nn.GroupNorm(8, in_ch)
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.norm2    = nn.GroupNorm(8, out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.dropout  = nn.Dropout(dropout)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv  = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        h   = self.norm(x).permute(0, 2, 3, 1).reshape(B * H * W, C)
        qkv = self.qkv(h).reshape(B, H * W, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1) * C ** -0.5).softmax(dim=-1)
        out  = (attn @ v).reshape(B * H * W, C)
        out  = self.proj(out).reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x + out


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """
    输入: x_t (B, 1, 28, 28) + 整数时间步 t ∈ {1,...,T}
    输出: 预测噪声 ε (B, 1, 28, 28)
    """

    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or config.Config
        chs         = cfg.CHANNELS
        tdim        = cfg.TIME_EMB_DIM
        img_ch      = cfg.IMG_CHANNELS
        dropout     = cfg.DROPOUT

        self.time_embed = SinusoidalTimeEmbedding(tdim)
        self.time_mlp   = nn.Sequential(
            nn.Linear(tdim, tdim * 4), nn.SiLU(), nn.Linear(tdim * 4, tdim)
        )

        self.input_conv = nn.Conv2d(img_ch, chs[0], 3, padding=1)

        self.down1 = nn.ModuleList([ResBlock(chs[0], chs[0], tdim, dropout),
                                    ResBlock(chs[0], chs[0], tdim, dropout)])
        self.down2 = nn.ModuleList([ResBlock(chs[0], chs[1], tdim, dropout),
                                    ResBlock(chs[1], chs[1], tdim, dropout)])
        self.down3 = nn.ModuleList([ResBlock(chs[1], chs[2], tdim, dropout),
                                    ResBlock(chs[2], chs[2], tdim, dropout)])
        self.down4 = nn.ModuleList([ResBlock(chs[2], chs[3], tdim, dropout),
                                    ResBlock(chs[3], chs[3], tdim, dropout)])

        self.ds1 = Downsample(chs[0])
        self.ds2 = Downsample(chs[1])
        self.ds3 = Downsample(chs[2])

        self.middle = nn.ModuleList([
            ResBlock(chs[3], chs[3], tdim, dropout),
            AttentionBlock(chs[3]),
            ResBlock(chs[3], chs[3], tdim, dropout),
        ])

        self.us3  = Upsample(chs[3])
        self.up3  = nn.ModuleList([ResBlock(chs[3] + chs[2], chs[2], tdim, dropout),
                                   ResBlock(chs[2], chs[2], tdim, dropout)])

        self.us2  = Upsample(chs[2])
        self.up2  = nn.ModuleList([ResBlock(chs[2] + chs[1], chs[1], tdim, dropout),
                                   ResBlock(chs[1], chs[1], tdim, dropout)])

        self.us1  = Upsample(chs[1])
        self.up1  = nn.ModuleList([ResBlock(chs[1] + chs[0], chs[0], tdim, dropout),
                                   ResBlock(chs[0], chs[0], tdim, dropout)])

        self.out  = nn.Sequential(
            nn.GroupNorm(8, chs[0]), nn.SiLU(),
            nn.Conv2d(chs[0], img_ch, 3, padding=1),
        )

    def forward(self, x, t):
        # Pad 28×28 → 32×32 so spatial dims halve cleanly 3 times
        H, W = x.shape[-2], x.shape[-1]
        x = F.pad(x, (2, 2, 2, 2))

        t_emb = self.time_mlp(self.time_embed(t.float()))

        x = self.input_conv(x)

        for blk in self.down1: x = blk(x, t_emb)
        s1 = x
        x = self.ds1(x)

        for blk in self.down2: x = blk(x, t_emb)
        s2 = x
        x = self.ds2(x)

        for blk in self.down3: x = blk(x, t_emb)
        s3 = x
        x = self.ds3(x)

        for blk in self.down4: x = blk(x, t_emb)

        for blk in self.middle:
            x = blk(x) if isinstance(blk, AttentionBlock) else blk(x, t_emb)

        x = self.us3(x)
        x = torch.cat([x, s3], dim=1)
        for blk in self.up3: x = blk(x, t_emb)

        x = self.us2(x)
        x = torch.cat([x, s2], dim=1)
        for blk in self.up2: x = blk(x, t_emb)

        x = self.us1(x)
        x = torch.cat([x, s1], dim=1)
        for blk in self.up1: x = blk(x, t_emb)

        x = self.out(x)
        return x[:, :, 2:2 + H, 2:2 + W]


class EMA:
    def __init__(self, model, decay=0.999):
        self.model  = model
        self.decay  = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = self.decay * self.shadow[n] + (1 - self.decay) * p.data

    def apply_shadow(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data = self.shadow[n]

    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.data = self.backup[n]
        self.backup = {}
