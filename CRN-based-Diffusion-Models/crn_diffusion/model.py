"""
U-Net 预测网络：给定 (x_t, t) → 预测边际动量 p_t。

架构：DDPM 风格的 U-Net，带 sinusoidal 时间嵌入和 GroupNorm 归一化。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from . import config


class SinusoidalTimeEmbedding(nn.Module):
    """将时间步 t ∈ ℝ 映射到 d 维 embedding。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        embeddings = math.log(10000) / (half - 1)
        embeddings = torch.exp(torch.arange(half, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ResBlock(nn.Module):
    """带时间调节的残差块。"""

    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.dropout = nn.Dropout(dropout)

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """简化的自注意力块（在低分辨率层使用）。"""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.permute(0, 2, 3, 1).reshape(B * H * W, C)
        qkv = self.qkv(h).reshape(B, H * W, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = C ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.reshape(B * H * W, C)
        out = self.proj(out)
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
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
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net，预测边际动量 p_t。

    输入: x_t (B, 1, 28, 28) + 时间嵌入 t
    输出: p_t (B, 1, 28, 28)
    """

    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or config.Config
        chs = cfg.CHANNELS
        time_emb_dim = cfg.TIME_EMB_DIM
        img_ch = cfg.IMG_CHANNELS
        dropout = cfg.DROPOUT

        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.input_conv = nn.Conv2d(img_ch, chs[0], 3, padding=1)

        # Encoder
        self.down1 = nn.ModuleList([
            ResBlock(chs[0], chs[0], time_emb_dim, dropout),
            ResBlock(chs[0], chs[0], time_emb_dim, dropout),
        ])
        self.down2 = nn.ModuleList([
            ResBlock(chs[0], chs[1], time_emb_dim, dropout),
            ResBlock(chs[1], chs[1], time_emb_dim, dropout),
        ])
        self.down3 = nn.ModuleList([
            ResBlock(chs[1], chs[2], time_emb_dim, dropout),
            ResBlock(chs[2], chs[2], time_emb_dim, dropout),
        ])
        self.down4 = nn.ModuleList([
            ResBlock(chs[2], chs[3], time_emb_dim, dropout),
            ResBlock(chs[3], chs[3], time_emb_dim, dropout),
        ])

        self.downsample1 = Downsample(chs[0])
        self.downsample2 = Downsample(chs[1])
        self.downsample3 = Downsample(chs[2])

        self.middle = nn.ModuleList([
            ResBlock(chs[3], chs[3], time_emb_dim, dropout),
            ResBlock(chs[3], chs[3], time_emb_dim, dropout),
        ])

        # Decoder
        # upsampleN takes the output of the previous stage as input,
        # so its channel count must match that stage's output, not the skip.
        self.upsample3 = Upsample(chs[3])          # 512 → 512, then cat with x3_skip(chs[2]=256)
        self.up3 = nn.ModuleList([
            ResBlock(chs[3] + chs[2], chs[2], time_emb_dim, dropout),
            ResBlock(chs[2], chs[2], time_emb_dim, dropout),
        ])

        self.upsample2 = Upsample(chs[2])          # 256 → 256, then cat with x2_skip(chs[1]=128)
        self.up2 = nn.ModuleList([
            ResBlock(chs[2] + chs[1], chs[1], time_emb_dim, dropout),
            ResBlock(chs[1], chs[1], time_emb_dim, dropout),
        ])

        self.upsample1 = Upsample(chs[1])          # 128 → 128, then cat with x1_skip(chs[0]=64)
        self.up1 = nn.ModuleList([
            ResBlock(chs[1] + chs[0], chs[0], time_emb_dim, dropout),
            ResBlock(chs[0], chs[0], time_emb_dim, dropout),
        ])

        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, chs[0]),
            nn.SiLU(),
            nn.Conv2d(chs[0], img_ch, 3, padding=1),  # img_ch=1 for MNIST, 3 for CIFAR-100
        )

    def forward(self, x, t):
        # Pad spatial dims to the nearest multiple of 8 so three halvings stay integer.
        # 28×28 → 32×32 (pad 2 each side); 32×32 → 32×32 (no pad needed).
        H_orig, W_orig = x.shape[-2], x.shape[-1]
        pad_h = (8 - H_orig % 8) % 8
        pad_w = (8 - W_orig % 8) % 8
        # distribute padding: more on right/bottom
        ph_top, ph_bot = pad_h // 2, pad_h - pad_h // 2
        pw_left, pw_right = pad_w // 2, pad_w - pad_w // 2
        x = F.pad(x, (pw_left, pw_right, ph_top, ph_bot))

        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)

        x = self.input_conv(x)

        for blk in self.down1:
            x = blk(x, t_emb)
        x1_skip = x                  # (B, 64, 32, 32)
        x = self.downsample1(x)      # (B, 64, 16, 16)

        for blk in self.down2:
            x = blk(x, t_emb)
        x2_skip = x                  # (B, 128, 16, 16)
        x = self.downsample2(x)      # (B, 128, 8, 8)

        for blk in self.down3:
            x = blk(x, t_emb)
        x3_skip = x                  # (B, 256, 8, 8)
        x = self.downsample3(x)      # (B, 256, 4, 4)

        for blk in self.down4:
            x = blk(x, t_emb)       # (B, 512, 4, 4)

        for blk in self.middle:
            if isinstance(blk, AttentionBlock):
                x = blk(x)
            else:
                x = blk(x, t_emb)

        x = self.upsample3(x)        # (B, 512, 8, 8)
        x = torch.cat([x, x3_skip], dim=1)   # (B, 768, 8, 8)
        for blk in self.up3:
            x = blk(x, t_emb)       # (B, 256, 8, 8)

        x = self.upsample2(x)        # (B, 256, 16, 16)
        x = torch.cat([x, x2_skip], dim=1)   # (B, 384, 16, 16)
        for blk in self.up2:
            x = blk(x, t_emb)       # (B, 128, 16, 16)

        x = self.upsample1(x)        # (B, 128, 32, 32)
        x = torch.cat([x, x1_skip], dim=1)   # (B, 192, 32, 32)
        for blk in self.up1:
            x = blk(x, t_emb)       # (B, 64, 32, 32)

        x = self.output_conv(x)      # (B, 1, 32, 32)

        # Crop back to original spatial size
        x = x[:, :, ph_top:ph_top + H_orig, pw_left:pw_left + W_orig]
        return x


class UNetCIFAR(nn.Module):
    """
    增强版 U-Net，专为 CIFAR-100 (32×32×3) 设计。
    相比 UNet：在 8×8 和 16×16 分辨率额外加 AttentionBlock，
    以捕捉 CIFAR-100 复杂纹理的中频结构。
    MNIST 训练继续使用原 UNet，互不影响。
    """

    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or config.Config
        chs = cfg.CHANNELS          # [64, 128, 256, 512]
        time_emb_dim = cfg.TIME_EMB_DIM
        img_ch = cfg.IMG_CHANNELS   # 3 for CIFAR-100
        dropout = cfg.DROPOUT

        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.input_conv = nn.Conv2d(img_ch, chs[0], 3, padding=1)

        # Encoder
        self.down1 = nn.ModuleList([
            ResBlock(chs[0], chs[0], time_emb_dim, dropout),
            ResBlock(chs[0], chs[0], time_emb_dim, dropout),
        ])
        self.down2 = nn.ModuleList([
            ResBlock(chs[0], chs[1], time_emb_dim, dropout),
            ResBlock(chs[1], chs[1], time_emb_dim, dropout),
        ])
        self.attn2 = AttentionBlock(chs[1])   # attention at 16×16

        self.down3 = nn.ModuleList([
            ResBlock(chs[1], chs[2], time_emb_dim, dropout),
            ResBlock(chs[2], chs[2], time_emb_dim, dropout),
        ])
        self.attn3 = AttentionBlock(chs[2])   # attention at 8×8

        self.down4 = nn.ModuleList([
            ResBlock(chs[2], chs[3], time_emb_dim, dropout),
            ResBlock(chs[3], chs[3], time_emb_dim, dropout),
        ])

        self.downsample1 = Downsample(chs[0])
        self.downsample2 = Downsample(chs[1])
        self.downsample3 = Downsample(chs[2])

        self.middle = nn.ModuleList([
            ResBlock(chs[3], chs[3], time_emb_dim, dropout),
            AttentionBlock(chs[3]),             # attention at 4×4
            ResBlock(chs[3], chs[3], time_emb_dim, dropout),
        ])

        # Decoder
        self.upsample3 = Upsample(chs[3])
        self.up3 = nn.ModuleList([
            ResBlock(chs[3] + chs[2], chs[2], time_emb_dim, dropout),
            ResBlock(chs[2], chs[2], time_emb_dim, dropout),
        ])
        self.up_attn3 = AttentionBlock(chs[2])  # attention at 8×8

        self.upsample2 = Upsample(chs[2])
        self.up2 = nn.ModuleList([
            ResBlock(chs[2] + chs[1], chs[1], time_emb_dim, dropout),
            ResBlock(chs[1], chs[1], time_emb_dim, dropout),
        ])
        self.up_attn2 = AttentionBlock(chs[1])  # attention at 16×16

        self.upsample1 = Upsample(chs[1])
        self.up1 = nn.ModuleList([
            ResBlock(chs[1] + chs[0], chs[0], time_emb_dim, dropout),
            ResBlock(chs[0], chs[0], time_emb_dim, dropout),
        ])

        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, chs[0]),
            nn.SiLU(),
            nn.Conv2d(chs[0], img_ch, 3, padding=1),
        )

    def forward(self, x, t):
        H_orig, W_orig = x.shape[-2], x.shape[-1]
        pad_h = (8 - H_orig % 8) % 8
        pad_w = (8 - W_orig % 8) % 8
        ph_top, ph_bot = pad_h // 2, pad_h - pad_h // 2
        pw_left, pw_right = pad_w // 2, pad_w - pad_w // 2
        x = F.pad(x, (pw_left, pw_right, ph_top, ph_bot))

        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)

        x = self.input_conv(x)

        for blk in self.down1:
            x = blk(x, t_emb)
        x1_skip = x
        x = self.downsample1(x)

        for blk in self.down2:
            x = blk(x, t_emb)
        x = self.attn2(x)
        x2_skip = x
        x = self.downsample2(x)

        for blk in self.down3:
            x = blk(x, t_emb)
        x = self.attn3(x)
        x3_skip = x
        x = self.downsample3(x)

        for blk in self.down4:
            x = blk(x, t_emb)

        for blk in self.middle:
            if isinstance(blk, AttentionBlock):
                x = blk(x)
            else:
                x = blk(x, t_emb)

        x = self.upsample3(x)
        x = torch.cat([x, x3_skip], dim=1)
        for blk in self.up3:
            x = blk(x, t_emb)
        x = self.up_attn3(x)

        x = self.upsample2(x)
        x = torch.cat([x, x2_skip], dim=1)
        for blk in self.up2:
            x = blk(x, t_emb)
        x = self.up_attn2(x)

        x = self.upsample1(x)
        x = torch.cat([x, x1_skip], dim=1)
        for blk in self.up1:
            x = blk(x, t_emb)

        x = self.output_conv(x)
        x = x[:, :, ph_top:ph_top + H_orig, pw_left:pw_left + W_orig]
        return x


class EMA:
    """
    指数移动平均权重更新（DDPM 标准做法）。
    """

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)

    x = torch.rand(8, 1, 28, 28, device=device)
    t = torch.rand(8, device=device) * config.Config.T
    out = model(x, t)

    print(f"输入 shape:  {x.shape}")
    print(f"输出 shape:  {out.shape}")
    print(f"参数总量:    {sum(p.numel() for p in model.parameters()):,}")
    print(f"输出范围:    [{out.min():.3f}, {out.max():.3f}]")
