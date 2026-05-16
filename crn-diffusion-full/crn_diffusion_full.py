"""
CRN Diffusion Model — MNIST

数学框架: model.py 的精确 HJ (二分法求解特征线方程)
  - 前向: forward_tau_leap (Binomial + Poisson 精确子步, v0=100, T=5)
  - 条件动量: conditional_momentum (精确 HJ 二分法)
  - 反向: sample_reverse (HJ 动量驱动的跳跃采样)

网络: CRN-based-Diffusion-Models 的 U-Net (时间嵌入 + GroupNorm + Attention)
训练超参: batch=256, lr=1e-4, epochs=200, grad_clip=1.0, EMA=0.999
"""

import argparse
import math
import os
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import wandb

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

V0          = 100.0
T           = 5.0
S_EPS       = 0.02
N_TAU_STEPS = 100
TARGET_CLIP = 20.0
P_CLIP      = 6.0
N_SAMPLE_STEPS = 200

BATCH_SIZE  = 256
NUM_WORKERS = 4
EPOCHS      = 200
LR              = 1e-4
LR_WARMUP_EPOCHS = 5
LR_MIN           = 1e-6
EMA_DECAY        = 0.999
GRAD_CLIP   = 1.0

CHANNELS      = [64, 128, 256, 512]
TIME_EMB_DIM  = 128
DROPOUT       = 0.1

SAMPLE_EVERY       = 5
NUM_SAMPLES        = 64
SAMPLE_STEPS_QUICK = 50

IMG_SIZE     = 28
IMG_CHANNELS = 1

WANDB_PROJECT = "crn-diffusion-mnist"
WANDB_ENTITY  = None
WANDB_MODE    = "online"
WANDB_KEY     = "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"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR   = os.path.join(SCRIPT_DIR, "checkpoints")
SAMPLE_DIR = os.path.join(SCRIPT_DIR, "samples")

# ---------------------------------------------------------------------------
# CRN math (直接来自 model.py，逻辑零改动)
# ---------------------------------------------------------------------------

def _as_time_column(t: Tensor, like: Tensor) -> Tensor:
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=like.device, dtype=like.dtype)
    t = t.to(device=like.device, dtype=like.dtype)
    if t.ndim == 0:
        t = t.expand(like.shape[0])
    while t.ndim < like.ndim:
        t = t.unsqueeze(-1)
    return t


@torch.no_grad()
def quantize_to_counts(x: Tensor, v0: float = V0) -> Tensor:
    n = torch.round((v0 * x).clamp_min(0.0))
    return n / v0


@torch.no_grad()
def forward_tau_leap(
    x0: Tensor,
    s: Tensor,
    v0: float = V0,
    n_steps: int = N_TAU_STEPS,
) -> Tensor:
    """
    n_{t+dt} = Binomial(n_t, exp(-dt)) + Poisson(v0 * (1 - exp(-dt)))
    """
    x0 = quantize_to_counts(x0, v0=v0)
    n = torch.round((v0 * x0).clamp_min(0.0))

    s_col = _as_time_column(s, x0).expand_as(x0)
    dt = s_col / float(n_steps)

    for _ in range(n_steps):
        survival_prob = torch.exp(-dt).clamp(0.0, 1.0).expand_as(n)
        birth_mean    = (v0 * (1.0 - torch.exp(-dt))).clamp_min(0.0).expand_as(n)

        survivors = torch.distributions.Binomial(
            total_count=n, probs=survival_prob
        ).sample()
        births = torch.poisson(birth_mean)
        n = (survivors + births).clamp_min(0.0)

    return n / v0


def conditional_momentum(
    x0: Tensor,
    xs: Tensor,
    s: Tensor,
    max_iter: int = 80,
    tol: float = 1e-7,
    delta_floor: float = 1e-6,
    p_clip: float = TARGET_CLIP,
) -> Tensor:
    """
    精确 HJ 条件动量，二分法求解特征线端点方程:
        xs/(1+δ) - a*x0/(1+a*δ) - (1-a) = 0,  a = exp(-s)
    p_t = log(1 + δ)
    """
    x0, xs = torch.broadcast_tensors(x0, xs)
    s = _as_time_column(s, xs).expand_as(xs)

    a  = torch.exp(-s)
    mu = 1.0 + (x0 - 1.0) * a

    def residual(delta: Tensor) -> Tensor:
        denom_t = (1.0 + delta).clamp_min(delta_floor)
        denom_0 = (1.0 + a * delta).clamp_min(delta_floor)
        return xs / denom_t - a * x0 / denom_0 - (1.0 - a)

    g0         = xs - mu
    near_typical = g0.abs() < tol
    delta      = torch.zeros_like(xs)

    # Case 1: xs > typical path → δ > 0
    pos_mask = g0 > tol
    if pos_mask.any():
        l = torch.zeros_like(xs)
        r = torch.ones_like(xs)
        f_r = residual(r)
        for _ in range(80):
            need = pos_mask & (f_r > 0.0)
            if not need.any():
                break
            r   = torch.where(need, r * 2.0, r)
            f_r = residual(r)
        for _ in range(max_iter):
            mid   = 0.5 * (l + r)
            f_mid = residual(mid)
            go_r  = f_mid > 0.0
            l = torch.where(pos_mask & go_r,  mid, l)
            r = torch.where(pos_mask & ~go_r, mid, r)
            if torch.max((r[pos_mask] - l[pos_mask]).abs()).item() < tol:
                break
        delta = torch.where(pos_mask, 0.5 * (l + r), delta)

    # Case 2: xs < typical path → -1 < δ < 0
    neg_mask = g0 < -tol
    if neg_mask.any():
        l   = torch.full_like(xs, -1.0 + delta_floor)
        r   = torch.zeros_like(xs)
        f_l = residual(l)
        has_bracket = neg_mask & (f_l > 0.0)
        for _ in range(max_iter):
            mid   = 0.5 * (l + r)
            f_mid = residual(mid)
            go_r  = f_mid > 0.0
            l = torch.where(has_bracket & go_r,  mid, l)
            r = torch.where(has_bracket & ~go_r, mid, r)
            if has_bracket.any():
                if torch.max((r[has_bracket] - l[has_bracket]).abs()).item() < tol:
                    break
        delta_neg = torch.where(has_bracket, 0.5 * (l + r), l)
        delta = torch.where(neg_mask, delta_neg, delta)

    delta = torch.where(near_typical, torch.zeros_like(delta), delta)

    safe_floor    = max(delta_floor, 100.0 * torch.finfo(xs.dtype).eps)
    one_plus_delta = (1.0 + delta).clamp_min(safe_floor)
    p = torch.log(one_plus_delta)
    p = torch.nan_to_num(p, nan=0.0, posinf=p_clip, neginf=-p_clip)
    return p.clamp(-p_clip, p_clip)


@torch.no_grad()
def sample_reverse(
    model: nn.Module,
    shape: Tuple[int, ...],
    v0: float = V0,
    T_val: float = T,
    n_steps: int = N_SAMPLE_STEPS,
    p_clip: float = P_CLIP,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    反向跳跃采样:
        birth ~ Pois(v0 * x * exp(-p) * tau)
        death ~ Pois(v0 * exp(p) * tau)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    n   = torch.poisson(torch.full(shape, float(v0), device=device))
    tau = T_val / float(n_steps)

    for i in reversed(range(n_steps)):
        s_val = (i + 1) * tau
        s     = torch.full((shape[0],), s_val, device=device)
        x     = n / v0

        with autocast():
            p = model(x, s)
        p = torch.nan_to_num(p, nan=0.0, posinf=p_clip, neginf=-p_clip).clamp(-p_clip, p_clip)

        birth_rate = (v0 * x * torch.exp(-p) * tau).clamp_min(0.0)
        death_rate = (v0 * torch.exp(p) * tau).clamp_min(0.0)

        births = torch.poisson(birth_rate)
        deaths = torch.poisson(death_rate)
        deaths = torch.minimum(deaths, n + births)
        n      = (n + births - deaths).clamp_min(0.0)

    return n / v0

# ---------------------------------------------------------------------------
# U-Net (来自 CRN-based-Diffusion-Models，网络结构零改动)
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = TIME_EMB_DIM):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        emb  = math.log(10000) / (half - 1)
        emb  = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb  = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int = TIME_EMB_DIM):
        super().__init__()
        self.norm1    = nn.GroupNorm(8, in_ch)
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.norm2    = nn.GroupNorm(8, out_ch)
        self.drop     = nn.Dropout(DROPOUT)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm  = nn.GroupNorm(8, channels)
        self.qkv   = nn.Linear(channels, channels * 3)
        self.proj  = nn.Linear(channels, channels)
        self.scale = channels ** -0.5

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        h    = self.norm(x).permute(0, 2, 3, 1).reshape(B, H * W, C)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * self.scale, dim=-1)
        h    = self.proj(torch.bmm(attn, v)).reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x + h


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """
    输入: (B, 1, 28, 28) + t → 输出: (B, 1, 28, 28) 预测边际 HJ 动量
    内部 pad 到 32×32 再 crop 回来。
    """

    def __init__(self):
        super().__init__()
        chs = CHANNELS  # [64, 128, 256, 512]

        self.time_embed = SinusoidalTimeEmbedding(TIME_EMB_DIM)
        self.time_mlp   = nn.Sequential(
            nn.Linear(TIME_EMB_DIM, TIME_EMB_DIM * 4),
            nn.SiLU(),
            nn.Linear(TIME_EMB_DIM * 4, TIME_EMB_DIM),
        )

        self.input_conv = nn.Conv2d(IMG_CHANNELS, chs[0], 3, padding=1)

        # Encoder
        self.down1 = nn.ModuleList([ResBlock(chs[0], chs[0]), ResBlock(chs[0], chs[0])])
        self.ds1   = Downsample(chs[0])
        self.down2 = nn.ModuleList([ResBlock(chs[0], chs[1]), ResBlock(chs[1], chs[1])])
        self.ds2   = Downsample(chs[1])
        self.down3 = nn.ModuleList([ResBlock(chs[1], chs[2]), ResBlock(chs[2], chs[2])])
        self.ds3   = Downsample(chs[2])
        self.down4 = nn.ModuleList([ResBlock(chs[2], chs[3]), ResBlock(chs[3], chs[3])])

        # Bottleneck
        self.mid = nn.ModuleList([
            ResBlock(chs[3], chs[3]),
            AttentionBlock(chs[3]),
            ResBlock(chs[3], chs[3]),
        ])

        # Decoder
        self.us3   = Upsample(chs[3])
        self.up3   = nn.ModuleList([ResBlock(chs[3] + chs[2], chs[2]), ResBlock(chs[2], chs[2])])
        self.us2   = Upsample(chs[2])
        self.up2   = nn.ModuleList([ResBlock(chs[2] + chs[1], chs[1]), ResBlock(chs[1], chs[1])])
        self.us1   = Upsample(chs[1])
        self.up1   = nn.ModuleList([ResBlock(chs[1] + chs[0], chs[0]), ResBlock(chs[0], chs[0])])

        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, chs[0]),
            nn.SiLU(),
            nn.Conv2d(chs[0], IMG_CHANNELS, 3, padding=1),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = F.pad(x, (2, 2, 2, 2))          # 28×28 → 32×32
        t_emb = self.time_mlp(self.time_embed(t))
        x = self.input_conv(x)

        for blk in self.down1: x = blk(x, t_emb)
        x1 = x;  x = self.ds1(x)

        for blk in self.down2: x = blk(x, t_emb)
        x2 = x;  x = self.ds2(x)

        for blk in self.down3: x = blk(x, t_emb)
        x3 = x;  x = self.ds3(x)

        for blk in self.down4: x = blk(x, t_emb)

        for blk in self.mid:
            x = blk(x) if isinstance(blk, AttentionBlock) else blk(x, t_emb)

        x = self.us3(x);  x = torch.cat([x, x3], dim=1)
        for blk in self.up3: x = blk(x, t_emb)

        x = self.us2(x);  x = torch.cat([x, x2], dim=1)
        for blk in self.up2: x = blk(x, t_emb)

        x = self.us1(x);  x = torch.cat([x, x1], dim=1)
        for blk in self.up1: x = blk(x, t_emb)

        x = self.output_conv(x)
        return x[:, :, 2:2 + IMG_SIZE, 2:2 + IMG_SIZE]   # crop 回 28×28


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.decay   = decay
        self.shadow  = {k: v.clone().detach() for k, v in model.named_parameters()}
        self._backup: dict = {}

    def update(self, model: nn.Module) -> None:
        for k, v in model.named_parameters():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.detach()

    def apply_shadow(self, model: nn.Module) -> None:
        self._backup = {k: v.clone().detach() for k, v in model.named_parameters()}
        for k, v in model.named_parameters():
            v.data.copy_(self.shadow[k])

    def restore(self, model: nn.Module) -> None:
        for k, v in model.named_parameters():
            v.data.copy_(self._backup[k])


# ---------------------------------------------------------------------------
# LR Scheduler: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def build_scheduler(optimizer: optim.Optimizer, total_epochs: int) -> optim.lr_scheduler.LambdaLR:
    def lr_lambda(epoch: int) -> float:
        # epoch is 0-indexed here (LambdaLR calls with 0, 1, 2, ...)
        if epoch < LR_WARMUP_EPOCHS:
            return (epoch + 1) / LR_WARMUP_EPOCHS
        progress = (epoch - LR_WARMUP_EPOCHS) / max(total_epochs - LR_WARMUP_EPOCHS, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return LR_MIN / LR + (1.0 - LR_MIN / LR) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_dataloader(data_root: str = "./data", train: bool = True) -> DataLoader:
    ds = datasets.MNIST(data_root, train=train, download=True,
                        transform=transforms.ToTensor())
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=train,
                      num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    ema: EMA,
    device: torch.device,
    epoch: int,
    run=None,
) -> float:
    model.train()
    total_loss   = 0.0
    skipped      = 0
    global_offset = (epoch - 1) * len(loader)
    epoch_start  = time.time()

    for step, (imgs, _) in enumerate(loader):
        x0 = imgs.float().to(device)

        # 时间采样: Uniform(s_eps, T)，与 model.py 一致
        s = torch.empty(x0.shape[0], device=device).uniform_(S_EPS, T)

        with torch.no_grad():
            xs     = forward_tau_leap(x0, s, v0=V0, n_steps=N_TAU_STEPS)
            target = conditional_momentum(x0, xs, s, p_clip=TARGET_CLIP)

        if not torch.isfinite(target).all():
            skipped += 1
            continue

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            pred = model(xs, s)
            loss = F.mse_loss(pred, target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        loss_val    = loss.item()
        total_loss += loss_val

        if run is not None and step % 50 == 0:
            run.log({
                "train/loss_step": loss_val,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/clip_frac": (target.abs() >= TARGET_CLIP - 1e-6).float().mean().item(),
                "train/zero_frac": (xs <= 0.0).float().mean().item(),
            }, step=global_offset + step)

    n       = len(loader) - skipped
    avg     = total_loss / max(n, 1)
    elapsed = time.time() - epoch_start
    global_step = global_offset + len(loader)
    print(f"Epoch {epoch:03d} | loss {avg:.6f} | time {elapsed:.1f}s | skipped {skipped}")

    if run is not None:
        run.log({"epoch/loss": avg, "epoch/skipped": skipped, "epoch/time": elapsed, "epoch": epoch}, step=global_step)

    return avg, global_step

# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------

def sample_images(
    model: nn.Module,
    ema: EMA,
    epoch: int,
    steps: int,
    device: torch.device,
    tag: str = "",
    run=None,
    global_step: int = 0,
) -> str:
    ema.apply_shadow(model)
    samples = sample_reverse(
        model,
        shape=(NUM_SAMPLES, IMG_CHANNELS, IMG_SIZE, IMG_SIZE),
        v0=V0, T_val=T, n_steps=steps, p_clip=P_CLIP, device=device,
    )
    ema.restore(model)

    os.makedirs(SAMPLE_DIR, exist_ok=True)
    fname = os.path.join(SAMPLE_DIR, f"samples_epoch{epoch:03d}{tag}.png")
    save_image(samples.clamp(0.0, 1.0), fname, nrow=8, normalize=False)
    print(f"Saved samples -> {fname}")

    if run is not None:
        run.log({"samples": wandb.Image(fname, caption=f"epoch {epoch}{tag}")}, step=global_step)

    return fname

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       choices=["train", "sample", "all"], default="all")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--epochs",     type=int, default=EPOCHS)
    parser.add_argument("--steps",      type=int, default=N_SAMPLE_STEPS)
    parser.add_argument("--data-root",  type=str, default="./data")
    parser.add_argument("--wandb-mode", type=str, default=WANDB_MODE,
                        choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    model = UNet().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    ema = EMA(model)

    start_epoch = 1
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema.shadow = {k: v.to(device) for k, v in ckpt["ema_shadow"].items()}
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from {args.checkpoint} (epoch {start_epoch - 1})")

    if args.mode == "sample":
        if args.checkpoint is None:
            raise ValueError("--checkpoint required for sample mode")
        sample_images(model, ema, epoch=start_epoch - 1,
                      steps=args.steps, device=device)
        return

    # wandb
    wandb.login(key=WANDB_KEY)
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        mode=args.wandb_mode,
        config={
            "V0": V0, "T": T, "S_EPS": S_EPS,
            "N_TAU_STEPS": N_TAU_STEPS, "TARGET_CLIP": TARGET_CLIP,
            "BATCH_SIZE": BATCH_SIZE, "LR": LR,
            "EMA_DECAY": EMA_DECAY, "EPOCHS": args.epochs,
            "GRAD_CLIP": GRAD_CLIP, "LR_WARMUP_EPOCHS": LR_WARMUP_EPOCHS, "LR_MIN": LR_MIN,
            "CHANNELS": CHANNELS,
            "TIME_EMB_DIM": TIME_EMB_DIM, "DROPOUT": DROPOUT,
        },
    )
    print(f"wandb run: {run.url}")

    loader    = get_dataloader(data_root=args.data_root, train=True)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = build_scheduler(optimizer, args.epochs)
    scaler    = GradScaler()

    global_step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        _, global_step = train_epoch(model, loader, optimizer, scaler, ema, device, epoch, run=run)
        scheduler.step()

        if run is not None:
            run.log({"train/lr": scheduler.get_last_lr()[0]}, step=global_step)

        if epoch % SAMPLE_EVERY == 0 or epoch == args.epochs:
            sample_images(model, ema, epoch=epoch,
                          steps=SAMPLE_STEPS_QUICK, device=device, run=run, global_step=global_step)

            ckpt_path = os.path.join(CKPT_DIR, f"ckpt_epoch{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "ema_shadow": ema.shadow,
                "optimizer": optimizer.state_dict(),
            }, ckpt_path)
            print(f"Checkpoint saved -> {ckpt_path}")

    # 训练结束后跑完整步数采样
    sample_images(model, ema, epoch=args.epochs,
                  steps=N_SAMPLE_STEPS, device=device, tag="_final", run=run, global_step=global_step)
    run.finish()


if __name__ == "__main__":
    main()
