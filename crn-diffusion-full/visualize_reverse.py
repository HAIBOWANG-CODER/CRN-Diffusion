"""
可视化 reverse process 的每一步，生成类似参考图的网格图。
用法:
    python visualize_reverse.py --checkpoint checkpoints/ckpt_epoch0200.pt
    python visualize_reverse.py --checkpoint checkpoints/ckpt_epoch0200.pt --n_samples 8 --n_show 10
"""

import argparse
import os
import math
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision.utils import make_grid, save_image

# 复用主文件里的模型和参数
import sys
sys.path.insert(0, os.path.dirname(__file__))
from crn_diffusion_full import (
    UNet, EMA,
    V0, T, P_CLIP, IMG_SIZE, IMG_CHANNELS,
    N_SAMPLE_STEPS,
)


@torch.no_grad()
def reverse_sample_with_trajectory(
    model,
    n_samples: int,
    steps: int,
    device: torch.device,
    n_show: int = 10,
):
    """
    反向采样，同时记录 n_show 个均匀分布的中间步骤。

    返回:
        trajectory: list of Tensor (B, C, H, W)，长度 = n_show，
                    第一个是纯噪声 x_T，最后一个是最终生成 x_0
    """
    model.eval()
    tau = T / float(steps)

    # 从平稳先验采样 x_T
    n = torch.poisson(
        torch.full((n_samples, IMG_CHANNELS, IMG_SIZE, IMG_SIZE), float(V0), device=device)
    )

    # 选取要记录的步骤索引（均匀分布，包含首尾）
    record_at = set(
        int(round(i)) for i in torch.linspace(steps - 1, 0, n_show).tolist()
    )

    trajectory = [(n / V0).clamp(0, 1)]  # 记录 x_T

    for i in reversed(range(steps)):
        s_val = (i + 1) * tau
        s = torch.full((n_samples,), s_val, device=device)
        x = n / V0

        with autocast():
            p = model(x, s)
        p = torch.nan_to_num(p, nan=0.0, posinf=P_CLIP, neginf=-P_CLIP).clamp(-P_CLIP, P_CLIP)

        birth_rate = (V0 * x * torch.exp(-p) * tau).clamp_min(0.0)
        death_rate = (V0 * torch.exp(p) * tau).clamp_min(0.0)

        births = torch.poisson(birth_rate)
        deaths = torch.poisson(death_rate)
        deaths = torch.minimum(deaths, n + births)
        n = (n + births - deaths).clamp_min(0.0)

        if i in record_at:
            trajectory.append((n / V0).clamp(0, 1))

    # 确保最后一帧在列表中
    if len(trajectory) < n_show + 1:
        trajectory.append((n / V0).clamp(0, 1))

    return trajectory


def make_trajectory_grid(trajectory, n_samples, nrow=None):
    """
    将轨迹拼成网格图。
    行 = 每个样本，列 = 时间步（从左噪声到右生成图）
    """
    n_steps = len(trajectory)
    if nrow is None:
        nrow = n_steps  # 每行放所有时间步

    # trajectory[t] shape: (B, C, H, W)
    # 按列排列：先所有样本在 step0，再所有样本在 step1，...
    # 改为按行排列：每行是一个样本的完整轨迹
    frames = []
    for b in range(n_samples):
        for t in range(n_steps):
            frames.append(trajectory[t][b])  # (C, H, W)

    grid = make_grid(frames, nrow=n_steps, padding=2, normalize=False)
    return grid


def main():
    parser = argparse.ArgumentParser(description="Visualize CRN reverse process")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint (.pt file)")
    parser.add_argument("--n_samples", type=int, default=8,
                        help="Number of samples (rows in the grid)")
    parser.add_argument("--n_show", type=int, default=10,
                        help="Number of timesteps to show (columns in the grid)")
    parser.add_argument("--steps", type=int, default=N_SAMPLE_STEPS,
                        help="Total reverse sampling steps")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (default: reverse_trajectory.png)")
    parser.add_argument("--use_ema", action="store_true", default=True,
                        help="Use EMA weights (default: True)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载模型
    model = UNet().to(device)
    ema = EMA(model)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    if args.use_ema and "ema_shadow" in ckpt:
        ema.shadow = {k: v.to(device) for k, v in ckpt["ema_shadow"].items()}
        ema.apply_shadow(model)
        print("Using EMA weights")

    model.eval()

    # 采样轨迹
    print(f"Sampling {args.n_samples} trajectories with {args.steps} steps, showing {args.n_show} frames...")
    trajectory = reverse_sample_with_trajectory(
        model,
        n_samples=args.n_samples,
        steps=args.steps,
        device=device,
        n_show=args.n_show,
    )
    print(f"Trajectory frames: {len(trajectory)}")

    # 生成网格图
    grid = make_trajectory_grid(trajectory, n_samples=args.n_samples)

    # 保存
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = args.output or os.path.join(script_dir, "reverse_trajectory.png")
    save_image(grid, out_path)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
