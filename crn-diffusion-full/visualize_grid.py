"""
生成 16×16 大网格图，随机采样展示模型生成效果。

用法:
    python visualize_grid.py --checkpoint checkpoints/ckpt_epoch200.pt
    python visualize_grid.py --checkpoint checkpoints/ckpt_epoch200.pt --rows 16 --cols 16
"""

import argparse
import os
import sys
import torch
from torchvision.utils import make_grid, save_image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crn_diffusion_full import (
    UNet, EMA, sample_reverse,
    V0, T, P_CLIP, IMG_SIZE, IMG_CHANNELS, N_SAMPLE_STEPS,
)


def main():
    parser = argparse.ArgumentParser(description="Generate 16x16 sample grid (CRN-full)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--cols", type=int, default=16)
    parser.add_argument("--steps", type=int, default=N_SAMPLE_STEPS)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--use_ema", action="store_true", default=True)
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

    # 生成样本
    n_total = args.rows * args.cols
    print(f"Generating {n_total} samples ({args.rows}×{args.cols})...")
    with torch.no_grad():
        samples = sample_reverse(
            model,
            shape=(n_total, IMG_CHANNELS, IMG_SIZE, IMG_SIZE),
            v0=V0, T_val=T, n_steps=args.steps, p_clip=P_CLIP,
            device=device,
        ).clamp(0, 1)

    # 拼网格图
    grid = make_grid(samples.cpu(), nrow=args.cols, padding=2, normalize=False)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = args.output or os.path.join(script_dir, "sample_grid.png")
    save_image(grid, out_path)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
