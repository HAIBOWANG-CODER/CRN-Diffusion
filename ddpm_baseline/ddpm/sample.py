"""
独立采样脚本：加载 checkpoint，生成 MNIST 图像。
用法：python run.py --mode sample [--ckpt path/to/ckpt.pt] [--n 64] [--steps 1000] [--ddim]
"""
import argparse
import glob
import torch
from pathlib import Path
from torchvision.utils import save_image, make_grid

from . import config
from .model import UNet, EMA
from .diffusion import GaussianDiffusion


def generate(ckpt_path=None, n=64, steps=None, use_ddim=False, output_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ckpt_path is None:
        ckpts = sorted(glob.glob(str(config.CHECKPOINT_DIR / "ckpt_*.pt")))
        if not ckpts:
            print("ERROR: no checkpoint found. Train first.")
            return
        ckpt_path = ckpts[-1]
        print(f"Auto-selected latest: {ckpt_path}")

    model = UNet().to(device)
    ema   = EMA(model, decay=config.Config.EMA_DECAY)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if "ema_shadow" in ckpt:
        ema.shadow = ckpt["ema_shadow"]
        ema.apply_shadow()
    print(f"Loaded: {ckpt_path}  epoch={ckpt.get('epoch', '?')}")

    diffusion = GaussianDiffusion(device=device)
    steps     = steps or (config.Config.SAMPLE_STEPS_QUICK if use_ddim else config.Config.SAMPLE_STEPS)
    shape     = (n, config.Config.IMG_CHANNELS, config.Config.IMG_SIZE, config.Config.IMG_SIZE)

    print(f"Generating {n} samples  steps={steps}  ddim={use_ddim} ...")
    samples = diffusion.sample(model, shape, steps=steps, use_ddim=use_ddim)
    samples = (samples.clamp(-1.0, 1.0) + 1.0) / 2.0

    odir = Path(output_dir) if output_dir else config.GENERATED_DIR
    odir.mkdir(parents=True, exist_ok=True)

    grid      = make_grid(samples, nrow=int(n ** 0.5), padding=2)
    tag       = "ddim" if use_ddim else "ddpm"
    save_path = odir / f"generated_{tag}_n{n}_steps{steps}.png"
    save_image(grid, save_path)
    print(f"Saved grid: {save_path}")

    pt_path = odir / f"samples_{tag}_n{n}_steps{steps}.pt"
    torch.save(samples, pt_path)
    print(f"Saved tensor: {pt_path}")
    return samples, save_path
