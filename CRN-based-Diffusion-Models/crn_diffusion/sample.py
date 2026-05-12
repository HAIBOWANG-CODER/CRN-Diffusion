"""
独立采样脚本：加载 checkpoint，从 Poisson 先验生成 MNIST 图像。
"""
import argparse
import torch
from pathlib import Path
from . import config
from .model import UNet, EMA
from .forward import CRNForwardProcess
from torchvision.utils import save_image, make_grid


def reverse_sample(model, xT, device, crn, steps=200):
    """
    从 Poisson 先验 xT 反向采样到 x0。DDPM 风格 x0-预测。
    """
    from torch.cuda.amp import autocast
    model.eval()
    dt_step = config.Config.T / steps
    v0 = config.Config.V0

    x = xT.clone()
    time_grid = torch.linspace(config.Config.T, dt_step, steps, device=device)

    for s in time_grid:
        t_prev = (s - dt_step).clamp(min=0.0)
        with torch.no_grad():
            t_batch = s.expand(x.size(0))
            with autocast():
                p = model(x, t_batch)

        with torch.no_grad():
            et = torch.exp(-s)
            a = p * (1.0 - et) / v0
            x0_pred = ((a + x - 1.0 + et) / (et * (1.0 - a)).clamp(min=1e-6)).clamp(0.0, 1.0)

            if t_prev < 1e-6:
                x = x0_pred
            else:
                et_prev = torch.exp(-t_prev)
                surviving_rate = (v0 * x0_pred * et_prev).clamp(min=0.0)
                new_rate = (v0 * (1.0 - et_prev)).clamp(min=0.0)
                x = (torch.poisson(surviving_rate) + torch.poisson(new_rate.expand_as(x0_pred))) / v0

    return x


def generate_samples(ckpt_path, n=64, steps=200, output_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)
    ema = EMA(model, decay=config.Config.EMA_DECAY)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if "ema_shadow" in ckpt:
        ema.shadow = ckpt["ema_shadow"]
        ema.apply_shadow()
    print(f"Loaded checkpoint: {ckpt_path}, epoch: {ckpt.get('epoch', 'N/A')}")

    crn = CRNForwardProcess(device=device)

    xT = torch.poisson(config.Config.V0 * torch.ones(n, 1, config.Config.IMG_SIZE,
                                                        config.Config.IMG_SIZE, device=device))
    xT = xT.float() / config.Config.V0

    print(f"Generating {n} samples from Poisson prior (v0={config.Config.V0})...")
    samples = reverse_sample(model, xT, device, crn, steps=steps)

    odir = Path(output_dir) if output_dir else config.GENERATED_DIR
    odir.mkdir(parents=True, exist_ok=True)

    grid = make_grid(samples, nrow=int(n ** 0.5), padding=2, normalize=True)
    save_path = odir / f"generated_n{n}_steps{steps}.png"
    save_image(grid, save_path)
    print(f"Saved grid to {save_path}")

    samples_path = odir / f"samples_n{n}_steps{steps}.pt"
    torch.save(samples, samples_path)
    print(f"Saved tensor to {samples_path}")

    return samples, save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument("--n", type=int, default=64, help="Number of samples")
    parser.add_argument("--steps", type=int, default=200, help="Sampling steps")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    if args.ckpt is None:
        import glob
        checkpoints = sorted(glob.glob(str(config.CHECKPOINT_DIR / "ckpt_*.pt")))
        if checkpoints:
            args.ckpt = checkpoints[-1]
            print(f"No checkpoint specified, using latest: {args.ckpt}")
        else:
            print("ERROR: No checkpoint found. Please train first or specify --ckpt.")
            exit(1)

    generate_samples(args.ckpt, args.n, args.steps, args.output)
