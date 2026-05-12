"""
评估模块：FID + 生图可视化 + wandb 记录。
"""
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from . import config
from .model import UNet, EMA
from .forward import CRNForwardProcess
from torchvision.utils import make_grid, save_image


def compute_fid_from_dirs(real_dir, gen_dir):
    """
    用 torch-fidelity 计算 FID。
    需要两个目录中各自保存为 .png 文件（格式：整数 0-255，28x28）。

    Args:
        real_dir: 真实图像目录
        gen_dir:  生成图像目录

    Returns:
        fid_score: float
    """
    try:
        from torch_fidelity import calculate_metrics
        metrics = calculate_metrics(
            input1=gen_dir,
            input2=real_dir,
            cuda=True,
            fid=True,
            isc=False,
            kid=False,
        )
        return metrics["frechet_inception_distance"]
    except ImportError:
        print("torch-fidelity not installed. Install with: pip install torch-fidelity")
        return None


def save_generated_images(samples, n, save_dir):
    """将生成样本逐张保存为 RGB .png 用于 FID 计算。"""
    sd = Path(save_dir)
    sd.mkdir(parents=True, exist_ok=True)

    for i in range(min(n, len(samples))):
        img = samples[i].squeeze().cpu().numpy()
        img = (img * 255).clip(0, 255).astype("uint8")
        Image.fromarray(img, mode="L").convert("RGB").save(sd / f"gen_{i:05d}.png")


def visualize_comparison(real_samples, gen_samples, save_path):
    """真实 vs 生成 像素分布对比图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real_flat = real_samples.flatten().cpu().numpy()
    gen_flat  = gen_samples.flatten().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(real_flat, bins=50, alpha=0.7, color="blue", density=True)
    axes[0].set_title("Real MNIST Pixel Distribution")
    axes[0].set_xlabel("Pixel value (normalized)")
    axes[0].set_ylabel("Density")

    axes[1].hist(gen_flat, bins=50, alpha=0.7, color="orange", density=True)
    axes[1].set_title("Generated Pixel Distribution")
    axes[1].set_xlabel("Pixel value (normalized)")
    axes[1].set_ylabel("Density")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved distribution comparison to {save_path}")


def _reverse_step(x, s, model, v0, device):
    """Single DDPM-style reverse step at time s."""
    from torch.cuda.amp import autocast
    dt_step = config.Config.T / config.Config.SAMPLE_STEPS
    t_prev = (s - dt_step).clamp(min=0.0)
    with torch.no_grad():
        t_batch = s.expand(x.size(0))
        with autocast():
            p = model(x, t_batch)
        et = torch.exp(-s)
        a = p * (1.0 - et) / v0
        x0_pred = ((a + x - 1.0 + et) / (et * (1.0 - a)).clamp(min=1e-6)).clamp(0.0, 1.0)
        if t_prev < 1e-6:
            return x0_pred
        et_prev = torch.exp(-t_prev)
        surviving_rate = (v0 * x0_pred * et_prev).clamp(min=0.0)
        new_rate = (v0 * (1.0 - et_prev)).clamp(min=0.0)
        return (torch.poisson(surviving_rate) + torch.poisson(new_rate.expand_as(x0_pred))) / v0


def visualize_intermediate_steps(model, crn, device, n=8, save_path=None):
    """可视化反向采样中间过程：从纯噪声到数字。"""
    model.eval()
    steps = config.Config.SAMPLE_STEPS
    v0 = config.Config.V0

    xT = torch.poisson(v0 * torch.ones(n, 1, config.Config.IMG_SIZE,
                                        config.Config.IMG_SIZE, device=device))
    xT = xT.float() / v0
    x = xT.clone()

    checkpoints = {0, int(steps * 0.1), int(steps * 0.3), int(steps * 0.5),
                   int(steps * 0.7), int(steps * 0.9), steps - 1}
    saved_frames = []

    dt_step = config.Config.T / steps
    time_grid = torch.linspace(config.Config.T, dt_step, steps, device=device)

    for i, s in enumerate(time_grid):
        x = _reverse_step(x, s, model, v0, device)
        if i in checkpoints:
            saved_frames.append(x.clone())

    row = torch.cat(saved_frames, dim=0)
    grid = make_grid(row, nrow=n, padding=2, normalize=True)
    path = save_path or config.GENERATED_DIR / "intermediate_steps.png"
    save_image(grid, path)
    print(f"Saved intermediate steps to {path}")
    return path


def reverse_sample(model, xT, device, crn, steps=200):
    """同 train.py 中的反向采样（DDPM 风格 x0-预测）。"""
    model.eval()
    v0 = config.Config.V0
    dt_step = config.Config.T / steps
    x = xT.clone()
    time_grid = torch.linspace(config.Config.T, dt_step, steps, device=device)
    for s in time_grid:
        x = _reverse_step(x, s, model, v0, device)
    return x


def evaluate(ckpt_path, n_samples=1000, n_display=64, wandb_run=None):
    """
    主评估流程：
      1. 生成 n_samples 张图像
      2. 保存逐张 .png 用于 FID
      3. 计算 FID
      4. 可视化分布对比
      5. 保存中间过程
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)
    ema = EMA(model, decay=config.Config.EMA_DECAY)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if "ema_shadow" in ckpt:
        ema.shadow = ckpt["ema_shadow"]
        ema.apply_shadow()
    print(f"Loaded checkpoint: {ckpt_path}, epoch: {ckpt.get('epoch', '?')}")

    crn = CRNForwardProcess(device=device)

    print(f"Generating {n_samples} samples...")
    gen_samples = []
    batch_size = 100
    for i in range(0, n_samples, batch_size):
        n_cur = min(batch_size, n_samples - i)
        xT = torch.poisson(config.Config.V0 * torch.ones(n_cur, 1, config.Config.IMG_SIZE,
                                                           config.Config.IMG_SIZE, device=device))
        xT = xT.float() / config.Config.V0
        samples = reverse_sample(model, xT, device, crn, steps=config.Config.SAMPLE_STEPS)
        gen_samples.append(samples.cpu())
        print(f"  [{i + n_cur}/{n_samples}]")
    gen_samples = torch.cat(gen_samples, dim=0)

    gen_dir = config.GENERATED_DIR / "fid_input"
    real_dir = config.REAL_MNIST_DIR / "fid_real"

    print("Saving generated images for FID...")
    save_generated_images(gen_samples, n_samples, gen_dir)

    real_samples_path = config.REAL_MNIST_DIR / "real_samples.pt"
    if not real_dir.exists() or not list(real_dir.glob("real_*.png")):
        from .data import save_real_mnist_samples
        print("Real MNIST PNG samples not found, generating...")
        save_real_mnist_samples(n=1000)

    print("Computing FID...")
    fid = compute_fid_from_dirs(str(real_dir), str(gen_dir))
    print(f"FID: {fid:.4f}" if fid is not None else "FID: N/A (torch-fidelity not installed)")

    print("Generating visualization...")
    vis_path = config.GENERATED_DIR / "eval_visualization.png"
    if not real_samples_path.exists():
        from .data import save_real_mnist_samples
        save_real_mnist_samples(n=1000)
    visualize_comparison(
        torch.load(real_samples_path),
        gen_samples,
        vis_path
    )

    int_path = visualize_intermediate_steps(model, crn, device, n=min(n_display, 8))

    disp_samples = gen_samples[:n_display]
    disp_grid = make_grid(disp_samples, nrow=int(n_display ** 0.5),
                           padding=2, normalize=True, value_range=(0, 1))
    disp_path = config.GENERATED_DIR / "eval_generated_grid.png"
    save_image(disp_grid, disp_path)

    results = {
        "fid": fid,
        "vis_path": str(vis_path),
        "intermediate_path": str(int_path),
        "grid_path": str(disp_path),
    }

    if wandb_run is not None:
        import wandb
        wandb_run.log({
            "eval/fid": fid if fid is not None else -1,
            "eval/generated_grid": wandb.Image(str(disp_path)),
            "eval/distribution": wandb.Image(str(vis_path)),
            "eval/intermediate_steps": wandb.Image(str(int_path)),
        })

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--n", type=int, default=1000, help="Number of samples for FID")
    parser.add_argument("--n_display", type=int, default=64)
    args = parser.parse_args()

    results = evaluate(args.ckpt, n_samples=args.n, n_display=args.n_display)
    print(f"\nResults: {results}")
