"""
评估模块：FID + 生图可视化 + 像素分布对比 + wandb 记录。
与 CRN 项目评估流程完全一致。
"""
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision.utils import make_grid, save_image

from . import config
from .model import UNet, EMA
from .diffusion import GaussianDiffusion


# ── FID ──────────────────────────────────────────────────────────────────────

def compute_fid_from_dirs(real_dir, gen_dir):
    try:
        from torch_fidelity import calculate_metrics
        metrics = calculate_metrics(
            input1=str(gen_dir),
            input2=str(real_dir),
            cuda=True,
            fid=True,
            isc=False,
            kid=False,
        )
        return metrics["frechet_inception_distance"]
    except ImportError:
        print("torch-fidelity not installed. pip install torch-fidelity")
        return None


def save_generated_images(samples, n, save_dir):
    """逐张保存 .png 用于 FID 计算。samples ∈ [0,1]。"""
    sd = Path(save_dir)
    sd.mkdir(parents=True, exist_ok=True)
    for i in range(min(n, len(samples))):
        img = samples[i].squeeze().cpu().numpy()
        img = (img * 255).clip(0, 255).astype("uint8")
        Image.fromarray(img, mode="L").save(sd / f"gen_{i:05d}.png")


def save_real_mnist_images(n, save_dir):
    """从 MNIST 训练集中随机取 n 张，保存为 .png 用于 FID 参考。"""
    from .data import get_dataloader
    sd = Path(save_dir)
    sd.mkdir(parents=True, exist_ok=True)
    # Use train split (always available); shuffle to get diverse digits
    loader = get_dataloader(train=True, batch_size=n, shuffle=True, drop_last=False)
    imgs, _ = next(iter(loader))          # imgs ∈ [-1, 1]
    imgs = imgs[:n]
    imgs = (imgs.clamp(-1, 1) + 1) / 2   # → [0, 1]
    for i in range(len(imgs)):
        img = imgs[i].squeeze().numpy()
        img = (img * 255).clip(0, 255).astype("uint8")
        Image.fromarray(img, mode="L").save(sd / f"real_{i:05d}.png")
    return imgs


# ── Visualizations ───────────────────────────────────────────────────────────

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
    axes[0].set_xlabel("Pixel value (normalized [0,1])")
    axes[0].set_ylabel("Density")

    axes[1].hist(gen_flat, bins=50, alpha=0.7, color="orange", density=True)
    axes[1].set_title("DDPM Generated Pixel Distribution")
    axes[1].set_xlabel("Pixel value (normalized [0,1])")
    axes[1].set_ylabel("Density")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved distribution comparison: {save_path}")


def visualize_intermediate_steps(model, diffusion, device, n=8, save_path=None):
    """
    可视化 DDIM 反向采样中间过程：从纯噪声到数字。
    保存各阶段 denoising 状态拼接成的 grid。
    """
    model.eval()
    steps = config.Config.SAMPLE_STEPS_QUICK
    shape = (n, config.Config.IMG_CHANNELS, config.Config.IMG_SIZE, config.Config.IMG_SIZE)

    # Build the full DDIM timestep list
    ts      = list(range(diffusion.T, 0, -diffusion.T // steps))[:steps]
    ts_prev = ts[1:] + [0]

    checkpoints = {
        0,
        int(len(ts) * 0.1),
        int(len(ts) * 0.3),
        int(len(ts) * 0.5),
        int(len(ts) * 0.7),
        int(len(ts) * 0.9),
        len(ts) - 1,
    }
    saved_frames = []

    x = torch.randn(shape, device=device)
    with torch.no_grad():
        for i, (t, t_prev) in enumerate(zip(ts, ts_prev)):
            t_prev_safe = max(t_prev, 1)
            x = diffusion.ddim_step(model, x, t, t_prev_safe)
            if t_prev == 0:
                B   = x.shape[0]
                t_b = torch.full((B,), 1, device=device, dtype=torch.long)
                eps = model(x, t_b)
                x   = diffusion._predict_x0(x, t_b, eps).clamp(-1.0, 1.0)
            if i in checkpoints:
                # Convert to [0,1] for display
                saved_frames.append(((x.clamp(-1, 1) + 1) / 2).clone())

    row  = torch.cat(saved_frames, dim=0)
    grid = make_grid(row, nrow=n, padding=2, normalize=False)
    path = Path(save_path) if save_path else config.GENERATED_DIR / "intermediate_steps.png"
    save_image(grid, path)
    print(f"Saved intermediate steps: {path}")
    return path


# ── Main evaluate ─────────────────────────────────────────────────────────────

def evaluate(ckpt_path, n_samples=1000, n_display=64, wandb_run=None):
    """
    主评估流程（与 CRN 项目对齐）：
      1. 生成 n_samples 张图像（DDIM 快速采样）
      2. 保存逐张 .png 用于 FID
      3. 保存真实 MNIST .png 用于 FID 参考
      4. 计算 FID
      5. 可视化像素分布对比
      6. 可视化中间 denoising 步骤
      7. 保存生成图 grid
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ──
    model = UNet().to(device)
    ema   = EMA(model, decay=config.Config.EMA_DECAY)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if "ema_shadow" in ckpt:
        ema.shadow = ckpt["ema_shadow"]
        ema.apply_shadow()
    print(f"Loaded: {ckpt_path}  epoch={ckpt.get('epoch', '?')}")

    diffusion = GaussianDiffusion(device=device)

    # ── Generate samples ──
    print(f"Generating {n_samples} samples (DDIM {config.Config.SAMPLE_STEPS_QUICK} steps)...")
    gen_samples = []
    batch = 100
    for i in range(0, n_samples, batch):
        n_cur  = min(batch, n_samples - i)
        shape  = (n_cur, config.Config.IMG_CHANNELS,
                  config.Config.IMG_SIZE, config.Config.IMG_SIZE)
        s = diffusion.sample(model, shape,
                             steps=config.Config.SAMPLE_STEPS_QUICK,
                             use_ddim=True)
        s = (s.clamp(-1, 1) + 1) / 2   # → [0, 1]
        gen_samples.append(s.cpu())
        print(f"  [{i + n_cur}/{n_samples}]")
    gen_samples = torch.cat(gen_samples, dim=0)

    # ── Save for FID ──
    gen_dir  = config.GENERATED_DIR / "fid_gen"
    real_dir = config.GENERATED_DIR / "fid_real"

    print("Saving generated images for FID...")
    save_generated_images(gen_samples, n_samples, gen_dir)

    print("Saving real MNIST images for FID...")
    real_samples = save_real_mnist_images(n_samples, real_dir)

    # ── FID ──
    print("Computing FID...")
    fid = compute_fid_from_dirs(real_dir, gen_dir)
    if fid is not None:
        print(f"FID: {fid:.4f}")
    else:
        print("FID: N/A")

    # ── Pixel distribution ──
    vis_path = config.GENERATED_DIR / "eval_distribution.png"
    # real_samples from save_real_mnist_images is already [0,1]
    visualize_comparison(real_samples[:n_samples], gen_samples, vis_path)

    # ── Intermediate steps ──
    int_path = visualize_intermediate_steps(
        model, diffusion, device,
        n=min(n_display, 8),
    )

    # ── Generated grid ──
    disp      = gen_samples[:n_display]
    grid      = make_grid(disp, nrow=int(n_display ** 0.5), padding=2, normalize=False)
    grid_path = config.GENERATED_DIR / "eval_generated_grid.png"
    save_image(grid, grid_path)
    print(f"Saved generated grid: {grid_path}")

    results = {
        "fid":               fid,
        "vis_path":          str(vis_path),
        "intermediate_path": str(int_path),
        "grid_path":         str(grid_path),
    }

    if wandb_run is not None:
        import wandb
        wandb_run.log({
            "eval/fid":                fid if fid is not None else -1,
            "eval/generated_grid":     wandb.Image(str(grid_path)),
            "eval/distribution":       wandb.Image(str(vis_path)),
            "eval/intermediate_steps": wandb.Image(str(int_path)),
        })

    return results
