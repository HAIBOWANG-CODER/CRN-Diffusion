"""
训练循环：CRN Diffusion Model。

训练流程：
    1. x0 ~ MNIST
    2. t ~ Uniform(0, T)
    3. x_t = ForwardCRN(x0, t)  (tau-leaping)
    4. p_target = HJSolver.conditional_momentum(x0, x_t, t)
    5. p_pred   = U-Net(x_t, t)
    6. Loss = MSE(p_pred, p_target)
    7. Backprop + Adam + EMA

日志（wandb）：
    - loss 曲线（逐 step）
    - 每 N epoch 可视化生图
"""
import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from . import config
from .data import get_dataloader
from .forward import CRNForwardProcess
from .hj_solver import conditional_momentum
from .model import UNet, EMA


def train_epoch(model, ema, optimizer, scaler, crn, device, epoch, wandb_run=None):
    model.train()
    total_loss = 0.0
    n_batches = 0
    epoch_start = time.time()

    loader = get_dataloader(train=True, batch_size=config.Config.BATCH_SIZE, shuffle=True)

    for batch_idx, (x0, _) in enumerate(loader):
        x0 = x0.to(device)

        B = x0.size(0)
        # t 下界设为 0.01 避免 t≈0 时 Newton 法数值不稳定
        t = torch.rand(B, device=device) * (config.Config.T - 0.01) + 0.01

        with torch.no_grad():
            x_t = crn(x0, t)
            pt_target = conditional_momentum(x0, x_t, t)
            # 跳过含 NaN/Inf 的 batch，避免污染模型权重
            if not torch.isfinite(pt_target).all():
                continue

        optimizer.zero_grad()

        with autocast():
            pt_pred = model(x_t, t)
            loss = nn.MSELoss()(pt_pred, pt_target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.Config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        ema.update()

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            continue
        total_loss += loss_val
        n_batches += 1

        if wandb_run is not None and batch_idx % 50 == 0:
            wandb_run.log({
                "train/loss": loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/step": epoch * len(loader) + batch_idx,
            })

    avg_loss = total_loss / n_batches
    elapsed = time.time() - epoch_start
    print(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f} | Time: {elapsed:.1f}s")

    if wandb_run is not None:
        wandb_run.log({
            "epoch/loss": avg_loss,
            "epoch/time": elapsed,
            "epoch": epoch,
        })

    return avg_loss


def sample_images(model, ema_shadow_model, crn, device, n=64, epoch=0, loss=None):
    """从 EMA 模型采样，保存带 epoch/loss 标注的生图 grid。"""
    model.eval()
    with torch.no_grad():
        xT = torch.poisson(config.Config.V0 * torch.ones(
            n, 1, config.Config.IMG_SIZE, config.Config.IMG_SIZE, device=device
        )).float() / config.Config.V0

    x = reverse_sample(model, xT, device, crn, steps=config.Config.SAMPLE_STEPS_QUICK)

    from torchvision.utils import save_image, make_grid
    from PIL import Image, ImageDraw
    import numpy as np

    grid = make_grid(x, nrow=int(n ** 0.5), padding=2, normalize=True)
    # Convert to PIL for annotation
    grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
    if grid_np.shape[2] == 1:
        pil_img = Image.fromarray(grid_np[:, :, 0], mode="L").convert("RGB")
    else:
        pil_img = Image.fromarray(grid_np)

    draw = ImageDraw.Draw(pil_img)
    label = f"Epoch {epoch}"
    if loss is not None:
        label += f"  Loss {loss:.4f}"
    draw.text((4, 4), label, fill=(255, 80, 80))

    save_path = config.SAMPLE_DIR / f"epoch{epoch:04d}.png"
    pil_img.save(str(save_path))
    return save_path


def reverse_sample(model, xT, device, crn, steps=200):
    """
    从 Poisson 先验 xT 反向采样到 x0。

    DDPM 风格的 x0-预测反向采样：
    给定模型输出 p = score = -(xt - μ) / σ²，其中
        μ = x0*e^{-t} + (1-e^{-t}), σ² = (1-e^{-t})*(x0*e^{-t}+1)/v0

    从 p 和 xt 线性求解 x0:
        x0 = [p*(1-et)/v0 + xt - 1 + et] / (et * [1 - p*(1-et)/v0])

    然后从 P(x_{t-dt} | x0) 精确采样下一步。
    """
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
            # Solve for x0 from the linear equation derived from the score:
            # x0 = [p*(1-et)/v0 + xt - 1 + et] / (et * [1 - p*(1-et)/v0])
            a = p * (1.0 - et) / v0
            numerator = a + x - 1.0 + et
            denominator = et * (1.0 - a)
            x0_pred = (numerator / denominator.clamp(min=1e-6)).clamp(min=0.0, max=1.0)

            if t_prev < 1e-6:
                x = x0_pred
            else:
                et_prev = torch.exp(-t_prev)
                surviving_rate = (v0 * x0_pred * et_prev).clamp(min=0.0)
                new_rate = (v0 * (1.0 - et_prev)).clamp(min=0.0)
                x = (torch.poisson(surviving_rate) + torch.poisson(new_rate.expand_as(x0_pred))) / v0

    return x


def main():
    import wandb
    wandb.login(key=config.Config.WANDB_KEY)

    wandb.init(
        project=config.Config.WANDB_PROJECT,
        entity=config.Config.WANDB_ENTITY,
        mode=config.Config.WANDB_MODE,
        config={
            "V0": config.Config.V0,
            "T":  config.Config.T,
            "DT": config.Config.DT,
            "BATCH_SIZE": config.Config.BATCH_SIZE,
            "LR": config.Config.LR,
            "EMA_DECAY": config.Config.EMA_DECAY,
            "EPOCHS": config.Config.EPOCHS,
            "IMG_SIZE": config.Config.IMG_SIZE,
        },
    )
    wandb_run = wandb.run

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = UNet().to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    ema = EMA(model, decay=config.Config.EMA_DECAY)
    optimizer = optim.Adam(model.parameters(), lr=config.Config.LR)
    scaler = GradScaler()
    crn = CRNForwardProcess(device=device)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(1, config.Config.EPOCHS + 1):
        loss = train_epoch(model, ema, optimizer, scaler, crn, device, epoch, wandb_run)

        if epoch % config.Config.SAMPLE_EVERY == 0:
            ema.apply_shadow()
            sample_path = sample_images(ema.model, None, crn, device,
                                        n=config.Config.NUM_SAMPLES, epoch=epoch, loss=loss)
            wandb_run.log({"sample/grid": wandb.Image(str(sample_path))}, step=epoch)
            ema.restore()

            checkpoint_path = config.CHECKPOINT_DIR / f"ckpt_epoch{epoch:04d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "ema_shadow": ema.shadow,
                "optimizer_state": optimizer.state_dict(),
                "loss": loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    wandb_run.finish()
    print("Training complete!")


if __name__ == "__main__":
    main()
