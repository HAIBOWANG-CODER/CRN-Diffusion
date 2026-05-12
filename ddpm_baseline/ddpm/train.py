"""
训练循环：DDPM on MNIST。
"""
import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from PIL import Image, ImageDraw

from . import config
from .data import get_dataloader
from .diffusion import GaussianDiffusion
from .model import UNet, EMA


def train_epoch(model, ema, optimizer, scaler, diffusion, device, epoch, wandb_run=None):
    model.train()
    total_loss = 0.0
    n_batches  = 0
    t0 = time.time()

    loader = get_dataloader(train=True, batch_size=config.Config.BATCH_SIZE)

    for batch_idx, (x0, _) in enumerate(loader):
        x0 = x0.to(device)                                    # [-1, 1]
        B  = x0.size(0)

        # Sample random timesteps t ~ Uniform{1,...,T}
        t = torch.randint(1, config.Config.T + 1, (B,), device=device)

        noise = torch.randn_like(x0)
        with torch.no_grad():
            x_t, _ = diffusion.q_sample(x0, t, noise)

        optimizer.zero_grad()
        with autocast():
            eps_pred = model(x_t, t)
            loss = nn.MSELoss()(eps_pred, noise)

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
        n_batches  += 1

        if wandb_run is not None and batch_idx % 50 == 0:
            wandb_run.log({
                "train/loss": loss_val,
                "train/lr":   optimizer.param_groups[0]["lr"],
                "train/step": (epoch - 1) * len(loader) + batch_idx,
            })

    avg_loss = total_loss / max(n_batches, 1)
    elapsed  = time.time() - t0
    print(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f} | Time: {elapsed:.1f}s")

    if wandb_run is not None:
        wandb_run.log({"epoch/loss": avg_loss, "epoch/time": elapsed, "epoch": epoch})

    return avg_loss


def sample_images(model, diffusion, device, n=64, epoch=0, loss=None):
    """DDIM 快速采样 + PIL 标注，保存到 samples/epochNNNN.png。"""
    shape = (n, config.Config.IMG_CHANNELS, config.Config.IMG_SIZE, config.Config.IMG_SIZE)
    samples = diffusion.sample(model, shape,
                               steps=config.Config.SAMPLE_STEPS_QUICK,
                               use_ddim=True)

    # samples ∈ [-1, 1]  →  [0, 1] for display
    samples = (samples.clamp(-1.0, 1.0) + 1.0) / 2.0

    from torchvision.utils import make_grid
    grid = make_grid(samples, nrow=int(n ** 0.5), padding=2)

    grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
    if grid_np.shape[2] == 1:
        pil_img = Image.fromarray(grid_np[:, :, 0], mode="L").convert("RGB")
    else:
        pil_img = Image.fromarray(grid_np)

    draw  = ImageDraw.Draw(pil_img)
    label = f"Epoch {epoch}"
    if loss is not None:
        label += f"  Loss {loss:.4f}"
    draw.text((4, 4), label, fill=(255, 80, 80))

    save_path = config.SAMPLE_DIR / f"epoch{epoch:04d}.png"
    pil_img.save(str(save_path))
    return save_path


def main():
    import wandb
    wandb.login(key=config.Config.WANDB_KEY)
    wandb.init(
        project=config.Config.WANDB_PROJECT,
        entity=config.Config.WANDB_ENTITY,
        mode=config.Config.WANDB_MODE,
        config={
            "T":           config.Config.T,
            "BETA_START":  config.Config.BETA_START,
            "BETA_END":    config.Config.BETA_END,
            "BATCH_SIZE":  config.Config.BATCH_SIZE,
            "LR":          config.Config.LR,
            "EMA_DECAY":   config.Config.EMA_DECAY,
            "EPOCHS":      config.Config.EPOCHS,
            "IMG_SIZE":    config.Config.IMG_SIZE,
            "CHANNELS":    config.Config.CHANNELS,
        },
    )
    wandb_run = wandb.run

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model     = UNet().to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    ema       = EMA(model, decay=config.Config.EMA_DECAY)
    optimizer = optim.Adam(model.parameters(), lr=config.Config.LR)
    scaler    = GradScaler()
    diffusion = GaussianDiffusion(device=device)

    for epoch in range(1, config.Config.EPOCHS + 1):
        loss = train_epoch(model, ema, optimizer, scaler, diffusion, device, epoch, wandb_run)

        if epoch % config.Config.SAMPLE_EVERY == 0:
            ema.apply_shadow()
            sample_path = sample_images(model, diffusion, device,
                                        n=config.Config.NUM_SAMPLES,
                                        epoch=epoch, loss=loss)
            wandb_run.log({"sample/grid": wandb.Image(str(sample_path))}, step=epoch)
            ema.restore()

            ckpt_path = config.CHECKPOINT_DIR / f"ckpt_epoch{epoch:04d}.pt"
            torch.save({
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "ema_shadow":     ema.shadow,
                "optimizer_state": optimizer.state_dict(),
                "loss":           loss,
            }, ckpt_path)
            print(f"Checkpoint: {ckpt_path}")

    wandb_run.finish()
    print("Training complete!")
