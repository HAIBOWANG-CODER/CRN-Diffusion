"""
CRN-based Diffusion Model — 主入口。

Usage:
    python main.py --mode train      # 训练
    python main.py --mode sample     # 采样
    python main.py --mode evaluate  # 评估
    python main.py --mode all       # 训练 → 采样 → 评估
"""
import argparse
import math
import torch

from . import config
from .config import apply_dataset
from .data import save_real_mnist_samples


def setup_wandb():
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
            "IMG_CHANNELS": config.Config.IMG_CHANNELS,
            "CHANNELS": config.Config.CHANNELS,
            "TIME_EMB_DIM": config.Config.TIME_EMB_DIM,
        },
    )
    return wandb.run


def run_train(wandb_run=None):
    from .train import train_epoch
    from .model import UNet, UNetCIFAR, EMA
    from .forward import CRNForwardProcess
    from torch.cuda.amp import GradScaler
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    ModelClass = UNetCIFAR if config.Config.DATASET == "cifar100" else UNet
    model = ModelClass().to(device)
    print(f"[Train] 模型: {ModelClass.__name__}, 参数量: {sum(p.numel() for p in model.parameters()):,}")

    ema = EMA(model, decay=config.Config.EMA_DECAY)
    optimizer = optim.Adam(model.parameters(), lr=config.Config.LR)

    # Cosine annealing with linear warmup
    def lr_lambda(epoch):
        warmup = config.Config.LR_WARMUP_EPOCHS
        total = config.Config.EPOCHS
        if epoch < warmup:
            return epoch / max(warmup, 1)
        progress = (epoch - warmup) / max(total - warmup, 1)
        min_ratio = config.Config.LR_MIN / config.Config.LR
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()
    crn = CRNForwardProcess(device=device)

    from .data import get_dataloader

    for epoch in range(1, config.Config.EPOCHS + 1):
        loss = train_epoch(model, ema, optimizer, scaler, crn, device, epoch, wandb_run)
        scheduler.step()

        if wandb_run:
            import wandb
            wandb_run.log({"train/lr": scheduler.get_last_lr()[0], "epoch": epoch})

        if epoch % config.Config.SAMPLE_EVERY == 0:
            ema.apply_shadow()
            from .train import sample_images
            sample_path = sample_images(ema.model, None, crn, device,
                                       n=config.Config.NUM_SAMPLES, epoch=epoch, loss=loss)
            if wandb_run:
                import wandb
                wandb_run.log({"sample/grid": wandb.Image(str(sample_path)), "sample/epoch": epoch})
            ema.restore()

            import os
            ckpt_path = config.CHECKPOINT_DIR / f"ckpt_epoch{epoch:04d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "ema_shadow": ema.shadow,
                "optimizer_state": optimizer.state_dict(),
            }, ckpt_path)
            print(f"[Train] Checkpoint: {ckpt_path}")


def run_sample(ckpt_path=None, n=64, steps=200):
    from .sample import generate_samples
    from .model import UNet, UNetCIFAR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Sample] Device: {device}")

    if ckpt_path is None:
        import glob
        checkpoints = sorted(glob.glob(str(config.CHECKPOINT_DIR / "ckpt_*.pt")))
        if checkpoints:
            ckpt_path = checkpoints[-1]
            print(f"[Sample] Auto-select latest: {ckpt_path}")
        else:
            print("[Sample] ERROR: No checkpoint found.")
            return

    ModelClass = UNetCIFAR if config.Config.DATASET == "cifar100" else UNet
    generate_samples(ckpt_path, n=n, steps=steps, model_class=ModelClass)


def run_evaluate(ckpt_path, n_samples=1000, wandb_run=None):
    from .evaluate import evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluate] Device: {device}")

    if ckpt_path is None:
        import glob
        checkpoints = sorted(glob.glob(str(config.CHECKPOINT_DIR / "ckpt_*.pt")))
        if checkpoints:
            ckpt_path = checkpoints[-1]
            print(f"[Evaluate] Auto-select latest: {ckpt_path}")
        else:
            print("[Evaluate] ERROR: No checkpoint found.")
            return

    return evaluate(ckpt_path, n_samples=n_samples, wandb_run=wandb_run)


def main():
    parser = argparse.ArgumentParser(description="CRN-based Diffusion Model")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "sample", "evaluate", "all"],
                        help="运行模式")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint 路径（sample/evaluate 模式可省略，自动选最新）")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="生成样本数量（默认 1000）")
    parser.add_argument("--n_display", type=int, default=64,
                        help="展示样本数量（默认 64）")
    parser.add_argument("--steps", type=int, default=200,
                        help="采样步数（默认 200）")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=list(config.DATASET_CONFIGS.keys()),
                        help="数据集（默认使用 config.DEFAULT_DATASET）")
    parser.add_argument("--wandb_mode", type=str, default=None,
                        choices=["online", "offline", "disabled"],
                        help="覆盖 wandb 模式")
    args = parser.parse_args()

    if args.dataset:
        apply_dataset(args.dataset)

    if args.wandb_mode:
        config.Config.WANDB_MODE = args.wandb_mode

    n_samples = args.n_samples or config.Config.EVAL_SAMPLES

    print("=" * 60)
    print("CRN-based Diffusion Model")
    print(f"Mode: {args.mode}")
    print(f"Device: cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataset: {config.Config.DATASET} ({config.Config.IMG_SIZE}x{config.Config.IMG_SIZE}x{config.Config.IMG_CHANNELS})")
    print(f"V0={config.Config.V0}, T={config.Config.T}, DT={config.Config.DT}")
    print(f"BATCH_SIZE={config.Config.BATCH_SIZE}, EPOCHS={config.Config.EPOCHS}")
    print("=" * 60)

    wandb_run = None

    if args.mode in ("train", "all"):
        wandb_run = setup_wandb()
        run_train(wandb_run=wandb_run)

    if args.mode == "sample":
        run_sample(ckpt_path=args.ckpt, n=args.n_display, steps=args.steps)

    if args.mode == "evaluate":
        wandb_run = setup_wandb()
        run_evaluate(args.ckpt, n_samples=n_samples, wandb_run=wandb_run)

    if args.mode == "all":
        run_sample(ckpt_path=args.ckpt, n=args.n_display, steps=args.steps)
        wandb_run = setup_wandb() if not wandb_run else wandb_run
        run_evaluate(None, n_samples=n_samples, wandb_run=wandb_run)

    if wandb_run:
        wandb_run.finish()

    print("\nDone!")


if __name__ == "__main__":
    main()
