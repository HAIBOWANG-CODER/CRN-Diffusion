import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
SAMPLE_DIR = PROJECT_ROOT / "samples"
GENERATED_DIR = PROJECT_ROOT / "generated"
REAL_MNIST_DIR = PROJECT_ROOT / "real_mnist"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
REAL_MNIST_DIR.mkdir(parents=True, exist_ok=True)


class Config:
    # ── CRN 物理参数 ──────────────────────────────────────────────
    V0 = 10.0    # reduced from 255 — momentum magnitude scales as √V0 ≈ 3, manageable for MSE
    T = 8.0
    DT = 0.02

    # ── 数据 ─────────────────────────────────────────────────────
    DATASET = "mnist"
    IMG_SIZE = 28
    IMG_CHANNELS = 1
    N_CLASSES = 10

    # ── 训练 ─────────────────────────────────────────────────────
    BATCH_SIZE = 256
    NUM_WORKERS = 4
    EPOCHS = 200
    LR = 1e-4
    EMA_DECAY = 0.999
    GRAD_CLIP = 1.0

    # ── 模型 ─────────────────────────────────────────────────────
    CHANNELS = [64, 128, 256, 512]
    TIME_EMB_DIM = 128
    DROPOUT = 0.1

    # ── 采样 ─────────────────────────────────────────────────────
    SAMPLE_EVERY = 5          # generate grid every N epochs during training
    NUM_SAMPLES = 64
    SAMPLE_STEPS = 200        # full quality steps (evaluate/sample scripts)
    SAMPLE_STEPS_QUICK = 50   # fast steps used during training monitoring

    # ── 评估 ─────────────────────────────────────────────────────
    EVAL_SAMPLES = 1000

    # ── wandb ────────────────────────────────────────────────────
    WANDB_PROJECT = "crn-diffusion-mnist"
    WANDB_ENTITY = None
    WANDB_MODE = "online"
    WANDB_KEY = "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"

    # ── 设备 ─────────────────────────────────────────────────────
    DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
