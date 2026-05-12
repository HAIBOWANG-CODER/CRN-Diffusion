from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
SAMPLE_DIR     = PROJECT_ROOT / "samples"
GENERATED_DIR  = PROJECT_ROOT / "generated"
DATA_DIR       = PROJECT_ROOT / "data"

for d in (CHECKPOINT_DIR, SAMPLE_DIR, GENERATED_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)


class Config:
    # ── DDPM 扩散参数 ─────────────────────────────────────────────
    T            = 1000          # 总时间步数
    BETA_START   = 1e-4
    BETA_END     = 0.02

    # ── 数据 ─────────────────────────────────────────────────────
    IMG_SIZE     = 28
    IMG_CHANNELS = 1             # MNIST 灰度

    # ── 训练（与 CRN 项目对齐）────────────────────────────────────
    BATCH_SIZE   = 256
    NUM_WORKERS  = 4
    EPOCHS       = 200
    LR           = 1e-4
    EMA_DECAY    = 0.999
    GRAD_CLIP    = 1.0

    # ── 模型（与 CRN 项目完全一致）────────────────────────────────
    CHANNELS     = [64, 128, 256, 512]
    TIME_EMB_DIM = 128
    DROPOUT      = 0.1

    # ── 采样 ─────────────────────────────────────────────────────
    SAMPLE_EVERY       = 5    # 每 N 个 epoch 生一次图
    NUM_SAMPLES        = 64
    SAMPLE_STEPS       = 1000  # 完整 DDPM 采样步数
    SAMPLE_STEPS_QUICK = 50    # DDIM 快速采样，训练监控用

    # ── wandb ────────────────────────────────────────────────────
    WANDB_PROJECT = "ddpm-mnist-baseline"
    WANDB_ENTITY  = None
    WANDB_MODE    = "online"
    WANDB_KEY     = "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"
