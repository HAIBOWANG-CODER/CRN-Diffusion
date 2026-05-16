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

DATASET_CONFIGS = {
    "mnist": {
        "img_size": 28,
        "img_channels": 1,
        "n_classes": 10,
        "data_root": REAL_MNIST_DIR,
    },
    "cifar100": {
        "img_size": 32,
        "img_channels": 3,
        "n_classes": 100,
        "data_root": PROJECT_ROOT / "cifar100_data",
    },
}

DEFAULT_DATASET = "mnist"


class Config:
    # ── 数据集选择 ────────────────────────────────────────────────
    DATASET = DEFAULT_DATASET

    @classmethod
    def _ds(cls):
        return DATASET_CONFIGS[cls.DATASET]

    @classmethod
    def get_img_size(cls):
        return cls._ds()["img_size"]

    @classmethod
    def get_img_channels(cls):
        return cls._ds()["img_channels"]

    @classmethod
    def get_n_classes(cls):
        return cls._ds()["n_classes"]

    @classmethod
    def get_data_root(cls):
        return cls._ds()["data_root"]

    # ── CRN 物理参数 ──────────────────────────────────────────────
    V0 = 100.0
    T = 5.0
    DT = 0.02
    T_MIN = 0.05   # minimum t for training (avoids near-zero variance blowup)

    # ── 静态属性（向后兼容，读取当前 DATASET 配置）────────────────
    IMG_SIZE = 28
    IMG_CHANNELS = 1
    N_CLASSES = 10

    # ── 训练 ─────────────────────────────────────────────────────
    BATCH_SIZE = 256
    NUM_WORKERS = 4
    EPOCHS = 200
    LR = 1e-4
    LR_WARMUP_EPOCHS = 5    # linear warmup epochs
    LR_MIN = 1e-6           # cosine annealing floor
    EMA_DECAY = 0.999
    GRAD_CLIP = 1.0

    # ── 模型 ─────────────────────────────────────────────────────
    CHANNELS = [64, 128, 256, 512]
    TIME_EMB_DIM = 128
    DROPOUT = 0.1

    # ── 采样 ─────────────────────────────────────────────────────
    SAMPLE_EVERY = 5
    NUM_SAMPLES = 64
    SAMPLE_STEPS = 200
    SAMPLE_STEPS_QUICK = 50

    # ── 评估 ─────────────────────────────────────────────────────
    EVAL_SAMPLES = 1000

    # ── wandb ────────────────────────────────────────────────────
    WANDB_PROJECT = "crn-diffusion-mnist"
    WANDB_ENTITY = None
    WANDB_MODE = "online"
    WANDB_KEY = "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"

    # ── 设备 ─────────────────────────────────────────────────────
    DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"


_DATASET_HYPERPARAMS = {
    "mnist": {},   # use defaults
    "cifar100": {
        # v0=255: CRN particle count n = pixel*255 maps exactly to 8-bit integers.
        # This also shrinks pt_target magnitude by 25x vs v0=10, stabilising training.
        "V0": 255.0,
        "T": 10.0,
        "T_MIN": 0.2,
        "EPOCHS": 500,
        "BATCH_SIZE": 128,
        "LR": 2e-4,
        "GRAD_CLIP": 0.5,
    },
}


def apply_dataset(dataset_name: str):
    """
    切换当前数据集，更新 Config 的静态属性，使旧代码无需改动。
    在 main.py / train.py 入口处调用一次即可。
    """
    assert dataset_name in DATASET_CONFIGS, \
        f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_CONFIGS)}"
    cfg = DATASET_CONFIGS[dataset_name]
    Config.DATASET = dataset_name
    Config.IMG_SIZE = cfg["img_size"]
    Config.IMG_CHANNELS = cfg["img_channels"]
    Config.N_CLASSES = cfg["n_classes"]
    cfg["data_root"].mkdir(parents=True, exist_ok=True)

    for key, val in _DATASET_HYPERPARAMS.get(dataset_name, {}).items():
        setattr(Config, key, val)
