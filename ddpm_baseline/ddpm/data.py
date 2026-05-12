"""
MNIST 数据加载：归一化到 [-1, 1]（DDPM 标准惯例）。
直接读取已下载的二进制文件，不依赖 torchvision 的自动下载（避免测试集网络问题）。
"""
import struct
import gzip
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from . import config

# Search order for MNIST raw files
_CANDIDATES = [
    Path(__file__).parent.parent / "data" / "MNIST" / "raw",
    Path(__file__).parent.parent.parent / "CRN-based-Diffusion-Models" / "real_mnist" / "MNIST" / "raw",
]


def _find_mnist_raw_dir():
    for p in _CANDIDATES:
        if (p / "train-images-idx3-ubyte").exists():
            return p
        if (p / "train-images-idx3-ubyte.gz").exists():
            return p
    return None


def _read_idx(path: Path):
    """Read an IDX file (possibly gzipped) and return a numpy array."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        n_dims = magic & 0xFF
        dims = [struct.unpack(">I", f.read(4))[0] for _ in range(n_dims)]
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(dims)


def _load_mnist_train():
    """Load MNIST training images from raw binary files. Returns float tensor in [-1, 1]."""
    raw_dir = _find_mnist_raw_dir()
    if raw_dir is None:
        raise RuntimeError(
            "MNIST raw files not found. Run training once with internet access, "
            "or manually place train-images-idx3-ubyte[.gz] in "
            f"{_CANDIDATES[0]}"
        )

    # Try uncompressed first, then .gz
    img_path = raw_dir / "train-images-idx3-ubyte"
    if not img_path.exists():
        img_path = raw_dir / "train-images-idx3-ubyte.gz"

    lbl_path = raw_dir / "train-labels-idx1-ubyte"
    if not lbl_path.exists():
        lbl_path = raw_dir / "train-labels-idx1-ubyte.gz"

    images = _read_idx(img_path)          # (60000, 28, 28) uint8
    labels = _read_idx(lbl_path)          # (60000,) uint8

    # Normalize: uint8 [0,255] → float [-1, 1]
    imgs_f = torch.from_numpy(images).float().unsqueeze(1) / 127.5 - 1.0  # (N,1,28,28)
    lbls   = torch.from_numpy(labels.copy()).long()
    return imgs_f, lbls


def get_dataloader(train=True, batch_size=None, shuffle=None, drop_last=None):
    bs        = batch_size or config.Config.BATCH_SIZE
    shuffle   = shuffle   if shuffle   is not None else train
    drop_last = drop_last if drop_last is not None else train

    if train:
        # Load from raw binary — no network needed
        imgs, lbls = _load_mnist_train()
        dataset = TensorDataset(imgs, lbls)
    else:
        # For eval split try torchvision; fall back to a held-out subset of train
        raw_dir = _find_mnist_raw_dir()
        test_path = raw_dir / "t10k-images-idx3-ubyte" if raw_dir else None
        test_gz   = raw_dir / "t10k-images-idx3-ubyte.gz" if raw_dir else None

        if test_path and test_path.exists() or test_gz and test_gz.exists():
            img_p = test_path if (test_path and test_path.exists()) else test_gz
            lbl_p = raw_dir / "t10k-labels-idx1-ubyte"
            if not lbl_p.exists():
                lbl_p = raw_dir / "t10k-labels-idx1-ubyte.gz"
            images = _read_idx(img_p)
            labels = _read_idx(lbl_p)
            imgs_f = torch.from_numpy(images).float().unsqueeze(1) / 127.5 - 1.0
            lbls   = torch.from_numpy(labels.copy()).long()
            dataset = TensorDataset(imgs_f, lbls)
        else:
            # Fall back: use last 10000 of training set as validation
            imgs, lbls = _load_mnist_train()
            dataset = TensorDataset(imgs[-10000:], lbls[-10000:])

    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=config.Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=drop_last,
    )
""