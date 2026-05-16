"""
数据集加载与预处理。

支持数据集：
  - mnist:    28x28x1, [0,1]
  - cifar100: 32x32x1 (灰度), [0,1]，训练集含 RandomHorizontalFlip

新增数据集只需在 config.DATASET_CONFIGS 中添加一项，
并在下方 _DATASET_REGISTRY 中注册对应的构建函数。
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image
from . import config


# ── 数据集注册表 ──────────────────────────────────────────────────────────────
# 每个 key 对应一个 callable(train: bool, root: Path) -> Dataset

def _build_mnist(train: bool, root: Path):
    tf = transforms.Compose([
        transforms.ToTensor(),          # [0,1], (1,28,28)
    ])
    return datasets.MNIST(root=str(root), train=train, download=True, transform=tf)


def _build_cifar100(train: bool, root: Path):
    tf_list = []
    if train:
        tf_list.append(transforms.RandomHorizontalFlip())
    tf_list.append(transforms.ToTensor())   # RGB [0,1], (3,32,32)
    return datasets.CIFAR100(root=str(root), train=train, download=True,
                              transform=transforms.Compose(tf_list))


_DATASET_REGISTRY = {
    "mnist":    _build_mnist,
    "cifar100": _build_cifar100,
}


# ── 公共接口 ──────────────────────────────────────────────────────────────────

def get_dataset(train: bool = True):
    name = config.Config.DATASET
    assert name in _DATASET_REGISTRY, \
        f"Dataset '{name}' not registered. Available: {list(_DATASET_REGISTRY)}"
    root = config.DATASET_CONFIGS[name]["data_root"]
    return _DATASET_REGISTRY[name](train, root)


def get_dataloader(train=True, batch_size=None, shuffle=None, drop_last=True):
    bs = batch_size or config.Config.BATCH_SIZE
    shuffle = shuffle if shuffle is not None else train

    loader = DataLoader(
        get_dataset(train=train),
        batch_size=bs,
        shuffle=shuffle,
        num_workers=config.Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=drop_last,
    )
    return loader


def save_real_mnist_samples(n=1000, save_dir=None):
    """保存真实 MNIST 样本用于 FID 计算（逐张 PNG 存入 fid_real/ 子目录）。"""
    from torchvision.utils import make_grid as _make_grid

    sd = Path(save_dir) if save_dir else config.REAL_MNIST_DIR
    fid_dir = sd / "fid_real"
    fid_dir.mkdir(parents=True, exist_ok=True)

    loader = get_dataloader(train=False, batch_size=n, shuffle=False, drop_last=False)
    imgs, _ = next(iter(loader))
    imgs_scaled = (imgs * config.Config.V0).clamp(0, config.Config.V0).long()
    torch.save(imgs_scaled.float() / config.Config.V0, sd / "real_samples.pt")

    grid = _make_grid(imgs_scaled.float() / config.Config.V0, nrow=int(n ** 0.5))
    grid_np = (grid.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
    Image.fromarray(grid_np, mode="RGB").save(sd / "real_grid.png")

    for i in range(min(n, len(imgs))):
        img_np = (imgs[i].squeeze().numpy() * 255).clip(0, 255).astype("uint8")
        rgb = Image.fromarray(img_np, mode="L").convert("RGB")
        rgb.save(fid_dir / f"real_{i:05d}.png")


def make_grid(tensor, nrow=8, padding=2, normalize=False):
    from torchvision.utils import make_grid as _make_grid
    return _make_grid(tensor, nrow=nrow, padding=padding, normalize=normalize, value_range=(0, 1))


if __name__ == "__main__":
    loader = get_dataloader(train=True, batch_size=256)
    x, y = next(iter(loader))
    print(f"Dataset : {config.Config.DATASET}")
    print(f"x shape : {x.shape}, dtype: {x.dtype}, range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"y shape : {y.shape}, unique sample: {y[:8].tolist()}")
    print(f"mean={x.mean():.4f}, var={x.var():.4f}")
