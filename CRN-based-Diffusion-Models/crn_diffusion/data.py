"""
MNIST 加载与预处理。
每个像素独立对应一条 CRN (∅ ⇌ X)，重整化后 x = n / v0 ∈ [0, 1)。
"""
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image
from . import config


class MNISTPoissonDataset(Dataset):
    """
    MNIST 数据集，返回归一化到 [0, 1) 的浮点张量。
    像素值 n ∈ {0, 1, ..., 255} → x = n / v0 ∈ [0, 1]。

    标签 y ∈ {0..9} 同时返回，用于分层采样。
    """

    def __init__(self, train=True, root=None):
        self.root = Path(root) if root else config.REAL_MNIST_DIR
        self.train = train

        self.dataset = datasets.MNIST(
            root=str(self.root),
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]),
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        x = img.float()                      # (1, 28, 28), [0, 1]
        return x, label


def get_dataloader(train=True, batch_size=None, shuffle=None, drop_last=True):
    bs = batch_size or config.Config.BATCH_SIZE
    shuffle = shuffle if shuffle is not None else train

    loader = DataLoader(
        MNISTPoissonDataset(train=train),
        batch_size=bs,
        shuffle=shuffle,
        num_workers=config.Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=drop_last,
    )
    return loader


def save_real_mnist_samples(n=1000, save_dir=None):
    """保存真实 MNIST 样本用于 FID 计算（逐张 PNG 存入 fid_real/ 子目录）。"""
    sd = Path(save_dir) if save_dir else config.REAL_MNIST_DIR
    fid_dir = sd / "fid_real"
    fid_dir.mkdir(parents=True, exist_ok=True)

    loader = get_dataloader(train=False, batch_size=n, shuffle=False, drop_last=False)
    imgs, _ = next(iter(loader))
    imgs_scaled = (imgs * config.Config.V0).clamp(0, config.Config.V0).long()
    torch.save(imgs_scaled.float() / config.Config.V0, sd / "real_samples.pt")

    grid = make_grid(imgs_scaled.float() / config.Config.V0, nrow=int(n ** 0.5))
    grid_np = (grid.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
    Image.fromarray(grid_np, mode="RGB").save(sd / "real_grid.png")

    for i in range(min(n, len(imgs))):
        img_np = (imgs[i].squeeze().numpy() * 255).clip(0, 255).astype("uint8")
        rgb = Image.fromarray(img_np, mode="L").convert("RGB")
        rgb.save(fid_dir / f"real_{i:05d}.png")


def make_grid(tensor, nrow=8, padding=2, normalize=False):
    """简单 grid 可视化，torch 兼容写法。"""
    from torchvision.utils import make_grid as _make_grid
    return _make_grid(tensor, nrow=nrow, padding=padding, normalize=normalize, value_range=(0, 1))


if __name__ == "__main__":
    loader = get_dataloader(train=True, batch_size=256)
    x, y = next(iter(loader))
    print(f"x shape: {x.shape}, dtype: {x.dtype}, range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"y shape: {y.shape}, unique: {y.unique()}")
    print(f"每像素均值: {x.mean():.4f}, 方差: {x.var():.4f}")
