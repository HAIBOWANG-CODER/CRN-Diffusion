# DDPM Baseline — MNIST

标准 DDPM 实现，作为 CRN-based Diffusion Model 的对比基线。

## 网络 & 训练参数（与 CRN 项目完全对齐）

| 参数 | 值 |
|------|-----|
| U-Net channels | [64, 128, 256, 512] |
| Time emb dim | 128 |
| Dropout | 0.1 |
| Batch size | 256 |
| Epochs | 200 |
| LR | 1e-4 |
| EMA decay | 0.999 |
| Grad clip | 1.0 |
| Sample every | 5 epochs |

## DDPM 扩散参数

| 参数 | 值 |
|------|-----|
| T（总步数） | 1000 |
| β schedule | linear, 1e-4 → 0.02 |
| 预测目标 | 噪声 ε |
| 训练采样器 | DDPM |
| 监控采样器 | DDIM 50 步（快速） |
| 完整采样 | DDPM 1000 步 |

## 运行

```bash
cd /Users/wanghaibo/Desktop/ddpm_baseline

# 训练（默认 online wandb）
python run.py --mode train

# 训练（禁用 wandb）
python run.py --mode train --wandb_mode disabled

# 采样（自动选最新 checkpoint，DDIM 50步）
python run.py --mode sample --ddim --steps 50

# 采样（完整 DDPM 1000步）
python run.py --mode sample --steps 1000
```

## 输出文件

- `samples/epochNNNN.png` — 每 5 个 epoch 生成的 8×8 图像 grid，左上角标注 epoch 和 loss
- `checkpoints/ckpt_epochNNNN.pt` — 模型权重
- `generated/` — `sample` 模式输出的图像

## 与 CRN 项目的区别

| | DDPM Baseline | CRN Diffusion |
|--|--|--|
| 噪声分布 | Gaussian | Poisson (CRN birth-death) |
| 时间步 | 离散 {1,...,1000} | 连续 [0, 8] |
| 目标 | 预测噪声 ε | 预测 score p_t = -(x_t-μ)/σ² |
| 前向采样 | 解析式加噪 | 精确线性生灭过程 |
| 反向采样 | DDPM/DDIM | DDPM风格 x_0 预测 |
| 数据归一化 | [-1, 1] | [0, 1] |
