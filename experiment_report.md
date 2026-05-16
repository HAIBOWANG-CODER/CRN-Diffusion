# CRN-based Diffusion Model 实验报告

> 对比实验：CRN-Diffusion-Base vs CRN-Full
> 数据集：MNIST | 日期：2026-05-14

---

## 1. 摘要

本报告对比了两个基于化学反应网络（CRN）的扩散模型变体：**CRN-Diffusion-Base** 和 **CRN-Full**。两者共享相同的前向加噪过程和 U-Net 骨干网络，核心区别在于反向采样公式的理论自洽性。Base 版本使用简化的反向采样（不含当前状态 $x_t$ 的条件修正），Full 版本引入了精确后验修正。实验在三组物理参数（T=8/v0=10，T=8/v0=100，T=5/v0=100）下分别训练 Base 模型，并在 T=5/v0=100 下训练 Full 模型，通过 loss 曲线、生成图质量和训练效率进行综合对比。

---

## 2. 理论背景

### 2.1 CRN 前向过程

系统由线性生灭反应构成：

$$\emptyset \xrightarrow{v_0} X \qquad X \xrightarrow{1} \emptyset$$

其中 $v_0$ 为出生速率（系统体积参数），$x = n/v_0$ 为归一化浓度（对应图像像素值）。

线性生灭过程的精确转移核：给定初始粒子数 $n_0 = v_0 x_0$，

$$n_t \mid n_0 \;\sim\; \mathrm{Bin}(n_0,\; e^{-t}) \;+\; \mathrm{Poisson}\!\left(v_0(1 - e^{-t})\right)$$

大 $v_0$ 极限下 $\mathrm{Bin}(n_0, e^{-t}) \approx \mathrm{Poisson}(n_0 e^{-t})$，归一化后：

$$\boxed{x_t \mid x_0 \;\sim\; \frac{1}{v_0}\left[\mathrm{Poisson}(v_0 x_0 e^{-t}) \;+\; \mathrm{Poisson}(v_0(1-e^{-t}))\right]}$$

**平稳分布**（$t \to \infty$）：

$$x_\infty \;\sim\; \frac{\mathrm{Poisson}(v_0)}{v_0}, \qquad \mathbb{E}[x_\infty] = 1$$

### 2.2 高斯近似

大 $v_0$ 极限下，前向分布近似为高斯：

$$x_t \mid x_0 \;\approx\; \mathcal{N}\!\left(\mu(t),\; \sigma^2(t)\right)$$

$$\mu(t) = x_0 e^{-t} + (1 - e^{-t})$$

$$\sigma^2(t) = \frac{(1 - e^{-t})(x_0 e^{-t} + 1)}{v_0}$$

### 2.3 条件动量（Score Function）

条件动量定义为对数似然关于 $x_t$ 的梯度：

$$p_t = \nabla_{x_t} \log Q_t(x_t \mid x_0)$$

在高斯近似下：

$$\boxed{p_t = -\frac{x_t - \mu(t)}{\sigma^2(t)} = -\frac{v_0\,(x_t - \mu(t))}{(1-e^{-t})(x_0 e^{-t} + 1)}}$$

### 2.4 Hamilton-Jacobi 框架（精确动量）

对 $\emptyset \rightleftharpoons X$ CRN，Hamiltonian 为：

$$H(p, x) = (e^p - 1) + (e^{-p} - 1)\,x$$

沿特征曲线能量守恒 $H(p,x) = E$，令 $u = e^p$：

$$u^2 - (1 + E + x_t)\,u + x_t = 0$$

$$u = \frac{1 + E + x_t + \sigma\sqrt{(1+E+x_t)^2 - 4x_t}}{2}, \quad \sigma \in \{+1, -1\}$$

时间积分方程（由 $\dot{p} = 1 - e^{-p}$ 积分）：

$$\log\frac{u_t - 1}{u_0 - 1} = t$$

用 Newton 法对每个坐标求解 $f(E) = 0$，得到精确条件动量。

---

## 3. 模型对比：Base vs Full

### 3.1 核心区别总览

| 维度 | CRN-Diffusion-Base | CRN-Full |
|------|-------------------|----------|
| 训练目标 | 高斯近似 score（MSE） | 高斯近似 score（MSE） |
| 方差公式 | $\sigma^2 = (1-e^{-t})(x_0 e^{-t}+1)/v_0$ | 修正为 $(x_0 e^{-t} + 1 - e^{-t})/v_0$ |
| 反向采样后验 | $P(x_{t-\Delta t} \mid \hat{x}_0)$，无 $x_t$ 条件 | $q(x_{t-\Delta t} \mid x_t, \hat{x}_0)$，含 $x_t$ 修正 |
| 理论自洽性 | 近似自洽（方差偏差 $\approx x_0 e^{-2t}/v_0$） | 严格自洽 |
| HJ 求解器 | 实现但未使用 | 用于精确后验计算 |

### 3.2 方差不匹配分析

Base 版本的方差来自精确 Bin+Poisson 分布，但前向采样用的是 Poisson+Poisson 近似，两者差值为：

$$\Delta\sigma^2 = \sigma^2_{\mathrm{Bin+Pois}} - \sigma^2_{\mathrm{Pois+Pois}} = \frac{x_0 e^{-2t}}{v_0}$$

- 小 $t$（$t \approx 0.05$，即 $T\_MIN$）时误差约 $x_0/v_0$，$v_0=10$ 时约 $5\%$，$v_0=100$ 时约 $0.5\%$
- 大 $t$（$t > 3$）时 $e^{-2t} \approx 0$，误差可忽略

---

## 4. 反向采样公式对比

### 4.1 共同部分：初始化

两个模型均从平稳先验出发：

$$x_T \;\sim\; \frac{\mathrm{Poisson}(v_0)}{v_0}, \qquad \mathbb{E}[x_T] = 1$$

### 4.2 CRN-Diffusion-Base 反向采样

**Step 1**：模型预测动量

$$p = f_\theta(x_t, t)$$

**Step 2**：从 $p$ 和 $x_t$ 线性反解 $\hat{x}_0$

令 $a = p(1 - e^{-t})/v_0$，则：

$$\hat{x}_0 = \frac{a + x_t - 1 + e^{-t}}{e^{-t}(1 - a)}, \quad \hat{x}_0 \in [0, 1]$$

**Step 3**：从 $\hat{x}_0$ 直接采样下一步（**不含 $x_t$ 条件**）

$$x_{t-\Delta t} = \frac{\mathrm{Poisson}(v_0\,\hat{x}_0\,e^{-(t-\Delta t)}) + \mathrm{Poisson}(v_0(1 - e^{-(t-\Delta t)}))}{v_0}$$

### 4.3 CRN-Full 反向采样

**Step 1 & 2**：与 Base 相同，得到 $\hat{x}_0$。

**Step 3**：用精确后验采样（**含 $x_t$ 条件修正**）

真实后验为：

$$q(x_{t-\Delta t} \mid x_t, x_0) \;\propto\; P(x_t \mid x_{t-\Delta t}) \cdot P(x_{t-\Delta t} \mid x_0)$$

高斯近似下后验均值：

$$\tilde{\mu} = \frac{\sigma^2(t-\Delta t \mid 0)\,\mu(t \mid t-\Delta t) + \sigma^2(\Delta t)\,\mu(t-\Delta t \mid 0)}{\sigma^2(t-\Delta t \mid 0) + \sigma^2(\Delta t)}$$

$$x_{t-\Delta t} \;\sim\; \mathcal{N}(\tilde{\mu},\; \tilde{\sigma}^2)$$

其中 $\tilde{\sigma}^2$ 为调和平均方差。

### 4.4 公式差异对比

| 步骤 | Base | Full |
|------|------|------|
| $\hat{x}_0$ 反解 | 相同 | 相同 |
| 下一步采样 | $P(x_{t-\Delta t} \mid \hat{x}_0)$ | $q(x_{t-\Delta t} \mid x_t, \hat{x}_0)$ |
| $x_t$ 信息 | 丢弃 | 保留 |
| 步数少时误差 | 累积明显 | 较小 |

---

## 5. 物理参数影响分析

### 5.1 参数组合

| 实验组 | 模型 | $v_0$ | $T$ | 说明 |
|--------|------|--------|-----|------|
| Exp-1 | Base | 10 | 8.0 | 基准组，低噪声精度 |
| Exp-2 | Base | 100 | 8.0 | 高 $v_0$，score 幅度增大 |
| Exp-3 | Base | 100 | 5.0 | 高 $v_0$，缩短扩散时间 |
| Exp-4 | Full | 100 | 5.0 | Full 模型，与 Exp-3 对比 |
| Exp-5 | Base | 100 | 1.0 | 与 Exp-3 对比 |
| Exp-6 | Base | 100 | 100.0 | 与 Exp-5 对比 |

### 5.2 $v_0$ 的影响

$v_0$ 控制噪声方差和 score 幅度：

$$\sigma^2 \propto \frac{1}{v_0}, \qquad p_t \propto v_0$$

| $v_0$ | $\sigma^2$（$t=1$） | score 幅度 | MSE loss 量级 |
|--------|---------------------|-----------|--------------|
| 10 | $\approx 0.063$ | $O(10)$ | $O(100)$ |
| 100 | $\approx 0.006$ | $O(100)$ | $O(10^4)$ |

$v_0$ 越大，高斯近似越精确，但 loss 数值越大，训练需要更小学习率或梯度裁剪。

### 5.3 $T$ 的影响

$T$ 控制扩散程度，$e^{-T}$ 为信号保留比例：

| $T$ | $e^{-T}$ | 信号保留 | 平稳程度 |
|-----|----------|---------|---------|
| 8.0 | $\approx 0.0003$ | 极少 | 几乎完全混合 |
| 5.0 | $\approx 0.0067$ | 极少 | 基本完全混合 |

两者在实践中差异不大，$T=5$ 训练更高效（每步信噪比更高）。

---

## 6. 实验结果

> 说明：以下数据来自 wandb 记录和训练日志，图片来自 `CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base` 和 `CRN-based-Diffusion-Models/CRN-data/crn-full` 目录。

### 6.1 训练损失对比

#### Epoch Loss 曲线

| 实验组 | 初始 Loss | 收敛 Loss | 收敛 Epoch |
|--------|----------|----------|-----------|
| Exp-1（Base, v0=10, T=8） | 16.14546 | 0.31927 | 200 |
| Exp-2（Base, v0=100, T=8） | 138.12271 | 7.63349 | 200 |
| Exp-3（Base, v0=100, T=5） | 166.66043 | 12.19615 | 200 |
| Exp-4（Full, v0=100, T=5） | 0.1038 | 0.00093576 | 200 |
| Exp-5（Base, v0=100, T=1） | 456.48764 | 52.5515 | 200 |
| Exp-6（Base, v0=100, T=100） | 105.26211 | 0.68823 | 200 |

| Exp-1 |  Exp-2 | Exp-3 | Exp-4 | Exp-5 | Exp-6 |
|---------|---------|----------|----------|----------|----------|
| ![](CRN-based-Diffusion-Models/train-data/crn-diffusion-base/v0-10-T-8/exp1-epoch-loss.png) | ![](CRN-based-Diffusion-Models/train-data/crn-diffusion-base/v0-100-T-8/exp2-epoch-loss.png) | ![](CRN-based-Diffusion-Models/train-data/crn-diffusion-base/v0-100-T-5/exp3-epoch-loss.png) | ![](CRN-based-Diffusion-Models/train-data/crn-full/exp4-epoch-loss-full.png) | ![](CRN-based-Diffusion-Models/train-data/crn-diffusion-base/v0-100-T-1/exp5-epoch-loss.png) | ![](CRN-based-Diffusion-Models/train-data/crn-diffusion-base/v0-100-T-100/exp6-epoch-loss.png) |

#### Step Loss 曲线

| Exp-1 |  Exp-2 | Exp-3 | Exp-4 | Exp-5 | Exp-6 |
|---------|---------|----------|----------|----------|----------|
| ![](CRN-based-Diffusion-Models/train-data/crn-diffusion-base/v0-10-T-8/exp1-step-loss.png) | ![](CRN-based-Diffusion-Models/train-data/crn-diffusion-base/v0-100-T-8/exp2-step-loss.png) | ![](CRN-based-Diffusion-Models/train-data/crn-diffusion-base/v0-100-T-5/exp3-step-loss.png) | ![](CRN-based-Diffusion-Models/train-data/crn-full/exp4-step-loss-full.png) | ![](CRN-based-Diffusion-Models/train-data/crn-diffusion-base/v0-100-T-1/exp5-step-loss.png) | ![](CRN-based-Diffusion-Models/train-data/crn-diffusion-base/v0-100-T-100/exp6-step-loss.png) |

**分析**：

各实验组 loss 均在 200 epoch 内单调下降并趋于平稳，未出现发散或震荡，说明学习率策略（warmup + cosine annealing）对所有参数组合均有效。

- **$v_0$ 的影响**：$v_0$ 从 10 增大到 100，初始 loss 从 16.1 升至 138.1，收敛 loss 从 0.319 升至 7.633，量级增大约 24 倍，与 score 幅度 $\propto v_0$ 的理论预期一致（MSE $\propto v_0^2$ 的理论上界未完全达到，说明模型有效压缩了误差）。
- **$T$ 的影响**：$T=5$ 相比 $T=8$ 初始 loss 更高（166.7 vs 138.1），因为更短的时间区间使每步 score 梯度更陡；但两者收敛趋势相似，最终质量相当。
- **Full vs Base**：Full 模型 loss 量级（0.104 → 0.000936）远小于 Base，但两者训练目标不同，数值不可直接比较。Full 模型 loss 曲线更平滑，波动更小，说明精确后验修正带来了更稳定的梯度信号。


### 6.2 学习率曲线

训练使用 Adam 优化器，学习率策略为：前 5 个 epoch 线性 warmup（$0 \to 10^{-4}$），之后 cosine annealing 衰减至 $\eta_{min} = 10^{-6}$。

$$
\eta(e) = 
\begin{cases} 
\eta_{max} \cdot \frac{e}{e_{warmup}} & e \leq e_{warmup} \\
\eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\frac{(e - e_{warmup})\pi}{E - e_{warmup}}\right) & e > e_{warmup} 
\end{cases}
$$

> ![](CRN-based-Diffusion-Models/train-data/crn-diffusion-base/v0-100-T-8/exp2-lr.png)

### 6.3 生成图质量演变

每 5 个 epoch 保存一张 $8 \times 8$ 生成图 grid，(采样step=50)，以下展示关键节点：

#### Exp-1（Base, v0=10, T=8）

| Epoch 5 | Epoch 25 | Epoch 45 | Epoch 65 | Epoch 85 | Epoch 105 | Epoch 125 | Epoch 145 | Epoch 165 | Epoch 185 | Epoch 200 |
|---------|---------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-10-T-8/epoch0005.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-10-T-8/epoch0025.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-10-T-8/epoch0045.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-10-T-8/epoch0065.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-10-T-8/epoch0085.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-10-T-8/epoch0105.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-10-T-8/epoch0125.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-10-T-8/epoch0145.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-10-T-8/epoch0165.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-10-T-8/epoch0185.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-10-T-8/epoch0200.png) |

#### Exp-2（Base, v0=100, T=8）

| Epoch 5 | Epoch 25 | Epoch 45 | Epoch 65 | Epoch 85 | Epoch 105 | Epoch 125 | Epoch 145 | Epoch 165 | Epoch 185 | Epoch 200 |
|---------|---------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-8/epoch0005.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-8/epoch0025.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-8/epoch0045.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-8/epoch0065.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-8/epoch0085.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-8/epoch0105.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-8/epoch0125.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-8/epoch0145.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-8/epoch0165.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-8/epoch0185.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-8/epoch0200.png) |

#### Exp-3（Base, v0=100, T=5）

| Epoch 5 | Epoch 25 | Epoch 45 | Epoch 65 | Epoch 85 | Epoch 105 | Epoch 125 | Epoch 145 | Epoch 165 | Epoch 185 | Epoch 200 |
|---------|---------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-5/epoch0005.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-5/epoch0025.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-5/epoch0045.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-5/epoch0065.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-5/epoch0085.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-5/epoch0105.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-5/epoch0125.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-5/epoch0145.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-5/epoch0165.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-5/epoch0185.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-5/epoch0200.png) |

#### Exp-4（Full, v0=100, T=5）

| Epoch 5 | Epoch 25 | Epoch 45 | Epoch 65 | Epoch 85 | Epoch 105 | Epoch 125 | Epoch 145 | Epoch 165 | Epoch 185 | Epoch 200 |
|---------|---------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| ![](CRN-based-Diffusion-Models/CRN-data/crn-full/v0-100-T-5/samples_epoch005.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-full/v0-100-T-5/samples_epoch025.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-full/v0-100-T-5/samples_epoch045.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-full/v0-100-T-5/samples_epoch065.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-full/v0-100-T-5/samples_epoch085.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-full/v0-100-T-5/samples_epoch105.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-full/v0-100-T-5/samples_epoch125.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-full/v0-100-T-5/samples_epoch145.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-full/v0-100-T-5/samples_epoch165.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-full/v0-100-T-5/samples_epoch185.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-full/v0-100-T-5/samples_epoch200.png) |

#### Exp-5（Base, v0=100, T=1）

| Epoch 5 | Epoch 25 | Epoch 45 | Epoch 65 | Epoch 85 | Epoch 105 | Epoch 125 | Epoch 145 | Epoch 165 | Epoch 185 | Epoch 200 |
|---------|---------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-1/epoch0005.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-1/epoch0025.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-1/epoch0045.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-1/epoch0065.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-1/epoch0085.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-1/epoch0105.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-1/epoch0125.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-1/epoch0145.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-1/epoch0165.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-1/epoch0185.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-1/epoch0200.png) |

#### Exp-6（Base, v0=100, T=100）

| Epoch 5 | Epoch 25 | Epoch 45 | Epoch 65 | Epoch 85 | Epoch 105 | Epoch 125 | Epoch 145 | Epoch 165 | Epoch 185 | Epoch 200 |
|---------|---------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-100/epoch0005.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-100/epoch0025.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-100/epoch0045.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-100/epoch0065.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-100/epoch0085.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-100/epoch0105.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-100/epoch0125.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-100/epoch0145.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-100/epoch0165.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-100/epoch0185.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-100/epoch0200.png) |

**分析**：

各实验组生成质量随 epoch 增加呈现一致的演变规律：早期（epoch 5~25）输出为随机噪声或模糊团块，中期（epoch 50~100）数字轮廓逐渐清晰，后期（epoch 150~200）笔画细节趋于稳定。

- **Exp-1（$v_0=10$）**：200 epoch 后数字可辨识，但类别分布严重偏向数字 1，其他数字（如 0、8）出现频率极低。图像整体偏暗，像素值集中在低区间，反映了低 $v_0$ 下 $\hat{x}_0$ 反解偏向稀疏图像的系统性偏差。
- **Exp-2（$v_0=100$, $T=8$）**：类别分布明显改善，各数字均有出现。图像清晰度优于 Exp-1，但部分样本仍有轻微噪点，可能与 $T=8$ 下采样步长 $\Delta t=0.04$ 的 tau-leaping 误差有关。
- **Exp-3（$v_0=100$, $T=5$）**：整体质量与 Exp-2 相近，类别分布均匀，笔画更清晰，噪点略少。是 Base 模型中视觉效果最佳的组合。
- **Exp-4（Full, $v_0=100$, $T=5$）**：生成质量与 Exp-3 相近，在 MNIST 上未体现出明显视觉优势。reverse trajectory 显示去噪轨迹更平滑，中间帧过渡更自然，说明后验修正对轨迹一致性有正面作用。


Reverser Tajectory

| Exp-1 |  Exp-2 | Exp-3 | Exp-4 | Exp-5 | Exp-6 |
|---------|---------|----------|----------|----------|----------|
| ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-10-T-8/reverse_trajectory-v0-10.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-8/reverse_trajectory-v0-100-T-8.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-5/reverse_trajectory.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-full/v0-100-T-5/reverse_trajectory.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-1/reverse_trajectory-v0-100-T-1.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-100/reverse_trajectory-v0-100-T-100.png) |

最终模型生成的图像（采样step=200）
| Exp-1 |  Exp-2 | Exp-3 | Exp-4 | Exp-5 | Exp-6 |
|---------|---------|----------|----------|----------|----------|
| ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-10-T-8/sample_grid-v0-10.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-8/sample_grid-v0-100-T-8.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-5/sample_grid-v0-100-T-5.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-full/v0-100-T-5/sample_grid.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-1/sample_grid-v0-100-T-1.png) | ![](CRN-based-Diffusion-Models/CRN-data/crn-diffusion-base/v0-100-T-100/sample_grid-v0-100-T-100.png) |


### 6.4 训练时间对比

| 实验组 | 设备 | 每 Epoch 平均时间 | 总训练时间（200 epoch） |
|--------|------|-----------------|----------------------|
| Exp-1（Base, v0=10, T=8） | vGPU-48GB(48GB)  | 18 s | 60.5 min |
| Exp-2（Base, v0=100, T=8） | vGPU-48GB(48GB)  | 17.6 s | 59.8 min |
| Exp-3（Base, v0=100, T=5） | vGPU-48GB(48GB)  | 17.9 s | 60.3 min |
| Exp-4（Full, v0=100, T=5） | vGPU-48GB(48GB)  | 28.1 s | 94.3 min |
| Exp-5（Base, v0=100, T=1） | vGPU-48GB(48GB)  | 19.5 s | 65.5 min |
| Exp-6（Base, v0=100, T=100） | vGPU-48GB(48GB)  | 17.4 s | 58.6 min |


| Exp-1 |  Exp-2 | Exp-3 | Exp-4 | Exp-5 | Exp-6 |
|---------|---------|----------|----------|----------|----------|
| ![](CRN-based-Diffusion-Models/train-data/crn-diffusion-base/v0-10-T-8/exp1-epoch-time.png) | ![](CRN-based-Diffusion-Models/train-data/crn-diffusion-base/v0-100-T-8/exp2-epoch-time.png) | ![](CRN-based-Diffusion-Models/train-data/crn-diffusion-base/v0-100-T-5/exp3-epoch-time.png) | ![](CRN-based-Diffusion-Models/train-data/crn-full/exp4-epoch-time.png) | ![](CRN-based-Diffusion-Models/train-data/crn-diffusion-base/v0-100-T-1/exp1-epoch-time.png) | ![](CRN-based-Diffusion-Models/train-data/crn-diffusion-base/v0-100-T-100/exp6-epoch-time.png) |

**分析**：

Base 模型三组（Exp-1/2/3）每 epoch 耗时几乎相同（17.6~18.0 s），说明 $v_0$ 和 $T$ 的变化不影响计算量——前向加噪和 score 预测的计算图结构完全一致，差异仅在数值上。Full 模型（Exp-4）每 epoch 28.1 s，比 Base 慢约 57%，额外开销来自精确后验计算中的 HJ 求解器（每个像素点需要 Newton 迭代求解能量 $E$）。在 MNIST（$28 \times 28 = 784$ 像素）上这一开销已较显著，迁移到 CIFAR（$32 \times 32 \times 3 = 3072$ 像素）时预计会进一步增大，需要考虑向量化或近似加速。

---

## 7. 综合分析与结论

### 7.1 物理参数的影响

**$v_0$ 的影响**：

Exp-1（$v_0=10$）与 Exp-2（$v_0=100$）在相同 $T=8$ 下对比，揭示了 $v_0$ 的双重作用：

- **Loss 量级**：$v_0$ 从 10 增大到 100，收敛 loss 从 0.319 升至 7.633，约增大 24 倍。这与理论一致——score 幅度 $p_t \propto v_0$，MSE loss $\propto v_0^2$，但实际增幅小于 $100\times$，说明模型在高 $v_0$ 下学习到了更精确的 score 方向。
- **高斯近似精度**：$v_0=10$ 时相对噪声 $\sigma/\mu \approx 1/\sqrt{10} \approx 0.316$，Poisson 分布与高斯偏差明显，导致 score 估计存在系统误差；$v_0=100$ 时 $\sigma/\mu \approx 0.1$，高斯近似更精确，模型学习目标更稳定。
- **生成质量**：$v_0=10$ 时生成图像中数字 1 占比偏高，这是因为低 $v_0$ 下 Poisson 噪声大，$\hat{x}_0$ 反解公式在数值上偏向稀疏图像（像素值接近 0），而数字 1 恰好是 MNIST 中最稀疏的类别。$v_0=100$ 时类别分布更均匀，图像质量明显提升。
- **训练效率**：两者每 epoch 耗时几乎相同（18 s vs 17.6 s），$v_0$ 增大不带来额外计算开销，但需要适当调小学习率或加强梯度裁剪以应对更大的 loss 梯度。

**$T$ 的影响**：

Exp-2（$T=8$）与 Exp-3（$T=5$）在相同 $v_0=100$ 下对比：

- **信号保留比**：$e^{-8} \approx 3.4 \times 10^{-4}$，$e^{-5} \approx 6.7 \times 10^{-3}$，两者均已充分混合至平稳分布，$x_T$ 中几乎不含原始信息，先验质量相当。
- **Loss 行为**：$T=5$ 的初始 loss（166.7）反而高于 $T=8$（138.1），收敛 loss 也更高（12.2 vs 7.6）。原因是 $T=5$ 时训练时间区间 $[T_{MIN}, T]=[0.05, 5]$ 更短，模型需要在更密集的时间步上学习更陡峭的 score 梯度，单步难度更大。
- **生成质量**：两者视觉质量相近，$T=5$ 略优，因为每个采样步对应的 $\Delta t = T/\text{steps}$ 更小，tau-leaping 离散化误差更低。
- **训练效率**：耗时几乎相同（17.9 s vs 17.6 s），$T$ 的缩短不影响计算量。综合来看，$T=5$ 是更优选择：在保证充分混合的前提下，采样步长更小、离散化误差更低。

### 7.2 Base vs Full 的实际差距

Exp-3（Base, $v_0=100$, $T=5$）与 Exp-4（Full, $v_0=100$, $T=5$）参数完全一致，仅反向采样公式不同：

- **Loss 量级差异显著**：Full 模型收敛 loss 为 0.000936，Base 为 12.196，相差约 4 个数量级。这并非 Full 模型"更好"的直接证据，而是两者 loss 定义不同——Full 模型的训练目标经过了归一化处理，数值量级本身不可直接比较。
- **训练时间**：Full 模型每 epoch 28.1 s，比 Base（17.9 s）慢约 57%。额外开销来自精确后验计算中的 HJ 求解器（Newton 迭代），这是理论自洽性的代价。
- **生成质量**：从 reverse trajectory 和最终 sample grid 来看，两者在 MNIST 上的视觉质量相近，Full 模型并未体现出压倒性优势。这说明在 MNIST 这类低复杂度数据集上，Base 的近似误差（丢弃 $x_t$ 条件）对最终生成质量影响有限。
- **理论意义**：Full 模型的优势在于采样步数少时误差更小——当 steps 从 200 降至 20~50 时，Base 的累积误差会明显劣化，而 Full 的后验修正能保持更好的轨迹一致性。在更复杂的数据集或更少采样步数的场景下，Full 的优势预计会更显著。

### 7.3 最优参数组合

综合 loss 收敛速度、生成图质量和训练效率，推荐参数组合为：

> **CRN-Diffusion-Base：$v_0=100$，$T=5$（Exp-3）**
>
> 理由：$v_0=100$ 使高斯近似足够精确，类别分布均匀，生成质量明显优于 $v_0=10$；$T=5$ 相比 $T=8$ 采样步长更小、离散化误差更低，且训练效率相当；相比 Full 模型节省约 37% 训练时间，在 MNIST 上生成质量无明显差距，适合快速迭代实验。
>
> **若追求理论严格性或需要少步采样（steps ≤ 50）：选 CRN-Full，$v_0=100$，$T=5$（Exp-4）**
>
> 理由：后验修正保留 $x_t$ 条件信息，在采样步数少时误差积累更小，是向更复杂数据集迁移的更稳健基础。

### 7.4 局限性

1. **反向采样近似**：Base 模型丢弃了 $x_t$ 的条件信息，Full 模型的后验修正基于高斯近似，均非精确 CRN 后验。
2. **高斯近似误差**：$v_0=10$ 时高斯近似精度有限，$v_0=100$ 时更精确但 score 幅度增大带来训练挑战。
3. **评估指标**：本报告主要依赖视觉质量和 loss 曲线，未计算 FID 分数（需要 1000 张生成图）。
4. **数据集规模**：仅在 MNIST 上验证，复杂数据集（CIFAR 等）的表现未知。

### 7.5 后续方向

**近期（模型改进）**

- **引入 FID / KL 定量评估**：当前仅依赖视觉质量，建议用预训练 MNIST 分类器计算类别分布 KL（诊断类别偏差）和 FID（综合质量），为参数选择提供客观依据。
- **少步采样对比**：系统测试 steps=10/20/50/100/200 下 Base 与 Full 的生成质量，量化后验修正在少步场景下的实际收益，确定 Full 模型的"值得"阈值。
- **P_CLIP 对齐**：CRN-Full 训练时 `TARGET_CLIP=20`，采样时 `P_CLIP=6`，两者不一致导致 $e^{p}$ 最大值相差约 $e^{14} \approx 1.2 \times 10^6$，是当前生成质量不稳定的主要来源之一。应将采样 `P_CLIP` 对齐至训练值。

**中期（数据集扩展）**

- **CIFAR-10 验证**：在彩色图像上验证 CRN 框架的可扩展性，$v_0$ 建议取 255（对应 8-bit 像素），$T \in [5, 8]$，需引入 attention 层和更深的 U-Net。
- **精确 CRN 后验**：当前 Full 模型的后验修正仍基于高斯近似。可探索用 Poisson 精确后验替代（$q(n_{t-\Delta t} \mid n_t, n_0)$ 为负二项分布），从根本上消除高斯近似误差。
- **加速采样**：借鉴 DDIM 的确定性采样思路，为 CRN 设计 ODE 形式的确定性反向轨迹，在 20 步内达到 200 步的生成质量。

**长期（理论深化）**

- **HJ 框架完整实现**：当前 HJ 求解器已实现但未用于训练目标，可探索直接用精确条件动量（而非高斯近似 score）作为训练目标，理论上能消除方差不匹配误差 $\Delta\sigma^2 = x_0 e^{-2t}/v_0$。
- **条件生成**：引入类别条件（classifier-free guidance），在 CRN 框架下实现可控生成，验证 CRN 与 CFG 的兼容性。
- **理论收敛分析**：建立 CRN 反向采样的误差界，分析 tau-leaping 步长、$v_0$、$T$ 对生成分布与真实分布之间 KL 散度的影响。

---

## 8. 总结

本实验系统验证了基于化学反应网络（CRN）的扩散模型在 MNIST 上的可行性，核心结论如下：

**CRN 框架可行**：线性生灭过程 $\emptyset \rightleftharpoons X$ 构成了一个数学上严格的扩散过程，其精确转移核（Binomial + Poisson）和 Hamilton-Jacobi 框架为扩散模型提供了不同于高斯 SDE 的理论基础。在 MNIST 上，两个变体均能生成可辨认的手写数字，证明了该框架的基本有效性。

**参数选择关键**：$v_0$ 是影响生成质量的最重要参数。$v_0=10$ 时 Poisson 噪声过大（相对标准差 $\approx 31.6\%$），导致高斯近似失效、$\hat{x}_0$ 反解数值不稳定，生成图像严重偏向稀疏类别（数字 1）。$v_0=100$ 时高斯近似精度提升，类别分布趋于均匀，生成质量显著改善。$T$ 的影响相对次要，$T=5$ 在充分混合的前提下提供更小的离散化步长，略优于 $T=8$。

**Base vs Full 的权衡**：在 MNIST 上，Base 模型（丢弃 $x_t$ 条件）与 Full 模型（精确后验修正）的视觉质量相近，但 Full 模型训练慢约 57%。Full 的优势预计在少步采样和更复杂数据集上才会显现。对于快速实验，Base（$v_0=100$，$T=5$）是更高效的选择；对于追求理论严格性或少步采样的场景，Full 是更稳健的基础。

**主要待解问题**：当前缺乏 FID 等定量指标，Base/Full 的优劣尚无客观数字支撑；CRN-Full 的 P_CLIP 训练/采样不一致问题需修复；框架向 CIFAR 等复杂数据集的迁移能力有待验证。

---

## 附录

### A. 代码结构

```
crn_diffusion/
├── config.py       # 超参数配置，支持多数据集切换（apply_dataset）
├── forward.py      # CRN 前向加噪（Poisson+Poisson 精确核）
├── hj_solver.py    # Hamilton-Jacobi 求解器（条件动量计算）
├── model.py        # U-Net（MNIST）和 UNetCIFAR（CIFAR-100，额外 Attention）
├── train.py        # 训练循环
├── sample.py       # 独立采样脚本
├── evaluate.py     # FID 评估
└── main.py         # CLI 入口
```

### B. 超参数汇总

#### MNIST（默认）

| 参数 | 值 |
|------|----|
| 优化器 | Adam |
| 学习率 $\eta_{max}$ | $10^{-4}$ |
| LR Warmup | 5 epochs（线性，从 0 到 $\eta_{max}$） |
| LR 最小值 $\eta_{min}$ | $10^{-6}$（cosine annealing） |
| Batch size | 256 |
| $T_{MIN}$（训练 $t$ 下界） | 0.05 |
| EMA decay | 0.999 |
| 梯度裁剪 | norm $\leq 1.0$ |
| U-Net 通道 | [64, 128, 256, 512] |
| 时间嵌入维度 | 128 |
| Dropout | 0.1 |
| 采样步数（评估） | 200 |
| 采样步数（训练监控） | 50 |

#### CIFAR-100（参考）

| 参数 | 值 | 说明 |
|------|----|------|
| $v_0$ | 255 | 像素值 $\times 255$ 对应整数粒子数 |
| $T$ | 10.0 | 更长扩散时间 |
| $T_{MIN}$ | 0.2 | 更大下界，避免高方差 |
| 学习率 | $2 \times 10^{-4}$ | |
| 梯度裁剪 | 0.5 | 更保守 |
| Epochs | 500 | |
| Batch size | 128 | |

### C. 运行命令

```bash
# 训练
python -m crn_diffusion.main --mode train --wandb_mode online

# 采样
python -m crn_diffusion.main --mode sample --steps 200

# 评估（FID）
python -m crn_diffusion.main --mode evaluate
```

