"""
CRN 前向加噪过程：∅ ⇌ X（最简复平衡 CRN）。

反应：
    birth  ∅ → X   速率: v0
    death  X → ∅   速率: x (per unit, i.e., per-particle rate = 1)

精确转移：对线性生灭过程，给定 n_0 粒子数，t 时刻分布为：
    n_t = Bin(n_0, e^{-t}) + Poisson(v0*(1 - e^{-t}))

在大 v0 极限下，Bin(n_0, e^{-t}) ≈ Poisson(n_0 * e^{-t})，所以：
    x_t = (Poisson(x_0 * v0 * e^{-t}) + Poisson(v0*(1-e^{-t}))) / v0

归一化后 x = n / v0，平稳分布均值 = 1。
"""
import torch
from . import config


class CRNForwardProcess:
    """
    前向 CRN noising：给定 x0 ∈ [0,1]，精确采样任意时刻 t 的状态。

    使用线性生灭过程的精确转移分布（Poisson近似Binomial，对大v0精确）：
        xt = (Poisson(v0 * x0 * e^{-t}) + Poisson(v0 * (1 - e^{-t}))) / v0
    """

    def __init__(self, v0=None, dt=None, device="cuda"):
        self.v0 = v0 or config.Config.V0
        self.dt = dt or config.Config.DT
        self.device = device

    def __call__(self, x0, t):
        """
        给定 x0 和时间 t，精确采样 xt。

        Args:
            x0: (B, 1, 28, 28) 或兼容 shape，像素 ∈ [0, 1]
            t:  标量或 (B,)

        Returns:
            xt: same shape as x0，值域 [0, ∞)
        """
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
        if x0.dim() == 2:
            x0 = x0.view(x0.size(0), 1, config.Config.IMG_SIZE, config.Config.IMG_SIZE)

        B = x0.size(0)
        if torch.is_tensor(t):
            t = t.to(x0.device)
            if t.dim() == 0:
                t = t.expand(B)
        else:
            t = torch.full((B,), float(t), dtype=x0.dtype, device=x0.device)

        t = t.view(B, 1, 1, 1)
        et = torch.exp(-t)

        # Surviving original particles: Poisson(v0 * x0 * e^{-t})
        surviving_rate = (self.v0 * x0 * et).clamp(min=0.0)
        surviving = torch.poisson(surviving_rate)

        # New particles: Poisson(v0 * (1 - e^{-t}))
        new_rate = (self.v0 * (1.0 - et)).clamp(min=0.0)
        new_particles = torch.poisson(new_rate.expand_as(x0))

        xt = (surviving + new_particles) / self.v0
        return xt

    def trajectory(self, x0, t_vals):
        """
        返回从 x0 到 t_vals 中每个时间点的精确样本。

        Args:
            x0: (B, 1, 28, 28)
            t_vals: (K,) 时间点列表，递增

        Returns:
            xs: (B, K+1, 28, 28)，包含 x0 在内
        """
        if x0.dim() == 2:
            x0 = x0.view(x0.size(0), 1, config.Config.IMG_SIZE, config.Config.IMG_SIZE)

        results = [x0]
        for t in t_vals:
            xt = self(x0, t)
            results.append(xt)

        return torch.stack(results, dim=1)


def sample_prior(n, v0=None, device="cuda"):
    """
    从平稳分布 Poisson(v0)/v0 采样（对应 t→∞）。
    """
    v0 = v0 or config.Config.V0
    n = int(n)
    shape = (n, 1, config.Config.IMG_SIZE, config.Config.IMG_SIZE)
    pois = torch.poisson(v0 * torch.ones(shape, device=device))
    return pois.float() / v0


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    crn = CRNForwardProcess(device=device)

    x0 = torch.rand(4, 1, 28, 28, device=device) * 0.3  # MNIST-like dark images
    x_t = crn(x0, t=3.0)
    print(f"x0 range: [{x0.min():.3f}, {x0.max():.3f}], mean={x0.mean():.4f}")
    print(f"x_t range: [{x_t.min():.3f}, {x_t.max():.3f}], mean={x_t.mean():.4f}")
    print(f"Expected x_t mean: {(x0.mean() * (1 - 1/2.718**3) + 1 * (1 - 1/2.718**3)):.4f}")

    # Verify: at t=0, xt should equal x0
    x_t0 = crn(x0, t=0.001)
    print(f"x_t at t≈0 mean: {x_t0.mean():.4f} (should be ≈ {x0.mean():.4f})")

    # Verify: at t→∞, xt should be ~ Poisson(v0)/v0, mean=1
    x_tinf = crn(x0, t=20.0)
    print(f"x_t at t=20 mean: {x_tinf.mean():.4f} (should be ≈ 1.0)")
