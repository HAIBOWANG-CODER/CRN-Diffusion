"""
Hamilton-Jacobi 反向求解器：给定 (x0, xt, t) → 条件动量 pt。

核心方程（∅ ⇌ X CRN）：
    H(p, x) = (e^p - 1) + (e^{-p} - 1) * x
    Hamilton-Jacobi:  ∂t S + H(∇S, x) = 0
    WKB:              Q_t(x|x0) ~ exp(-v0 * S)
    条件动量:         p_t = ∇_xt log Q_t = -v0 * ∇_xt S

沿特征曲线，H(p, x) = E（能量守恒），p_t 满足：
    u² - (1 + E + x_t) * u + x_t = 0,   u = e^{p_t}
    → u = (1 + E + x_t + σ√((1+E+x_t)² - 4x_t)) / 2,   σ ∈ {+1, -1}

时间积分（来自 ṗ = 1 - e^{-p}）：
    log((u_t - 1) / (u_0 - 1)) = t

整体：对每个坐标，求解一维方程 f(E) = 0 → 得到 E → 计算 u_t = e^{p_t}。
"""
import torch
import torch.nn.functional as F
from . import config


def _u_sigma(x, E, sigma):
    """
    二次方程的根 u = e^p。

    u = (1 + E + x + σ * sqrt((1+E+x)² - 4x)) / 2

    Args:
        x:     (B, D) 或 (B, 1, H, W)
        E:     (B, D)  与 x 必须 shape 完全一致
        sigma: scalar int  +1 或 -1

    Returns:
        u:     same shape as x
    """
    assert x.shape == E.shape, f"shape mismatch in _u_sigma: x={x.shape}, E={E.shape}"
    disc = (1 + E + x) ** 2 - 4 * x
    disc = torch.clamp(disc, min=1e-12)
    sqrt_disc = torch.sqrt(disc)
    return (1 + E + x + sigma * sqrt_disc) * 0.5


def _select_branch(x0, xt):
    """
    分支选择：决定 u_t = e^{p_t} 取二次方程的哪个根。

    反向过程中 p_t 驱动 xt → x0：
      - x0 < xt（需要减小 x）→ p_t < 0 → u_t < 1 → σ=-1（小根）
      - x0 >= xt（需要增大 x）→ p_t >= 0 → u_t >= 1 → σ=+1（大根）

    xt 可以超过 1（前向 CRN 平稳分布均值为 1，允许 xt > 1）。
    对 MNIST（x0 ∈ [0,1]，xt 扩散后均值趋向 1），大多数像素 x0 < xt。
    """
    return torch.where(x0 < xt, -torch.ones_like(x0), torch.ones_like(x0))


def _energy_equation(E, x0, xt, t, sigma):
    """
    f(E) = log((u_t - 1) / (u_0 - 1)) - t = 0

    All tensors are kept as (B, D) throughout so that Newton updates
    (f / grad) stay shape-consistent and t broadcasts correctly.

    Args:
        E, x0, xt, sigma: (B, D)
        t:                (B, 1)  — broadcasts over D

    Returns:
        f: (B, D)
    """
    # Normalise to (B, D) — inputs arrive already flattened from solve_batch_energy
    def _to_2d(x):
        if x.dim() > 2:
            return x.flatten(start_dim=1)
        if x.dim() == 1:
            return x.unsqueeze(0)
        return x  # already (B, D)

    x0_2d    = _to_2d(x0)
    xt_2d    = _to_2d(xt)
    E_2d     = _to_2d(E)
    sigma_2d = _to_2d(sigma)

    # t must be (B, 1) so it broadcasts over D
    if t.dim() == 0:
        t_bc = t
    elif t.dim() == 1:
        t_bc = t.unsqueeze(1)   # (B,) → (B, 1)
    else:
        t_bc = t.view(t.size(0), 1)

    assert xt_2d.shape == E_2d.shape, f"_energy_equation: xt={xt_2d.shape}, E={E_2d.shape}"
    assert x0_2d.shape == E_2d.shape, f"_energy_equation: x0={x0_2d.shape}, E={E_2d.shape}"
    assert xt_2d.shape == sigma_2d.shape, f"_energy_equation: xt={xt_2d.shape}, sigma={sigma_2d.shape}"

    u_t = _u_sigma(xt_2d, E_2d, sigma_2d)
    u_0 = _u_sigma(x0_2d, E_2d, sigma_2d)

    # u(t) = 1 + (u_0 - 1)*e^t, so (u_t-1) and (u_0-1) always share the same sign.
    # When u_0 ≈ 1 (p_0 ≈ 0, fixed point of the ODE), both sides are ~0 and the
    # equation is trivially satisfied with E=0 — treat f as 0 to avoid NaN from log(0/0).
    u0_minus_1 = u_0 - 1.0
    ut_minus_1 = u_t - 1.0
    degenerate = u0_minus_1.abs() < 1e-8

    numerator   = ut_minus_1.abs().clamp(min=1e-12)
    denominator = u0_minus_1.abs().clamp(min=1e-12)
    log_ratio = torch.log(numerator / denominator)

    f = log_ratio - t_bc
    # At the degenerate fixed point, f should be 0 (E=0 is the solution)
    f = torch.where(degenerate, torch.zeros_like(f), f)

    return f  # (B, D)


def _energy_grad(E, x0, xt, t, sigma):
    """
    df/dE，用自动微分计算（向量化 batch）。
    """
    with torch.enable_grad():
        E_ = E.detach().requires_grad_(True)
        f = _energy_equation(E_, x0, xt, t, sigma)
        grad = torch.autograd.grad(
            outputs=f.sum(),
            inputs=E_,
            retain_graph=False,
        )[0]
    return grad


def solve_batch_energy(x0, xt, t, sigma, max_iter=50, tol=1e-6):
    """
    批量 Newton 法求解能量 E。

    每个坐标独立求解同一类型的一维方程，
    全部并行（torch 向量化）。

    Args:
        x0:    (B, 784) 或 (B, 28, 28)
        xt:    同 shape
        t:     scalar 或 (B,)
        sigma: (B, 784) 或 (B, 28, 28)，每坐标独立
        max_iter: Newton 最大迭代次数
        tol:    收敛阈值（绝对值 f(E)）

    Returns:
        E:     same shape as x0, xt
    """
    x0 = x0.flatten(start_dim=1) if x0.dim() > 2 else x0
    xt = xt.flatten(start_dim=1) if xt.dim() > 2 else xt
    sigma = sigma.flatten(start_dim=1) if sigma.dim() > 2 else sigma

    B, D = x0.shape

    if t.dim() == 0:
        t = t.expand(B)
    t = t.view(B, 1)

    E = torch.zeros((B, D), dtype=x0.dtype, device=x0.device)

    for iteration in range(max_iter):
        with torch.no_grad():
            f = _energy_equation(E, x0, xt, t, sigma)
            if torch.all(torch.abs(f) < tol):
                break

        grad = _energy_grad(E, x0, xt, t, sigma)
        grad = torch.where(torch.abs(grad) < 1e-8, torch.ones_like(grad) * 1e-8, grad)
        delta = f / grad
        # 限制每步步长防止 Newton 发散
        delta = delta.clamp(-1.0, 1.0)

        E = E - delta
        # E lower bound: H(p,x) ≥ -x at p=0. For xt that can exceed 1,
        # we need E ≥ -max(xt). Use -10 as a safe lower bound.
        E = E.clamp(min=-10.0, max=50.0)
        # 如果出现 NaN 就重置为 0
        E = torch.where(torch.isfinite(E), E, torch.zeros_like(E))

    with torch.no_grad():
        f_final = _energy_equation(E, x0, xt, t, sigma)
        not_converged = torch.abs(f_final) > tol * 10

        if torch.any(not_converged):
            E_fallback = _rough_energy_estimate(x0, xt, t, sigma)
            E = torch.where(not_converged, E_fallback, E)

    return E


def _rough_energy_estimate(x0, xt, t, sigma):
    """
    当 Newton 失败时的粗糙 E 估计，作为后备。
    用泰勒展开到一阶：E ≈ 2 * (x0 - xt) / t（仅对小 t 有效）。
    对大 t，平稳分布 xt ≈ 1，方向由 x0 决定：
      - σ=+1 (x0 >= xt, need to grow): E ≈ 2*(x0 - xt)
      - σ=-1 (x0 < xt, need to shrink): E ≈ 2*(xt - x0) scaled by xt
    """
    x0_2d = x0.flatten(start_dim=1) if x0.dim() > 2 else x0
    xt_2d = xt.flatten(start_dim=1) if xt.dim() > 2 else xt
    # t arrives as (B, 1) from solve_batch_energy; keep it that way for broadcasting
    t_bc = t.view(t.size(0), 1) if t.dim() >= 1 else t
    small_t_approx = 2.0 * (x0_2d - xt_2d) / (t_bc.clamp(min=1e-6))
    # For large t, xt is near stationary (mean=1). A rough guess: E ≈ 2*(x0 - 1)
    large_t_approx = torch.clamp(2.0 * (x0_2d - 1.0), min=-9.0, max=49.0)
    return torch.where(t_bc > 0.5, large_t_approx, small_t_approx)


def conditional_momentum(x0, xt, t, v0=None):
    """
    条件动量 p_t = ∇_{x_t} log Q_t(x_t | x_0)。

    这个 CRN（∅ ⇌ X，birth=v0，death=x per unit）是线性生灭过程。
    给定 x_0，在时刻 t 的条件分布为：
        n_t | n_0 ~ Bin(n_0, e^{-t}) + Poisson(v0*(1 - e^{-t}))

    在大 v0 极限下近似为 Gaussian：
        x_t | x_0 ~ N(μ, σ²)
        μ(t)  = x_0 * e^{-t} + (1 - e^{-t})  =  1 - (1-x_0)*e^{-t}
        σ²(t) = (1-e^{-t}) * (x_0*e^{-t} + 1) / v0

    Gaussian score：
        p_t = ∂/∂x_t  log N(x_t; μ, σ²)
            = -(x_t - μ) / σ²
            = -(x_t - μ) * v0 / [(1-e^{-t}) * (x_0*e^{-t}+1)]

    Args:
        x0: (B, 1, H, W)，原始图像像素 ∈ [0, 1]
        xt: (B, 1, H, W)，t 时刻 CRN 状态（可 > 1）
        t:  scalar 或 (B,)，扩散时间
        v0: 覆盖 config.Config.V0

    Returns:
        pt: (B, 1, H, W)，条件动量
    """
    _v0 = v0 if v0 is not None else config.Config.V0
    H = W = config.Config.IMG_SIZE

    def _to_4d(x):
        if x.dim() == 2:
            return x.view(x.size(0), 1, H, W)
        if x.dim() == 3:
            return x.unsqueeze(1)
        return x

    x0 = _to_4d(x0)
    xt = _to_4d(xt)

    # Broadcast t to (B, 1, 1, 1)
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=x0.dtype, device=x0.device)
    t = t.to(x0.device)
    if t.dim() == 0:
        t = t.expand(x0.size(0))
    t = t.view(-1, 1, 1, 1)

    et = torch.exp(-t)                           # e^{-t}
    mu = x0 * et + (1.0 - et)                   # conditional mean
    # variance in x-space (divide by v0 for x = n/v0 scaling)
    var = (1.0 - et) * (x0 * et + 1.0) / _v0
    var = var.clamp(min=1e-8)                    # prevent division by zero

    pt = -(xt - mu) / var
    return pt


def marginal_momentum(xt, t, v0=None):
    """
    计算边际动量 E[p_t | x_t]。
    在训练中作为回归目标（通过对 x0 边缘化）。
    目前用单个确定性近似：取条件动量中的 σ=+1 分支。

    严格的做法需要对每个 xt 积分 over x0，但计算量太大。
    DDPM 中类似的近似（用确定性 reverse SDE）被证明是有效的。

    Args:
        xt: (B, 28, 28) 或 (B, 784)
        t:  scalar 或 (B,)

    Returns:
        pt: same shape
    """
    if xt.dim() == 2 and xt.shape[-1] == 784:
        xt = xt.view(xt.size(0), 1, config.Config.IMG_SIZE, config.Config.IMG_SIZE)

    sigma_plus = torch.ones_like(xt)
    E_approx = torch.clamp(2.0 * (1.0 - xt), min=-9.0, max=50.0)
    pt = torch.log(_u_sigma(xt, E_approx, 1).clamp(min=1e-8))

    return pt


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    B, C, H, W = 4, 1, 28, 28
    x0 = torch.rand(B, C, H, W, device=device) * 0.8
    xt = torch.rand(B, C, H, W, device=device) * 0.8
    t = torch.full((B,), 2.0, device=device)

    pt = conditional_momentum(x0, xt, t)
    print(f"x0 range: [{x0.min():.3f}, {x0.max():.3f}]")
    print(f"xt range: [{xt.min():.3f}, {xt.max():.3f}]")
    print(f"pt range: [{pt.min():.3f}, {pt.max():.3f}]")
    print(f"pt mean:  {pt.mean():.4f}")
    print(f"pt shape: {pt.shape}")

    sigma = _select_branch(x0, xt)
    print(f"sigma unique values: {sigma.unique()}")
    print(f"sigma=-1 count: {(sigma==-1).sum().item()}, sigma=+1 count: {(sigma==1).sum().item()}")

    E_test = solve_batch_energy(x0, xt, t, sigma, max_iter=30)
    print(f"\nE range: [{E_test.min():.3f}, {E_test.max():.3f}]")

    f_check = _energy_equation(E_test, x0, xt, t, sigma)
    print(f"Final |f(E)| max: {torch.abs(f_check).max():.2e}")
