from typing import Optional, Tuple

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "model.py requires PyTorch. Install it with `pip install torch` before running the demo."
    ) from exc


def _as_time_column(t: Tensor, like: Tensor) -> Tensor:
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=like.device, dtype=like.dtype)
    t = t.to(device=like.device, dtype=like.dtype)

    if t.ndim == 0:
        t = t.expand(like.shape[0])

    while t.ndim < like.ndim:
        t = t.unsqueeze(-1)

    return t


@torch.no_grad()
def quantize_to_counts(x: Tensor, v0: float = 100.0) -> Tensor:
    """
    Convert normalized continuous data x into the CRN lattice n/v0.

    For exact CRN dynamics, the state should be a count variable n.
    This function maps x -> round(v0*x)/v0.
    """
    n = torch.round((v0 * x).clamp_min(0.0))
    return n / v0


@torch.no_grad()
def forward_tau_leap(
    x0: Tensor,
    s: Tensor,
    v0: float = 100.0,
    n_steps: int = 100,
) -> Tensor:
    """
    Sample x_s from the forward birth-death CRN ∅ ⇌ X.

    This implementation keeps the tau-leaping spirit, but for this linear
    birth-death CRN uses a stable exact substep:
        n_{t+dt} = Binomial(n_t, exp(-dt)) + Poisson(v0 * (1 - exp(-dt))).

    This avoids the negative-count bias caused by sampling deaths as an
    unrestricted Poisson variable and then clamping.
    """
    x0 = quantize_to_counts(x0, v0=v0)
    n = torch.round((v0 * x0).clamp_min(0.0))

    s_col = _as_time_column(s, x0).expand_as(x0)
    dt = s_col / float(n_steps)

    for _ in range(n_steps):
        survival_prob = torch.exp(-dt).clamp(0.0, 1.0)
        survival_prob = survival_prob.expand_as(n)

        birth_mean = (v0 * (1.0 - torch.exp(-dt))).clamp_min(0.0)
        birth_mean = birth_mean.expand_as(n)

        survivors = torch.distributions.Binomial(
            total_count=n,
            probs=survival_prob,
        ).sample()

        births = torch.poisson(birth_mean)

        n = (survivors + births).clamp_min(0.0)

    return n / v0


def conditional_momentum(
    x0: Tensor,
    xs: Tensor,
    s: Tensor,
    max_iter: int = 80,
    tol: float = 1e-7,
    delta_floor: float = 1e-6,
    p_clip: float = 20.0,
) -> Tensor:
    """
    HJ conditional momentum for the birth-death CRN

        H(p,x) = e^p - 1 + x(e^{-p} - 1).

    Momentum convention:
        p_t = ∇_x S_t(x_t | x_0)
            = - (1/v0) ∇_{x_t} log Q_t(x_t | x_0)

    This function solves for terminal p_t using the characteristic equations.

    Instead of using the energy-branch formula directly, we solve a more stable
    scalar equation in delta = exp(p_t) - 1.

    Let a = exp(-s), u_t = exp(p_t) = 1 + delta, and
        u_0 = 1 + a delta.

    Along characteristics,
        u(s) - 1 = (u_t - 1) exp(s - t).

    The endpoint relation reduces to

        xs / (1 + delta)
        - a x0 / (1 + a delta)
        - (1 - a)
        = 0.

    The typical path corresponds to delta = 0, i.e.
        xs = 1 + (x0 - 1) exp(-s),
    and hence p_t = 0.
    """
    x0, xs = torch.broadcast_tensors(x0, xs)
    s = _as_time_column(s, xs).expand_as(xs)

    dtype = xs.dtype
    device = xs.device

    a = torch.exp(-s)
    mu = 1.0 + (x0 - 1.0) * a

    def residual(delta: Tensor) -> Tensor:
        # delta must satisfy delta > -1.
        denom_t = (1.0 + delta).clamp_min(delta_floor)
        denom_0 = (1.0 + a * delta).clamp_min(delta_floor)
        return xs / denom_t - a * x0 / denom_0 - (1.0 - a)

    g0 = xs - mu

    near_typical = g0.abs() < tol

    delta = torch.zeros_like(xs)

    # Case 1: xs > typical path -> p_t > 0 -> delta > 0.
    pos_mask = g0 > tol
    if pos_mask.any():
        left = torch.zeros_like(xs)
        right = torch.ones_like(xs)

        f_right = residual(right)

        # Expand right until residual(right) < 0 for positive cases.
        for _ in range(80):
            need_expand = pos_mask & (f_right > 0.0)
            if not need_expand.any():
                break
            right = torch.where(need_expand, right * 2.0, right)
            f_right = residual(right)

        # Bisection on [0, right].
        l = left.clone()
        r = right.clone()
        for _ in range(max_iter):
            mid = 0.5 * (l + r)
            f_mid = residual(mid)

            # For pos case: residual(0)>0, residual(root)=0, residual(right)<0.
            go_right = f_mid > 0.0
            l = torch.where(pos_mask & go_right, mid, l)
            r = torch.where(pos_mask & (~go_right), mid, r)

            if torch.max((r[pos_mask] - l[pos_mask]).abs()).item() < tol:
                break

        delta_pos = 0.5 * (l + r)
        delta = torch.where(pos_mask, delta_pos, delta)

    # Case 2: xs < typical path -> p_t < 0 -> -1 < delta < 0.
    neg_mask = g0 < -tol
    if neg_mask.any():
        left = torch.full_like(xs, -1.0 + delta_floor)
        right = torch.zeros_like(xs)

        f_left = residual(left)

        # Normally f_left > 0 and f_right < 0.
        # If xs is exactly zero, the root is at delta -> -1, so we use left.
        has_bracket = neg_mask & (f_left > 0.0)

        l = left.clone()
        r = right.clone()

        for _ in range(max_iter):
            mid = 0.5 * (l + r)
            f_mid = residual(mid)

            # For neg case: residual(left)>0, residual(0)<0.
            go_right = f_mid > 0.0
            l = torch.where(has_bracket & go_right, mid, l)
            r = torch.where(has_bracket & (~go_right), mid, r)

            if has_bracket.any():
                if torch.max((r[has_bracket] - l[has_bracket]).abs()).item() < tol:
                    break

        delta_neg = 0.5 * (l + r)

        # No finite bracket means xs is too close to zero; use limiting p -> -inf.
        delta_neg = torch.where(has_bracket, delta_neg, left)

        delta = torch.where(neg_mask, delta_neg, delta)

    delta = torch.where(near_typical, torch.zeros_like(delta), delta)

    # p_t = log(1 + delta): use clamp to avoid log(0) for delta near -1
    safe_floor = max(delta_floor, 100.0 * torch.finfo(xs.dtype).eps)
    one_plus_delta = (1.0 + delta).clamp_min(safe_floor)
    p = torch.log(one_plus_delta)

    # Numerical safeguard.
    p = torch.nan_to_num(p, nan=0.0, posinf=p_clip, neginf=-p_clip)
    p = p.clamp(-p_clip, p_clip)

    return p


class MomentumMLP(nn.Module):
    """Small MLP p_theta(x_s, s)."""

    def __init__(self, dim: int = 1, hidden: int = 128, depth: int = 3):
        super().__init__()

        layers = []
        in_dim = dim + 1

        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.SiLU()]
            in_dim = hidden

        layers.append(nn.Linear(hidden, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, s: Tensor) -> Tensor:
        s_col = _as_time_column(s, x).expand(*x.shape[:-1], 1)
        return self.net(torch.cat([x, s_col], dim=-1))


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x0: Tensor,
    v0: float = 100.0,
    n_tau_steps: int = 100,
    s_eps: float = 0.02,
    T: float = 5.0,
    target_clip: float = 20.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    One denoising-momentum matching step.

    Forward:
        CRN birth-death tau-leaping / exact substep.

    Target:
        HJ conditional momentum p_t = ∇S_t
        under the exact CRN Hamiltonian.

    Network:
        Learns marginal HJ momentum
        E[p_t(x_t|x_0) | x_t].
    """
    model.train()

    x0 = quantize_to_counts(x0, v0=v0)

    s = torch.rand(
        x0.shape[0],
        device=x0.device,
        dtype=x0.dtype,
    ) * (T - s_eps) + s_eps

    with torch.no_grad():
        xs = forward_tau_leap(
            x0,
            s,
            v0=v0,
            n_steps=n_tau_steps,
        )

        target = conditional_momentum(
            x0,
            xs,
            s,
            p_clip=target_clip,
        )

        clip_frac = (target.abs() >= target_clip - 1e-6).float().mean()
        zero_frac = (xs <= 0.0).float().mean()

    pred = model(xs, s)

    loss = F.mse_loss(pred, target)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return loss.detach(), target.detach(), clip_frac.detach(), zero_frac.detach()


@torch.no_grad()
def sample_reverse(
    model: nn.Module,
    shape: Tuple[int, int],
    v0: float = 100.0,
    T: float = 5.0,
    n_steps: int = 100,
    p_clip: float = 20.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Reverse jump sampler initialized by the stationary prior

        n_T ~ Poisson(v0),
        x_T = n_T / v0.

    The model predicts HJ momentum p.

    Backward-time drift corresponds to

        dx/d(-t) = x exp(-p) - exp(p).

    Therefore, in one reverse step:
        birth count  ~ Pois(v0 * x * exp(-p) * tau)
        death count  ~ Pois(v0 * exp(p) * tau)

    We keep the state as counts n internally to avoid non-lattice states.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    n = torch.poisson(
        torch.full(
            shape,
            float(v0),
            device=device,
            dtype=dtype,
        )
    )

    tau = T / float(n_steps)

    for i in reversed(range(n_steps)):
        s_val = (i + 1) * tau
        s = torch.full(
            (shape[0],),
            s_val,
            device=device,
            dtype=dtype,
        )

        x = n / v0

        p = model(x, s)
        p = torch.nan_to_num(p, nan=0.0, posinf=p_clip, neginf=-p_clip)
        p = p.clamp(-p_clip, p_clip)

        birth_rate = (v0 * x * torch.exp(-p) * tau).clamp_min(0.0)
        death_rate = (v0 * torch.exp(p) * tau).clamp_min(0.0)

        births = torch.poisson(birth_rate)
        deaths = torch.poisson(death_rate)

        # Prevent negative counts. This is a practical boundary safeguard.
        deaths = torch.minimum(deaths, n + births)

        n = (n + births - deaths).clamp_min(0.0)

    return n / v0


def demo() -> None:
    try:
        import torchvision
        import torchvision.transforms as T
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "demo() requires torchvision. Install it with `pip install torchvision`."
        ) from exc

    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dim = 28 * 28
    v0 = 100.0

    batch_size = 128
    n_epochs = 5
    n_tau_steps = 30
    n_sample_steps = 200
    T_horizon = 5.0

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Lambda(lambda x: x.view(-1)),
        ]
    )

    dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = MomentumMLP(
        dim=dim,
        hidden=512,
        depth=4,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    step = 0

    for epoch in range(n_epochs):
        for x0, _ in loader:
            x0 = x0.to(device)

            loss, target, clip_frac, zero_frac = train_step(
                model,
                optimizer,
                x0,
                v0=v0,
                n_tau_steps=n_tau_steps,
                T=T_horizon,
                target_clip=20.0,
            )

            if step % 100 == 0:
                print(
                    f"epoch={epoch} "
                    f"step={step:05d} "
                    f"loss={loss.item():.6f} "
                    f"target_min={target.min().item():.3f} "
                    f"target_max={target.max().item():.3f} "
                    f"clip_frac={clip_frac.item():.4f} "
                    f"zero_frac={zero_frac.item():.4f}"
                )

            step += 1

    samples = sample_reverse(
        model,
        shape=(16, dim),
        v0=v0,
        T=T_horizon,
        n_steps=n_sample_steps,
        p_clip=6.0,
        device=device,
    )

    samples = samples.view(16, 1, 28, 28).clamp(0.0, 1.0)

    try:
        torchvision.utils.save_image(
            samples,
            "mnist_samples.png",
            nrow=4,
        )
        print("Saved generated samples to mnist_samples.png")
    except Exception:
        print(
            "samples shape:",
            samples.shape,
            "min/max:",
            samples.min().item(),
            samples.max().item(),
        )


if __name__ == "__main__":
    demo()