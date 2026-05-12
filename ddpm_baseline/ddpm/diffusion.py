"""
DDPM 前向加噪 + 反向采样（DDPM & DDIM）。

前向：q(x_t | x_0) = N(x_t; sqrt(ᾱ_t)*x_0, (1-ᾱ_t)*I)
反向：p_θ(x_{t-1}|x_t) 用 U-Net 预测噪声 ε_θ，还原 x_{t-1}
DDIM：确定性快速采样，eta=0
"""
import torch
import torch.nn.functional as F
from . import config


class GaussianDiffusion:
    def __init__(self, device="cpu"):
        cfg = config.Config
        self.T      = cfg.T
        self.device = device

        # Linear beta schedule
        betas  = torch.linspace(cfg.BETA_START, cfg.BETA_END, self.T, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)        # ᾱ_t,  shape (T,)

        self.betas      = betas
        self.alphas     = alphas
        self.alpha_bar  = alpha_bar                     # ᾱ_t
        self.alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)  # ᾱ_{t-1}

        # Precompute quantities used in forward & reverse
        self.sqrt_alpha_bar       = alpha_bar.sqrt()
        self.sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()

        # Reverse process posterior variance β̃_t
        self.posterior_variance = betas * (1.0 - self.alpha_bar_prev) / (1.0 - alpha_bar)
        self.posterior_log_var  = torch.log(self.posterior_variance.clamp(min=1e-20))

        # Coefficients for x_0 reconstruction from x_t and ε
        self.sqrt_recip_alpha_bar      = (1.0 / alpha_bar).sqrt()
        self.sqrt_recip_m1_alpha_bar   = (1.0 / alpha_bar - 1.0).sqrt()

    def _extract(self, arr, t, shape):
        """Index arr with batch of timesteps t, broadcast to shape."""
        vals = arr[t - 1]           # t is 1-indexed
        while vals.dim() < len(shape):
            vals = vals.unsqueeze(-1)
        return vals.expand(shape)

    # ── Forward process ──────────────────────────────────────────

    def q_sample(self, x0, t, noise=None):
        """
        Sample x_t given x_0:  x_t = sqrt(ᾱ_t)*x_0 + sqrt(1-ᾱ_t)*ε
        t: (B,) integer tensor in {1,...,T}
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab  = self._extract(self.sqrt_alpha_bar,           t, x0.shape)
        sqrt_1ab = self._extract(self.sqrt_one_minus_alpha_bar, t, x0.shape)
        return sqrt_ab * x0 + sqrt_1ab * noise

    # ── Reverse helpers ──────────────────────────────────────────

    def _predict_x0(self, x_t, t, eps_pred):
        """Reconstruct x_0 from x_t and predicted ε."""
        r1 = self._extract(self.sqrt_recip_alpha_bar,    t, x_t.shape)
        r2 = self._extract(self.sqrt_recip_m1_alpha_bar, t, x_t.shape)
        return r1 * x_t - r2 * eps_pred

    # ── DDPM reverse step ────────────────────────────────────────

    def p_sample(self, model, x_t, t_tensor):
        """
        One DDPM reverse step: sample x_{t-1} given x_t.
        t_tensor: (B,) integer tensor, same value for all elements.
        """
        eps_pred = model(x_t, t_tensor)
        x0_pred  = self._predict_x0(x_t, t_tensor, eps_pred).clamp(-1.0, 1.0)

        # Posterior mean  μ̃_t = coef1 * x_0 + coef2 * x_t
        ab    = self._extract(self.alpha_bar,      t_tensor, x_t.shape)
        ab_p  = self._extract(self.alpha_bar_prev, t_tensor, x_t.shape)
        b     = self._extract(self.betas,          t_tensor, x_t.shape)
        denom = 1.0 - ab

        coef1 = ab_p.sqrt() * b / denom
        coef2 = self.alphas[t_tensor[0] - 1].sqrt() * (1.0 - ab_p) / denom
        mu    = coef1 * x0_pred + coef2 * x_t

        # Add noise only when t > 1
        log_var = self._extract(self.posterior_log_var, t_tensor, x_t.shape)
        noise   = torch.randn_like(x_t)
        mask    = (t_tensor > 1).float().view(-1, 1, 1, 1)
        return mu + mask * (0.5 * log_var).exp() * noise

    # ── DDIM reverse step (deterministic, eta=0) ─────────────────

    def ddim_step(self, model, x_t, t, t_prev):
        """
        One DDIM step from timestep t to t_prev.
        t, t_prev: Python ints
        """
        B    = x_t.shape[0]
        t_b  = torch.full((B,), t,      device=x_t.device, dtype=torch.long)
        tp_b = torch.full((B,), t_prev, device=x_t.device, dtype=torch.long)

        eps_pred = model(x_t, t_b)
        x0_pred  = self._predict_x0(x_t, t_b, eps_pred).clamp(-1.0, 1.0)

        ab   = self.alpha_bar[t - 1]
        ab_p = self.alpha_bar[t_prev - 1] if t_prev >= 1 else torch.tensor(1.0, device=x_t.device)

        # Deterministic DDIM direction (eta=0)
        x_t_prev = ab_p.sqrt() * x0_pred + (1.0 - ab_p).sqrt() * eps_pred
        return x_t_prev

    # ── Full sampling loops ───────────────────────────────────────

    @torch.no_grad()
    def sample(self, model, shape, steps=None, use_ddim=False):
        """
        Generate samples from Gaussian noise.
        shape: (B, C, H, W)
        steps: number of sampling steps (default = T for DDPM, or subset for DDIM)
        use_ddim: True → DDIM (deterministic), False → DDPM
        """
        device = next(model.parameters()).device
        x = torch.randn(shape, device=device)
        model.eval()

        if use_ddim:
            steps = steps or config.Config.SAMPLE_STEPS_QUICK
            # Uniform subset of timesteps from T down to 1
            ts = list(range(self.T, 0, -self.T // steps))[:steps]
            ts_prev = ts[1:] + [0]
            for t, t_prev in zip(ts, ts_prev):
                t_prev_safe = max(t_prev, 1)
                x = self.ddim_step(model, x, t, t_prev_safe)
                if t_prev == 0:
                    # Final step: no noise, return denoised
                    B = x.shape[0]
                    t_b = torch.full((B,), 1, device=device, dtype=torch.long)
                    eps = model(x, t_b)
                    x = self._predict_x0(x, t_b, eps).clamp(-1.0, 1.0)
                    break
        else:
            steps = steps or self.T
            ts = list(range(self.T, 0, -self.T // steps))[:steps]
            for t in ts:
                t_b = torch.full((shape[0],), t, device=device, dtype=torch.long)
                x = self.p_sample(model, x, t_b)

        return x
