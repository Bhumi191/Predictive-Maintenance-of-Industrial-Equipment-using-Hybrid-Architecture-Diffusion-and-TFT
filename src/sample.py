
import numpy as np
import torch

from model import DiT


def load_ema(path="../results/dit_fd001.pt"):
    ck = torch.load(path, map_location="cpu")
    cfg = ck["cfg"]
    model = DiT(d=cfg["d"], depth=cfg["depth"])
    model.load_state_dict(ck["ema"])
    return model.eval(), cfg


@torch.no_grad()
def ddim_sample(model, c, ab, steps=50, guidance=1.0, n_bins=10,
                L=30, channels=14, clip=1.5):
    """Deterministic DDIM (eta=0). c: (N,) RUL bins. guidance=1.0 -> plain conditional;
    >1 pushes samples toward the condition (classifier-free guidance)."""
    n, T = len(c), len(ab)
    ts = torch.linspace(T - 1, 0, steps).long()
    x = torch.randn(n, L, channels)
    null = torch.full_like(c, n_bins)                      # the "no condition" slot
    for i, t in enumerate(ts):
        tb = torch.full((n,), int(t))
        eps = model(x, tb, c)
        if guidance != 1.0:
            eps_u = model(x, tb, null)
            eps = eps_u + guidance * (eps - eps_u)
        a_t = ab[t]
        x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-clip, clip)
        eps = (x - a_t.sqrt() * x0) / (1 - a_t).sqrt()     # keep eps consistent with clipped x0
        a_prev = ab[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
    return x


def realism_report(real, fake):
    """real, fake: (N, L, C) numpy. Returns per-sensor mean gap, std ratio, correlation gap."""
    r = real.reshape(-1, real.shape[-1])
    f = fake.reshape(-1, fake.shape[-1])
    mean_gap = np.abs(r.mean(0) - f.mean(0))
    std_ratio = f.std(0) / r.std(0)
    corr_gap = np.abs(np.corrcoef(r.T) - np.corrcoef(f.T)).mean()
    return mean_gap, std_ratio, corr_gap


@torch.no_grad()
def ddim_denoise(model, x, c, ab, t_star=75, steps=15, clip=1.5):
    """SDEdit-style denoising: noise x only up to t_star, then run the reverse process to 0.
    x: (N, L, C) real windows, c: (N,) RUL bins. Small t_star = gentle denoise."""
    a0 = ab[t_star]
    xt = a0.sqrt() * x + (1 - a0).sqrt() * torch.randn_like(x)
    ts = torch.linspace(t_star, 0, steps).long()
    for i, t in enumerate(ts):
        tb = torch.full((len(x),), int(t))
        eps = model(xt, tb, c)
        a_t = ab[t]
        x0 = ((xt - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-clip, clip)
        eps = (xt - a_t.sqrt() * x0) / (1 - a_t).sqrt()
        a_prev = ab[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
        xt = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
    return xt
