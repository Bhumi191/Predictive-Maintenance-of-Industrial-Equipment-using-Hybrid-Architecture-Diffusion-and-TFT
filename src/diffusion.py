"""Noise schedule and forward noising (torch version of the Day 1 step 8)."""
import torch


def make_schedule(T=1000, b0=1e-4, b1=0.02):
    betas = torch.linspace(b0, b1, T)
    alpha_bar = torch.cumprod(1 - betas, 0)
    return betas, alpha_bar


def q_sample(x0, t, alpha_bar, eps=None):
    """x_t = sqrt(ab_t) * x0 + sqrt(1 - ab_t) * eps.  x0: (B, L, C), t: (B,)"""
    if eps is None:
        eps = torch.randn_like(x0)
    a = alpha_bar[t][:, None, None]
    return a.sqrt() * x0 + (1 - a).sqrt() * eps, eps