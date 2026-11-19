"""Day 2: conditioning pieces for the DiT (timestep + RUL-bin embeddings)."""
import math
import torch
import torch.nn as nn


class TimestepEmbedding(nn.Module):
    """diffusion step t (B,) -> sinusoidal features -> MLP -> (B, d)"""

    def __init__(self, d, freq_dim=128):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)   # (B, freq_dim)
        return self.mlp(emb)                                 # (B, d)


class ConditionEmbedding(nn.Module):
    """RUL bin (B,) -> (B, d). Index n_bins is the 'no condition' slot (for CFG)."""

    def __init__(self, d, n_bins=10):
        super().__init__()
        self.n_bins = n_bins
        self.table = nn.Embedding(n_bins + 1, d)

    def forward(self, c, drop_prob=0.0):
        if drop_prob > 0:   # randomly hide the condition during training
            drop = torch.rand(c.shape[0], device=c.device) < drop_prob
            c = torch.where(drop, torch.full_like(c, self.n_bins), c)
        return self.table(c)


class Conditioner(nn.Module):
    """time vector + condition vector = the vector that drives adaLN in every block."""

    def __init__(self, d, n_bins=10):
        super().__init__()
        self.t_emb = TimestepEmbedding(d)
        self.c_emb = ConditionEmbedding(d, n_bins)

    def forward(self, t, c, drop_prob=0.0):
        return self.t_emb(t) + self.c_emb(c, drop_prob)


def rul_to_bin(rul, n_bins=10, cap=125):
    """RUL in [0, cap] -> integer bin in [0, n_bins-1]."""
    return torch.clamp((rul / cap * n_bins).long(), max=n_bins - 1)


class DiTBlock(nn.Module):
    """Transformer block whose LayerNorms and residuals are steered by the
    conditioning vector c (adaLN-Zero). x: (B, L, d), c: (B, d) -> (B, L, d)"""

    def __init__(self, d, heads=4, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Linear(mlp_ratio * d, d)
        )
        # c -> (gamma1, beta1, alpha1, gamma2, beta2, alpha2)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight)   # "Zero": block starts as identity
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c):
        g1, b1, a1, g2, b2, a2 = self.ada(c).unsqueeze(1).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + g1) + b1
        x = x + a1 * self.attn(h, h, h, need_weights=False)[0]
        h = self.norm2(x) * (1 + g2) + b2
        x = x + a2 * self.mlp(h)
        return x


class DiT(nn.Module):
    """Full denoiser: x_t (B, L, channels), t (B,), c (B,) -> predicted noise (B, L, channels)"""

    def __init__(self, channels=14, L=30, d=64, depth=4, heads=4, n_bins=10):
        super().__init__()
        self.inp = nn.Linear(channels, d)                       # one token per timestep
        self.pos = nn.Parameter(torch.zeros(1, L, d))           # learned positions
        nn.init.normal_(self.pos, std=0.02)
        self.cond = Conditioner(d, n_bins)                      # t + c -> (B, d)
        self.blocks = nn.ModuleList([DiTBlock(d, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))   # final scale/shift
        self.out = nn.Linear(d, channels)                       # d -> channels
        for m in (self.ada[1], self.out):                       # zero-init, as in DiT
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, t, c, drop_prob=0.0):
        h = self.inp(x) + self.pos
        cv = self.cond(t, c, drop_prob)
        for blk in self.blocks:
            h = blk(h, cv)
        shift, scale = self.ada(cv).unsqueeze(1).chunk(2, dim=-1)
        h = self.norm(h) * (1 + scale) + shift
        return self.out(h)