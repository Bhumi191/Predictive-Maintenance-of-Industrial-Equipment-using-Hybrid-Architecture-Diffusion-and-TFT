"""Day 5: train the DiT on C-MAPSS windows (epsilon-prediction, EMA weights)."""
import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from diffusion import make_schedule, q_sample
from model import DiT, rul_to_bin


@torch.no_grad()
def ema_update(ema, model, step, max_decay=0.999):
    decay = min(max_decay, (1 + step) / (10 + step))   # warm-up so early EMA isn't stuck
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(p.detach(), alpha=1 - decay)


@torch.no_grad()
def val_loss(model, X, y, ab, T, seed=0):
    """Same t and noise every call, so numbers are comparable across epochs."""
    g = torch.Generator().manual_seed(seed)
    t = torch.randint(0, T, (len(X),), generator=g)
    eps = torch.randn(X.shape, generator=g)
    xt, _ = q_sample(X, t, ab, eps)
    model.eval()
    loss = F.mse_loss(model(xt, t, y), eps).item()
    model.train()
    return loss


def train(npz="../data/processed/processed_FD001.npz", epochs=30, bs=128, lr=2e-4,
          d=64, depth=4, T=1000, save="../results/dit_fd001.pt"):
    data = np.load(npz)
    Xtr = torch.tensor(data["Xtr"]); ytr = rul_to_bin(torch.tensor(data["ytr"]))
    Xva = torch.tensor(data["Xva"]); yva = rul_to_bin(torch.tensor(data["yva"]))
    _, ab = make_schedule(T)

    model = DiT(d=d, depth=depth)
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True)

    history, step = [], 0
    for ep in range(epochs):
        total = 0.0
        for x0, y in loader:
            t = torch.randint(0, T, (len(x0),))                # fresh t every batch
            xt, eps = q_sample(x0, t, ab)                       # fresh noise every batch
            loss = F.mse_loss(model(xt, t, y, drop_prob=0.1), eps)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema_update(ema, model, step)
            step += 1
            total += loss.item() * len(x0)
        tr, va = total / len(Xtr), val_loss(ema, Xva, yva, ab, T)
        history.append((tr, va))
        print(f"epoch {ep + 1:3d} | train {tr:.4f} | val (EMA) {va:.4f}")

    os.makedirs(os.path.dirname(save), exist_ok=True)
    torch.save({"ema": ema.state_dict(), "cfg": dict(d=d, depth=depth, T=T)}, save)
    print("saved", save)
    return ema, history