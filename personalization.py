import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import NUM_CLASSES, HIDDEN_SIZE, INPUT_SIZE

PROFILES_DIR = Path("profiles")


class UserAdapter(nn.Module):
    def __init__(self, dim=HIDDEN_SIZE, bottleneck=16, n_cls=NUM_CLASSES):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.bias = nn.Parameter(torch.zeros(n_cls))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, emb):
        return emb + self.up(F.gelu(self.down(emb)))


class PersonalizedModel:
    def __init__(self, base, device):
        self.base = base.eval().requires_grad_(False)
        self.device = device
        self.adapter = UserAdapter().to(device)
        self.feat_mean = np.zeros(INPUT_SIZE, np.float32)
        self.buffer = []  # (seq[T,F], label) для дообучения

    @torch.no_grad()
    def predict_logits(self, seq):
        x = torch.from_numpy(self._norm(seq)).float().unsqueeze(0).to(self.device)
        emb = self.base.embed(x)
        return self.base.classify(self.adapter(emb)) + self.adapter.bias

    def _norm(self, seq):
        seq = np.asarray(seq, np.float32).copy()
        mask = seq.any(axis=1)  # нулевые кадры (нет руки) не трогаем
        seq[mask] -= self.feat_mean
        return seq

    def calibrate_features(self, seqs):
        # среднее смещение признаков человека относительно "эталона" (0)
        frames = np.concatenate([s[s.any(axis=1)] for s in seqs])
        self.feat_mean = (frames.mean(0) * 0.5).astype(np.float32)  # 0.5 = демпфирование

    def add_sample(self, seq, label_idx):
        self.buffer.append((np.asarray(seq, np.float32), label_idx))

    def fit(self, steps=30, lr=1e-3, l2=1e-2, batch=16):
        if len(self.buffer) < 5:
            return
        X = torch.from_numpy(np.stack([self._norm(s) for s, _ in self.buffer])).float().to(self.device)
        y = torch.tensor([l for _, l in self.buffer], device=self.device)
        with torch.no_grad():
            emb = self.base.embed(X)  # база заморожена: эмбеддинги считаем один раз
        params = list(self.adapter.parameters())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        self.adapter.train()
        for _ in range(steps):
            idx = torch.randperm(len(y))[:batch]
            logits = self.base.classify(self.adapter(emb[idx])) + self.adapter.bias
            reg = sum((p ** 2).sum() for p in params)  # якорь к нулю = к базовой модели
            loss = F.cross_entropy(logits, y[idx], label_smoothing=0.1) + l2 * reg
            opt.zero_grad(); loss.backward(); opt.step()
        self.adapter.eval()

    def save(self, user):
        PROFILES_DIR.mkdir(exist_ok=True)
        torch.save(self.adapter.state_dict(), PROFILES_DIR / f"{user}.pt")
        np.save(PROFILES_DIR / f"{user}_mean.npy", self.feat_mean)

    def load(self, user):
        p = PROFILES_DIR / f"{user}.pt"
        if p.exists():
            self.adapter.load_state_dict(torch.load(p, map_location=self.device))
            self.feat_mean = np.load(PROFILES_DIR / f"{user}_mean.npy")