from collections import deque
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import NUM_CLASSES, HIDDEN_SIZE, INPUT_SIZE, SEQUENCE_LENGTH

PROFILES_DIR = Path("profiles")
BUFFER_MAXLEN = 200


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
        self.buffer = deque(maxlen=BUFFER_MAXLEN)  # (seq[T,F], label) для дообучения
        # тензор [1] long с индексом пользователя для user-условной модели; None = <unk>/без фичи
        self.user_idx = None

    def _to_window(self, seq):
        """Приводит последовательность к окну [1, SEQUENCE_LENGTH, F] + маска.

        Как SignDataset: длинные — center-crop, короткие — pad справа, паддинг в маске.
        """
        seq = self._norm(seq)
        n = seq.shape[0]
        if n > SEQUENCE_LENGTH:
            start = (n - SEQUENCE_LENGTH) // 2
            seq = seq[start: start + SEQUENCE_LENGTH]
            mask = np.zeros(SEQUENCE_LENGTH, dtype=bool)
        else:
            seq = np.pad(seq, ((0, SEQUENCE_LENGTH - n), (0, 0)), mode="constant")
            mask = np.concatenate([
                np.zeros(n, dtype=bool),
                np.ones(SEQUENCE_LENGTH - n, dtype=bool),
            ])
        x = torch.from_numpy(seq).float().unsqueeze(0).to(self.device)
        m = torch.from_numpy(mask).bool().unsqueeze(0).to(self.device)
        return x, m

    @torch.no_grad()
    def predict_logits(self, seq):
        x, mask = self._to_window(seq)
        emb = self.base.embed(x, src_key_padding_mask=mask, user_idx=self.user_idx)
        return self.base.classify(self.adapter(emb)) + self.adapter.bias

    def _norm(self, seq):
        seq = np.asarray(seq, np.float32).copy()
        mask = seq.any(axis=1)  # нулевые кадры (нет руки) не трогаем
        seq[mask] -= self.feat_mean
        return seq

    def calibrate_features(self, seqs):
        # среднее смещение признаков человека относительно "эталона" (0)
        frames = [s[s.any(axis=1)] for s in seqs]
        frames = [f for f in frames if len(f) > 0]
        if not frames:
            return
        all_frames = np.concatenate(frames)
        self.feat_mean = (all_frames.mean(0) * 0.5).astype(np.float32)  # 0.5 = демпфирование

    def add_sample(self, seq, label_idx):
        self.buffer.append((np.asarray(seq, np.float32), label_idx))

    def fit(self, steps=30, lr=1e-3, l2=1e-2, batch=16):
        if len(self.buffer) < 5:
            return
        windows = [self._to_window(seq) for seq, _ in self.buffer]
        X = torch.cat([x for x, _ in windows], dim=0)
        mask = torch.cat([m for _, m in windows], dim=0)
        y = torch.tensor([l for _, l in self.buffer], device=self.device)
        uidx = None if self.user_idx is None else self.user_idx.expand(len(y))
        with torch.no_grad():
            emb = self.base.embed(X, src_key_padding_mask=mask, user_idx=uidx)  # база заморожена: эмбеддинги считаем один раз
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

    def load(self, user) -> bool:
        p = PROFILES_DIR / f"{user}.pt"
        if p.exists():
            self.adapter.load_state_dict(torch.load(p, map_location=self.device, weights_only=True))
            self.feat_mean = np.load(PROFILES_DIR / f"{user}_mean.npy")
            return True
        return False