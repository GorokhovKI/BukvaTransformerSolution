"""Оценка user-условной модели (P3 плана PLAN_USER_CONDITIONED.md).

Работает с обоими типами чекпойнтов:
- user_dim=0 (базовая модель): печатает метрики и подсказку.
- user_dim>0:
  1. Официальный test с <unk> — несмещённая проверка, что user-cond не сломал базу.
  2. Known users: train-сэмплы, cond vs <unk> (ЗАВЫШЕНО: сэмплы были в обучении —
     для несмещённой оценки нужен per-user holdout при обучении).
  3. Таблица по пользователям: acc(<unk>), acc(cond), Δ, n.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from constants import ANNOTATIONS_PATH, CHECKPOINT_PATH, FEATURES_DIR, SEQUENCE_LENGTH
from model import SignLanguageTransformer
from train import SignDataset, load_user_maps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SignLanguageTransformer(ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("user_vocab") or {}


@torch.no_grad()
def eval_batch(model, x, mask, uidx, batch=64):
    logits = []
    for i in range(0, len(x), batch):
        logits.append(
            model(
                x[i:i + batch].to(device),
                src_key_padding_mask=mask[i:i + batch].to(device),
                user_idx=uidx[i:i + batch].to(device),
            ).cpu()
        )
    return torch.cat(logits)


def build_tensors(ds):
    xs, ms, us, ys = [], [], [], []
    for i in range(len(ds)):
        x, y, m, u = ds[i]
        xs.append(x); ms.append(m); us.append(u); ys.append(y)
    return torch.stack(xs), torch.stack(ms), torch.stack(us), np.array(ys)


def report(name, y_true, logits):
    preds = logits.argmax(1).numpy()
    acc = accuracy_score(y_true, preds)
    f1w = f1_score(y_true, preds, average="weighted")
    f1m = f1_score(y_true, preds, average="macro")
    print(f"{name:<40} acc={acc:.4f}  f1w={f1w:.4f}  f1m={f1m:.4f}  n={len(y_true)}")
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    args = ap.parse_args()

    meta = np.load(Path(FEATURES_DIR) / "metadata.npy", allow_pickle=True).item()
    paths_all = np.array(meta["paths"])
    labels_all = np.array(meta["labels"])
    is_train = np.array(meta["is_train"], dtype=bool)

    model, vocab = load_model(args.checkpoint)
    print(f"Чекпойнт: {args.checkpoint}")
    print(f"user_dim={model.config.user_dim} | пользователей в словаре: {len(vocab)}")

    test_ds = SignDataset(paths_all[~is_train], labels_all[~is_train], SEQUENCE_LENGTH)
    x_t, m_t, u_t, y_t = build_tensors(test_ds)
    report("TEST (офиц. 20 неизвестных, <unk>)", y_t, eval_batch(model, x_t, m_t, u_t))

    if model.config.user_dim == 0:
        print(
            "\nМодель без user-cond (user_dim=0) — оценивать known users нечего.\n"
            "Обучение user-условной модели: python train.py (словарь создаётся из annotations.tsv)."
        )
        return

    _, att_to_user = load_user_maps(ANNOTATIONS_PATH)
    train_ds = SignDataset(paths_all[is_train], labels_all[is_train], SEQUENCE_LENGTH)
    x, m, _, y = build_tensors(train_ds)

    u_unk = torch.zeros(len(train_ds), dtype=torch.long)
    u_cond = torch.zeros(len(train_ds), dtype=torch.long)
    users = {u: [] for u in vocab}
    for i, p in enumerate(paths_all[is_train]):
        u = att_to_user.get(Path(p).stem)
        if u in vocab:
            u_cond[i] = vocab[u]
            users[u].append(i)
    mapped = (u_cond > 0).sum().item()
    print(f"\nTrain-сэмплов с известным пользователем: {mapped}/{len(train_ds)}")

    logits_unk = eval_batch(model, x, m, u_unk)
    logits_cond = eval_batch(model, x, m, u_cond)

    print("\n--- Known users (train-сэмплы; ЗАВЫШЕНО: сэмплы были в обучении) ---")
    acc_unk = report("all train <unk>", y, logits_unk)
    acc_cond = report("all train user-cond", y, logits_cond)
    print(f"Δ (cond − unk) = {acc_cond - acc_unk:+.4f}")

    print("\n--- По пользователям (train, n>=5) ---")
    rows = []
    for u, idxs in users.items():
        if len(idxs) < 5:
            continue
        idxs = np.array(idxs)
        preds_unk = logits_unk[idxs].argmax(1).numpy()
        preds_cond = logits_cond[idxs].argmax(1).numpy()
        y_u = y[idxs]
        rows.append((
            accuracy_score(y_u, preds_cond) - accuracy_score(y_u, preds_unk),
            u, accuracy_score(y_u, preds_unk), accuracy_score(y_u, preds_cond), len(idxs),
        ))
    rows.sort(reverse=True)
    print(f"{'user':<34}{'n':>4}{'acc(<unk>)':>12}{'acc(cond)':>12}{'Δ':>8}")
    for d, u, a_unk, a_cond, n in rows:
        print(f"{u:<34}{n:>4}{a_unk:>12.4f}{a_cond:>12.4f}{d:>+8.4f}")
    n_pos = sum(1 for d, *_ in rows if d > 0)
    print(f"\nПользователей с Δ>0: {n_pos}/{len(rows)}")


if __name__ == "__main__":
    main()
