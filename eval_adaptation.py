"""Оценка эффекта персональной адаптации на официальном test-сплите.

Для каждого из 20 пользователей test: адаптация на первых K сэмплах (только fit адаптера,
без калибровки — она измеренно вредит, см. PLAN_ADAPTIVITY.md), оценка на остальных.
Сравнение: базовая vs адаптированная точность.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from constants import ANNOTATIONS_PATH, CHECKPOINT_PATH, FEATURES_DIR, SEQUENCE_LENGTH
from model import SignLanguageTransformer
from personalization import PersonalizedModel

K_ADAPT = 5  # сэмплов на пользователя для адаптации


def to_window(seq: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Center-crop/pad до [1, SEQUENCE_LENGTH, F] + маска (как SignDataset)."""
    seq = np.asarray(seq, np.float32)
    n = seq.shape[0]
    if n > SEQUENCE_LENGTH:
        start = (n - SEQUENCE_LENGTH) // 2
        seq = seq[start: start + SEQUENCE_LENGTH]
        mask = np.zeros(SEQUENCE_LENGTH, dtype=bool)
    else:
        seq = np.pad(seq, ((0, SEQUENCE_LENGTH - n), (0, 0)), mode="constant")
        mask = np.concatenate([np.zeros(n, dtype=bool), np.ones(SEQUENCE_LENGTH - n, dtype=bool)])
    return (
        torch.from_numpy(seq).float().unsqueeze(0),
        torch.from_numpy(mask).bool().unsqueeze(0),
    )


@torch.no_grad()
def base_predict(model: SignLanguageTransformer, seq: np.ndarray) -> int:
    x, mask = to_window(seq)
    return int(model(x, src_key_padding_mask=mask).argmax(dim=1).item())


def main() -> None:
    device = torch.device("cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    base = SignLanguageTransformer(checkpoint["model_config"]).eval()
    base.load_state_dict(checkpoint["model_state_dict"])

    meta = np.load(f"{FEATURES_DIR}/metadata.npy", allow_pickle=True).item()
    paths = np.array(meta["paths"])
    labels = np.array(meta["labels"])
    is_train = np.array(meta["is_train"], dtype=bool)

    test_paths, test_labels = paths[~is_train], labels[~is_train]

    # attachment_id -> user_id
    df = pd.read_csv(ANNOTATIONS_PATH, sep="\t")
    id_to_user = dict(zip(df["attachment_id"].astype(str), df["user_id"]))

    by_user = defaultdict(list)
    for p, y in zip(test_paths, test_labels):
        by_user[id_to_user[Path(p).stem]].append((p, int(y)))

    users = sorted(by_user)
    usable = [u for u in users if len(by_user[u]) >= K_ADAPT + 1]
    print(f"test: {len(test_paths)} сэмплов | пользователей: {len(users)} | "
          f"участвуют (>= {K_ADAPT + 1} сэмплов): {len(usable)}\n")

    total_base = total_adapt = total = 0
    print(f"{'user_id':<14} {'n':>3} {'base':>7} {'adapt':>7} {'delta':>7}")
    print("-" * 42)
    for user in usable:
        samples = by_user[user]
        adapt_set, eval_set = samples[:K_ADAPT], samples[K_ADAPT:]

        pm = PersonalizedModel(base, device)
        for p, y in adapt_set:
            pm.add_sample(np.load(p), y)
        pm.fit()

        base_correct = sum(base_predict(base, np.load(p)) == y for p, y in eval_set)
        adapt_correct = sum(int(pm.predict_logits(np.load(p)).argmax(dim=1).item()) == y
                            for p, y in eval_set)
        n = len(eval_set)
        total_base += base_correct
        total_adapt += adapt_correct
        total += n
        print(f"{user[:14]:<14} {n:>3} {base_correct / n:>7.3f} {adapt_correct / n:>7.3f} "
              f"{adapt_correct / n - base_correct / n:>+7.3f}")
    print("-" * 42)
    print(f"{'ИТОГО':<14} {total:>3} {total_base / total:>7.3f} {total_adapt / total:>7.3f} "
          f"{total_adapt / total - total_base / total:>+7.3f}")

    full_test = sum(base_predict(base, np.load(p)) == y for p, y in zip(test_paths, test_labels))
    print(f"\nБаза (все test-сэмплы): {full_test / len(test_paths):.4f}")


if __name__ == "__main__":
    main()
