import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset

from constants import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    EPOCHS,
    FEATURES_DIR,
    LEARNING_RATE,
    NUM_CLASSES,
    SEQUENCE_LENGTH,
)
from model import SignLanguageTransformer
from plot import save_all_plots

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SignDataset(Dataset):
    def __init__(self, paths, labels, seq_len: int, augment: bool = False) -> None:
        self.paths = paths
        self.labels = labels
        self.seq_len = seq_len
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        features = np.load(self.paths[idx]).astype(np.float32)
        label = self.labels[idx]

        if self.augment:
            features += np.random.normal(0, 0.005, features.shape).astype(np.float32)
            features *= np.random.uniform(0.9, 1.1)

        curr_len = features.shape[0]
        if curr_len > self.seq_len:
            start = (curr_len - self.seq_len) // 2
            features = features[start: start + self.seq_len]
            mask = np.zeros(self.seq_len, dtype=bool)
        else:
            pad_len = self.seq_len - curr_len
            features = np.pad(features, ((0, pad_len), (0, 0)), mode="constant")
            mask = np.concatenate([
                np.zeros(curr_len, dtype=bool),
                np.ones(pad_len, dtype=bool),
            ])

        return (
            torch.from_numpy(features),
            torch.tensor(label, dtype=torch.long),
            torch.from_numpy(mask),
        )


def get_model_info(model: nn.Module) -> tuple[float, int]:
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 ** 2, sum(p.numel() for p in model.parameters())


def build_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    return torch.tensor(counts.sum() / (num_classes * counts), dtype=torch.float32, device=device)


def build_optimizer(model: nn.Module, lr: float) -> optim.AdamW:
    no_decay = {"norm", "bias", "cls_token"}
    decay_p = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not any(nd in n for nd in no_decay)
    ]
    no_decay_p = [
        p for n, p in model.named_parameters()
        if p.requires_grad and any(nd in n for nd in no_decay)
    ]
    return optim.AdamW(
        [
            {"params": decay_p, "weight_decay": 1e-2},
            {"params": no_decay_p, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def train_model() -> None:
    meta_path = Path(FEATURES_DIR) / "metadata.npy"
    if not meta_path.exists():
        print(f"Метаданные не найдены по пути {meta_path}. Сначала запустите preprocess.py")
        return

    meta = np.load(meta_path, allow_pickle=True).item()
    x_paths = np.array(meta["paths"])
    y_labels = np.array(meta["labels"])

    if "is_train" in meta:
        is_train = np.array(meta["is_train"], dtype=bool)
        x_train, y_train = x_paths[is_train], y_labels[is_train]
        x_val, y_val = x_paths[~is_train], y_labels[~is_train]
        print(f"Официальное разбиение: train={len(x_train)}, test={len(x_val)}")
    else:
        from sklearn.model_selection import train_test_split
        x_train, x_val, y_train, y_val = train_test_split(
            x_paths, y_labels, test_size=0.2, random_state=42, stratify=y_labels
        )
        print(f"Резервное разбиение 80/20: train={len(x_train)}, val={len(x_val)}")

    if "label_map" in meta:
        label_map = meta["label_map"]
        idx_to_class = {v: k for k, v in label_map.items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    else:
        class_names = [str(i) for i in range(NUM_CLASSES)]

    train_ds = SignDataset(x_train, y_train, SEQUENCE_LENGTH, augment=True)
    val_ds = SignDataset(x_val, y_val, SEQUENCE_LENGTH, augment=False)

    num_workers = 0
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = SignLanguageTransformer().to(device)
    nn.init.trunc_normal_(model.cls_token, std=0.02)
    size_mb, n_params = get_model_info(model)
    print(f"Модель: {n_params/1e6:.2f} млн параметров | {size_mb:.2f} МБ | устройство: {device}")

    class_weights = build_class_weights(y_train, NUM_CLASSES)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = build_optimizer(model, LEARNING_RATE)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_f1_weighted": [],
        "lr_history": [],
        "grad_norm_history": [],
        "class_names": class_names,
    }
    best_f1 = 0.0
    patience = 20
    no_improve = 0

    print(f"\nСтарт обучения: {EPOCHS} эпох | batch={BATCH_SIZE} | lr={LEARNING_RATE}")
    print("=" * 75)

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(x, src_key_padding_mask=mask)
            loss = criterion(output, y)
            loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            history["grad_norm_history"].append(float(grad_norm))
            history["lr_history"].append(float(optimizer.param_groups[0]["lr"]))

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_correct += (output.argmax(dim=1) == y).sum().item()
            train_total += y.size(0)

        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100.0 * train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                output = model(x, src_key_padding_mask=mask)
                val_loss += criterion(output, y).item()
                preds = output.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100.0 * val_correct / val_total
        epoch_val_f1 = f1_score(val_targets, val_preds, average="weighted")

        history["train_loss"].append(float(epoch_train_loss))
        history["val_loss"].append(float(epoch_val_loss))
        history["train_acc"].append(float(epoch_train_acc))
        history["val_acc"].append(float(epoch_val_acc))
        history["val_f1_weighted"].append(float(epoch_val_f1))

        print(
            f"Эпоха {epoch + 1:>3}/{EPOCHS} | "
            f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}% F1: {epoch_val_f1:.4f}"
        )

        if epoch_val_f1 > best_f1:
            best_f1 = epoch_val_f1
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model.config,
                    "epoch": epoch + 1,
                    "best_f1": best_f1,
                },
                CHECKPOINT_PATH,
            )
            print(f"  -> Лучшая модель сохранена (F1={best_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nРанняя остановка после {patience} эпох без улучшения F1.")
                break

    Path("results").mkdir(exist_ok=True)
    with open("results/training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("\nФинальная оценка на валидационной выборке...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_labels, all_proba = [], [], []
    with torch.no_grad():
        for x, y, mask in val_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            logits = model(x, src_key_padding_mask=mask)
            proba = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_proba.extend(proba.cpu().numpy())

    evaluation = {
        "y_true": list(map(int, all_labels)),
        "y_pred": list(map(int, all_preds)),
        "y_proba": np.asarray(all_proba, dtype=float).tolist(),
        "class_names": class_names,
        "model_size_mb": float(size_mb),
        "model_params": int(n_params),
        "top1_accuracy": float(accuracy_score(all_labels, all_preds)),
        "f1_weighted": float(f1_score(all_labels, all_preds, average="weighted")),
        "f1_macro": float(f1_score(all_labels, all_preds, average="macro")),
    }

    with open("results/evaluation_history.json", "w", encoding="utf-8") as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)

    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    with open("results/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    save_all_plots(
        history_path="results/training_history.json",
        evaluation_path="results/evaluation_history.json",
        save_dir="results",
    )

    print("\n" + "=" * 75)
    print(f"{'Размер (МБ)':<14} | {'Параметры (млн)':<18} | {'Top-1 Acc':<10} | {'F1 weighted'}")
    print(f"{size_mb:<14.2f} | {n_params/1e6:<18.2f} | {evaluation['top1_accuracy']:<10.4f} | {evaluation['f1_weighted']:.4f}")
    print("=" * 75)
    print("Готово! История и графики сохранены в папке 'results/'.")


if __name__ == "__main__":
    train_model()