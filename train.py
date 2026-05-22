import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Цветовая палитра (единая для всех графиков) ──────────────────────────────
PALETTE = {
    "train":   "#2196F3",
    "val":     "#FF5722",
    "grid":    "#E0E0E0",
    "text":    "#212121",
    "bg":      "#FAFAFA",
    "accent":  "#4CAF50",
    "warn":    "#FFC107",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["bg"],
    "axes.edgecolor":    PALETTE["grid"],
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["text"],
    "ytick.color":       PALETTE["text"],
    "text.color":        PALETTE["text"],
    "grid.color":        PALETTE["grid"],
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "legend.fontsize":   10,
})


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SignDataset(Dataset):
    def __init__(self, paths, labels, seq_len: int, augment: bool = False) -> None:
        self.paths   = paths
        self.labels  = labels
        self.seq_len = seq_len
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        features = np.load(self.paths[idx]).astype(np.float32)
        label    = self.labels[idx]

        if self.augment:
            features += np.random.normal(0, 0.005, features.shape).astype(np.float32)
            features *= np.random.uniform(0.9, 1.1)

        curr_len = features.shape[0]
        if curr_len > self.seq_len:
            start    = (curr_len - self.seq_len) // 2
            features = features[start: start + self.seq_len]
            mask     = np.zeros(self.seq_len, dtype=bool)
        else:
            pad_len  = self.seq_len - curr_len
            features = np.pad(features, ((0, pad_len), (0, 0)), mode="constant")
            mask     = np.concatenate([
                np.zeros(curr_len, dtype=bool),
                np.ones(pad_len,   dtype=bool),
            ])

        return (
            torch.from_numpy(features),
            torch.tensor(label, dtype=torch.long),
            torch.from_numpy(mask),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_model_info(model: nn.Module) -> tuple[float, int]:
    param_size  = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 ** 2, sum(p.numel() for p in model.parameters())


def build_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    return torch.tensor(counts.sum() / (num_classes * counts), dtype=torch.float32, device=device)


def build_optimizer(model: nn.Module, lr: float) -> optim.AdamW:
    no_decay     = {"norm", "bias", "cls_token"}
    decay_p      = [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)]
    no_decay_p   = [p for n, p in model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)]
    return optim.AdamW(
        [{"params": decay_p,    "weight_decay": 1e-2},
         {"params": no_decay_p, "weight_decay": 0.0}],
        lr=lr,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Визуализации
# ─────────────────────────────────────────────────────────────────────────────

def _savefig(fig, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_loss_accuracy(history: dict, save_dir: str) -> None:
    """1 & 2: кривые loss и accuracy на одном рисунке."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training dynamics", fontsize=15, fontweight="bold")

    for ax, train_key, val_key, ylabel, title in [
        (ax1, "train_loss", "val_loss",  "Loss",         "Loss"),
        (ax2, "train_acc",  "val_acc",   "Accuracy (%)", "Accuracy"),
    ]:
        ax.plot(epochs, history[train_key], color=PALETTE["train"], lw=2, label="Train")
        ax.plot(epochs, history[val_key],   color=PALETTE["val"],   lw=2, label="Validation")
        best_epoch = int(np.argmin(history[val_key]) if "loss" in val_key
                         else np.argmax(history[val_key])) + 1
        best_val   = (min if "loss" in val_key else max)(history[val_key])
        ax.axvline(best_epoch, color=PALETTE["accent"], ls="--", lw=1.2, label=f"Best epoch {best_epoch}")
        ax.scatter([best_epoch], [best_val], color=PALETTE["accent"], zorder=5, s=60)
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True, alpha=0.5)

    _savefig(fig, f"{save_dir}/01_loss_accuracy.png")


def plot_overfitting_gap(history: dict, save_dir: str) -> None:
    """3: разрыв между train и val accuracy — индикатор переобучения."""
    epochs = range(1, len(history["train_acc"]) + 1)
    gap    = [tr - vl for tr, vl in zip(history["train_acc"], history["val_acc"])]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(epochs, gap, alpha=0.3, color=PALETTE["warn"])
    ax.plot(epochs, gap, color=PALETTE["warn"], lw=2)
    ax.axhline(0, color=PALETTE["grid"], lw=1)
    ax.set_title("Overfitting gap  (Train Acc − Val Acc)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Gap (%)")
    ax.grid(True, alpha=0.5)
    _savefig(fig, f"{save_dir}/02_overfitting_gap.png")


def plot_lr_curve(lr_history: list[float], save_dir: str) -> None:
    """4: график изменения learning rate по шагам."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lr_history, color=PALETTE["train"], lw=1.5)
    ax.set_title("Learning rate schedule (OneCycleLR)")
    ax.set_xlabel("Batch step"); ax.set_ylabel("Learning rate")
    ax.set_yscale("log"); ax.grid(True, alpha=0.5)
    _savefig(fig, f"{save_dir}/03_lr_schedule.png")


def plot_grad_norm(grad_history: list[float], save_dir: str) -> None:
    """5: норма градиентов по шагам — диагностика стабильности обучения."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(grad_history, color=PALETTE["val"], lw=1, alpha=0.7, label="Grad norm")
    # Скользящее среднее
    window = max(1, len(grad_history) // 50)
    smoothed = np.convolve(grad_history, np.ones(window) / window, mode="valid")
    ax.plot(range(window - 1, len(grad_history)), smoothed,
            color=PALETTE["train"], lw=2, label=f"Moving avg ({window})")
    ax.axhline(1.0, color=PALETTE["warn"], ls="--", lw=1.2, label="Clip threshold")
    ax.set_title("Gradient L2 norm per batch")
    ax.set_xlabel("Batch step"); ax.set_ylabel("L2 norm")
    ax.legend(); ax.grid(True, alpha=0.5)
    _savefig(fig, f"{save_dir}/04_grad_norm.png")


def plot_confusion_matrix(y_true, y_pred, class_names: list[str], save_dir: str) -> None:
    """6 & 7: матрица ошибок — абсолютная и нормализованная."""
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(28, 12))
    fig.suptitle("Confusion matrices", fontsize=16, fontweight="bold")

    for ax, data, fmt, title in [
        (axes[0], cm,      "d",    "Absolute counts"),
        (axes[1], cm_norm, ".2f",  "Normalized (recall per class)"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, linewidths=0.3, linecolor=PALETTE["grid"],
                    cbar_kws={"shrink": 0.8})
        ax.set_title(title, fontsize=13)
        ax.set_ylabel("True class"); ax.set_xlabel("Predicted class")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    _savefig(fig, f"{save_dir}/05_confusion_matrices.png")


def plot_per_class_metrics(y_true, y_pred, class_names: list[str], save_dir: str) -> None:
    """8: precision / recall / F1 по каждому классу — горизонтальный bar chart."""
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall    = recall_score(   y_true, y_pred, average=None, zero_division=0)
    f1        = f1_score(       y_true, y_pred, average=None, zero_division=0)

    x   = np.arange(len(class_names))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(max(14, len(class_names) * 0.6), 6))
    ax.bar(x - w,   precision, w, label="Precision", color=PALETTE["train"],  alpha=0.85)
    ax.bar(x,       recall,    w, label="Recall",    color=PALETTE["val"],    alpha=0.85)
    ax.bar(x + w,   f1,        w, label="F1",        color=PALETTE["accent"], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.set_title("Per-class Precision / Recall / F1")
    ax.legend(); ax.grid(True, axis="y", alpha=0.5)
    _savefig(fig, f"{save_dir}/06_per_class_metrics.png")


def plot_top_confusions(y_true, y_pred, class_names: list[str],
                        save_dir: str, top_n: int = 15) -> None:
    """9: топ-N самых частых ошибок классификатора."""
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)            # убираем правильные предсказания
    flat    = cm.flatten()
    indices = np.argsort(flat)[::-1][:top_n]
    errors  = []
    for idx in indices:
        r, c = divmod(idx, len(class_names))
        if cm[r, c] > 0:
            errors.append((f"{class_names[r]}→{class_names[c]}", cm[r, c]))

    if not errors:
        return

    labels, counts = zip(*errors)
    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.4)))
    bars = ax.barh(labels[::-1], counts[::-1], color=PALETTE["val"], alpha=0.85)
    ax.bar_label(bars, padding=3)
    ax.set_xlabel("Number of errors")
    ax.set_title(f"Top-{len(errors)} most common misclassifications")
    ax.grid(True, axis="x", alpha=0.5)
    _savefig(fig, f"{save_dir}/07_top_confusions.png")


def plot_class_accuracy(y_true, y_pred, class_names: list[str], save_dir: str) -> None:
    """10: точность по каждому классу, отсортированная по возрастанию."""
    cm        = confusion_matrix(y_true, y_pred)
    per_class = cm.diagonal() / cm.sum(axis=1)
    order     = np.argsort(per_class)
    sorted_names = [class_names[i] for i in order]
    sorted_acc   = per_class[order]

    colors = [PALETTE["val"] if a < 0.7 else
              PALETTE["warn"] if a < 0.9 else
              PALETTE["accent"] for a in sorted_acc]

    fig, ax = plt.subplots(figsize=(10, max(5, len(class_names) * 0.35)))
    bars = ax.barh(sorted_names, sorted_acc, color=colors, alpha=0.88)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_xlim(0, 1.12); ax.set_xlabel("Accuracy")
    ax.set_title("Per-class accuracy (sorted)")
    ax.axvline(0.9, color=PALETTE["warn"], ls="--", lw=1, label="0.9 threshold")
    ax.legend(); ax.grid(True, axis="x", alpha=0.5)
    _savefig(fig, f"{save_dir}/08_per_class_accuracy.png")


def plot_roc_pr_curves(y_true, y_proba, class_names: list[str],
                       save_dir: str, max_classes: int = 10) -> None:
    """11 & 12: ROC-кривые и Precision-Recall кривые (one-vs-rest, топ-N классов по AUC)."""
    n_cls     = len(class_names)
    y_bin     = label_binarize(y_true, classes=list(range(n_cls)))
    # proba shape: (N, n_cls)

    # Считаем AUC для каждого класса и берём топ max_classes
    aucs = []
    for i in range(n_cls):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        aucs.append(auc(fpr, tpr))
    top_idx = np.argsort(aucs)[:max_classes]   # классы с наименьшим AUC (труднее всего)

    cmap = plt.cm.get_cmap("tab10", max_classes)

    # ROC
    fig, ax = plt.subplots(figsize=(9, 7))
    for k, i in enumerate(top_idx):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, lw=1.5, color=cmap(k), label=f"{class_names[i]} (AUC={aucs[i]:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC curves — {max_classes} lowest-AUC classes")
    ax.legend(loc="lower right", fontsize=8); ax.grid(True, alpha=0.4)
    _savefig(fig, f"{save_dir}/09_roc_curves.png")

    # Precision-Recall
    fig, ax = plt.subplots(figsize=(9, 7))
    for k, i in enumerate(top_idx):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, lw=1.5, color=cmap(k),
                label=f"{class_names[i]} (AP={pr_auc:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall curves — {max_classes} lowest-AUC classes")
    ax.legend(loc="upper right", fontsize=8); ax.grid(True, alpha=0.4)
    _savefig(fig, f"{save_dir}/10_pr_curves.png")


def plot_prediction_confidence(y_true, y_proba, y_pred, save_dir: str) -> None:
    """13: распределение уверенности модели (correct vs wrong predictions)."""
    max_prob = y_proba.max(axis=1)
    correct  = (np.array(y_pred) == np.array(y_true))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(max_prob[correct],  bins=30, alpha=0.7, color=PALETTE["accent"],
            label=f"Correct ({correct.sum()})",  density=True)
    ax.hist(max_prob[~correct], bins=30, alpha=0.7, color=PALETTE["val"],
            label=f"Wrong ({(~correct).sum()})", density=True)
    ax.set_xlabel("Max softmax probability"); ax.set_ylabel("Density")
    ax.set_title("Prediction confidence distribution")
    ax.legend(); ax.grid(True, alpha=0.5)
    _savefig(fig, f"{save_dir}/11_confidence_distribution.png")


def plot_summary_dashboard(history: dict, y_true, y_pred, class_names: list[str],
                           save_dir: str) -> None:
    """14: сводный дашборд — все ключевые метрики на одном листе."""
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall    = recall_score(   y_true, y_pred, average=None, zero_division=0)
    f1        = f1_score(       y_true, y_pred, average=None, zero_division=0)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Training summary dashboard", fontsize=16, fontweight="bold", y=1.01)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history["train_loss"], color=PALETTE["train"], lw=2, label="Train")
    ax1.plot(epochs, history["val_loss"],   color=PALETTE["val"],   lw=2, label="Val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(True, alpha=0.4)

    # Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history["train_acc"], color=PALETTE["train"], lw=2, label="Train")
    ax2.plot(epochs, history["val_acc"],   color=PALETTE["val"],   lw=2, label="Val")
    ax2.set_title("Accuracy (%)"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(True, alpha=0.4)

    # Overfitting gap
    ax3 = fig.add_subplot(gs[0, 2])
    gap = [tr - vl for tr, vl in zip(history["train_acc"], history["val_acc"])]
    ax3.fill_between(epochs, gap, alpha=0.3, color=PALETTE["warn"])
    ax3.plot(epochs, gap, color=PALETTE["warn"], lw=2)
    ax3.set_title("Overfitting gap"); ax3.set_xlabel("Epoch"); ax3.grid(True, alpha=0.4)

    # Per-class F1 barh
    ax4 = fig.add_subplot(gs[1, :2])
    order = np.argsort(f1)
    ax4.barh([class_names[i] for i in order], f1[order],
             color=[PALETTE["val"] if v < 0.7 else
                    PALETTE["warn"] if v < 0.9 else
                    PALETTE["accent"] for v in f1[order]],
             alpha=0.85)
    ax4.set_xlim(0, 1.05); ax4.set_xlabel("F1 score")
    ax4.set_title("Per-class F1 (sorted)"); ax4.grid(True, axis="x", alpha=0.4)

    # Final metrics text
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    top1   = accuracy_score(y_true, y_pred)
    f1_w   = f1_score(y_true, y_pred, average="weighted")
    f1_mac = f1_score(y_true, y_pred, average="macro")
    lines  = [
        f"Top-1 Accuracy : {top1:.4f}",
        f"F1 Weighted    : {f1_w:.4f}",
        f"F1 Macro       : {f1_mac:.4f}",
        f"",
        f"Best Val Acc   : {max(history['val_acc']):.2f}%",
        f"Best Val Loss  : {min(history['val_loss']):.4f}",
        f"Epochs trained : {len(history['train_loss'])}",
    ]
    ax5.text(0.05, 0.95, "\n".join(lines), transform=ax5.transAxes,
             fontsize=12, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor=PALETTE["bg"],
                       edgecolor=PALETTE["grid"], alpha=0.9))
    ax5.set_title("Final metrics")

    _savefig(fig, f"{save_dir}/00_summary_dashboard.png")


def save_all_plots(
    history: dict,
    lr_history: list[float],
    grad_history: list[float],
    y_true, y_pred,
    y_proba,
    class_names: list[str],
    save_dir: str = "results",
) -> None:
    Path(save_dir).mkdir(exist_ok=True)
    print("\nГенерация графиков...")
    plot_summary_dashboard(history, y_true, y_pred, class_names, save_dir)
    plot_loss_accuracy(history, save_dir)
    plot_overfitting_gap(history, save_dir)
    plot_lr_curve(lr_history, save_dir)
    plot_grad_norm(grad_history, save_dir)
    plot_confusion_matrix(y_true, y_pred, class_names, save_dir)
    plot_per_class_metrics(y_true, y_pred, class_names, save_dir)
    plot_top_confusions(y_true, y_pred, class_names, save_dir)
    plot_class_accuracy(y_true, y_pred, class_names, save_dir)
    plot_roc_pr_curves(y_true, np.array(y_proba), class_names, save_dir)
    plot_prediction_confidence(y_true, np.array(y_proba), y_pred, save_dir)
    print(f"Все графики сохранены в '{save_dir}/'")


# ─────────────────────────────────────────────────────────────────────────────
# Train loop
# ─────────────────────────────────────────────────────────────────────────────

def train_model() -> None:
    meta_path = Path(FEATURES_DIR) / "metadata.npy"
    if not meta_path.exists():
        print(f"Metadata не найдена по пути {meta_path}. Запустите preprocess.py")
        return

    meta     = np.load(meta_path, allow_pickle=True).item()
    X_paths  = np.array(meta["paths"])
    y_labels = np.array(meta["labels"])

    if "is_train" in meta:
        is_train = np.array(meta["is_train"], dtype=bool)
        X_train, y_train = X_paths[is_train],  y_labels[is_train]
        X_val,   y_val   = X_paths[~is_train], y_labels[~is_train]
        print(f"Официальный сплит: train={len(X_train)}, test={len(X_val)}")
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_paths, y_labels, test_size=0.2, random_state=42, stratify=y_labels
        )
        print(f"Fallback сплит 80/20: train={len(X_train)}, val={len(X_val)}")

    if "label_map" in meta:
        label_map    = meta["label_map"]
        idx_to_class = {v: k for k, v in label_map.items()}
        class_names  = [idx_to_class[i] for i in range(len(idx_to_class))]
    else:
        class_names = [str(i) for i in range(NUM_CLASSES)]

    train_ds = SignDataset(X_train, y_train, SEQUENCE_LENGTH, augment=True)
    val_ds   = SignDataset(X_val,   y_val,   SEQUENCE_LENGTH, augment=False)

    num_workers  = 0
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=num_workers, pin_memory=(device.type == "cuda"))

    model = SignLanguageTransformer().to(device)
    nn.init.trunc_normal_(model.cls_token, std=0.02)
    size_mb, n_params = get_model_info(model)
    print(f"Model: {n_params/1e6:.2f}M params | {size_mb:.2f} MB | device: {device}")

    class_weights = build_class_weights(y_train, NUM_CLASSES)
    criterion     = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer     = build_optimizer(model, LEARNING_RATE)
    scheduler     = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader), epochs=EPOCHS,
        pct_start=0.1, anneal_strategy="cos",
        div_factor=10.0, final_div_factor=100.0,
    )

    history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    lr_history   = []   # per-batch LR
    grad_history = []   # per-batch grad norm
    best_f1      = 0.0
    patience     = 20
    no_improve   = 0

    print(f"\nСтарт обучения: {EPOCHS} epochs | batch={BATCH_SIZE} | lr={LEARNING_RATE}")
    print("=" * 75)

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(x, src_key_padding_mask=mask)
            loss   = criterion(output, y)
            loss.backward()

            # Логируем норму ДО clip
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            grad_history.append(grad_norm)
            lr_history.append(optimizer.param_groups[0]["lr"])

            optimizer.step()
            scheduler.step()

            train_loss    += loss.item()
            train_correct += (output.argmax(dim=1) == y).sum().item()
            train_total   += y.size(0)

        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc  = 100.0 * train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                output = model(x, src_key_padding_mask=mask)
                val_loss    += criterion(output, y).item()
                preds        = output.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total   += y.size(0)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc  = 100.0 * val_correct / val_total
        epoch_val_f1   = f1_score(val_targets, val_preds, average="weighted")

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(
            f"Epoch {epoch+1:>3}/{EPOCHS} | "
            f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}% F1: {epoch_val_f1:.4f}"
        )

        if epoch_val_f1 > best_f1:
            best_f1 = epoch_val_f1; no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config":     model.config,
                "epoch":            epoch + 1,
                "best_f1":          best_f1,
            }, CHECKPOINT_PATH)
            print(f"  → Best model saved (F1={best_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping после {patience} эпох без улучшения F1.")
                break

    with open("training_history.json", "w") as f:
        json.dump(history, f)

    # ── Финальная оценка ────────────────────────────────────────────────────
    print("\nФинальная оценка на тестовой выборке...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)["model_state_dict"])
    model.eval()

    all_preds, all_labels, all_proba = [], [], []
    with torch.no_grad():
        for x, y, mask in val_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            logits = model(x, src_key_padding_mask=mask)
            proba  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_proba.extend(proba.cpu().numpy())

    # Все графики разом
    save_all_plots(
        history     = history,
        lr_history  = lr_history,
        grad_history= grad_history,
        y_true      = all_labels,
        y_pred      = all_preds,
        y_proba     = all_proba,
        class_names = class_names,
        save_dir    = "results",
    )

    # Текстовый отчёт
    report = classification_report(all_labels, all_preds,
                                   target_names=class_names, digits=4)
    Path("results").mkdir(exist_ok=True)
    with open("results/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    top1   = accuracy_score(all_labels, all_preds)
    f1_w   = f1_score(all_labels, all_preds, average="weighted")
    print("\n" + "=" * 75)
    print(f"{'Size (MB)':<12} | {'Params (M)':<12} | {'Top-1 Acc':<10} | {'F1 Weighted'}")
    print(f"{size_mb:<12.2f} | {n_params/1e6:<12.2f} | {top1:<10.4f} | {f1_w:.4f}")
    print("=" * 75)
    print("Готово! Результаты сохранены в 'results/'.")


if __name__ == "__main__":
    train_model()