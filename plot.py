import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

PALETTE = {
    "train": "#2196F3",
    "val": "#FF5722",
    "grid": "#E0E0E0",
    "text": "#212121",
    "bg": "#FAFAFA",
    "accent": "#4CAF50",
    "warn": "#FFC107",
}

plt.rcParams.update({
    "figure.facecolor": PALETTE["bg"],
    "axes.facecolor": PALETTE["bg"],
    "axes.edgecolor": PALETTE["grid"],
    "axes.labelcolor": PALETTE["text"],
    "xtick.color": PALETTE["text"],
    "ytick.color": PALETTE["text"],
    "text.color": PALETTE["text"],
    "grid.color": PALETTE["grid"],
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
})


def _savefig(fig, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"Сохранён файл: {path}")


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_loss(history: dict, save_dir: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history["train_loss"], color=PALETTE["train"], lw=2, label="Обучающая выборка")
    ax.plot(epochs, history["val_loss"], color=PALETTE["val"], lw=2, label="Валидационная выборка")
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    best_val = min(history["val_loss"])
    ax.axvline(best_epoch, color=PALETTE["accent"], ls="--", lw=1.2, label=f"Лучшая эпоха: {best_epoch}")
    ax.scatter([best_epoch], [best_val], color=PALETTE["accent"], zorder=5, s=60)
    ax.set_title("Функция потерь по эпохам")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Потери")
    ax.legend()
    ax.grid(True, alpha=0.5)
    _savefig(fig, f"{save_dir}/01_loss.png")


def plot_accuracy(history: dict, save_dir: str) -> None:
    epochs = range(1, len(history["train_acc"]) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history["train_acc"], color=PALETTE["train"], lw=2, label="Обучающая выборка")
    ax.plot(epochs, history["val_acc"], color=PALETTE["val"], lw=2, label="Валидационная выборка")
    best_epoch = int(np.argmax(history["val_acc"])) + 1
    best_val = max(history["val_acc"])
    ax.axvline(best_epoch, color=PALETTE["accent"], ls="--", lw=1.2, label=f"Лучшая эпоха: {best_epoch}")
    ax.scatter([best_epoch], [best_val], color=PALETTE["accent"], zorder=5, s=60)
    ax.set_title("Точность по эпохам")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Точность, %")
    ax.legend()
    ax.grid(True, alpha=0.5)
    _savefig(fig, f"{save_dir}/02_accuracy.png")


def plot_f1(history: dict, save_dir: str) -> None:
    epochs = range(1, len(history["val_f1_weighted"]) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history["val_f1_weighted"], color=PALETTE["accent"], lw=2)
    best_epoch = int(np.argmax(history["val_f1_weighted"])) + 1
    best_val = max(history["val_f1_weighted"])
    ax.axvline(best_epoch, color=PALETTE["warn"], ls="--", lw=1.2, label=f"Лучшая эпоха: {best_epoch}")
    ax.scatter([best_epoch], [best_val], color=PALETTE["warn"], zorder=5, s=60)
    ax.set_title("Взвешенная F1-мера на валидации")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("F1")
    ax.legend()
    ax.grid(True, alpha=0.5)
    _savefig(fig, f"{save_dir}/03_f1_weighted.png")


def plot_overfitting_gap(history: dict, save_dir: str) -> None:
    epochs = range(1, len(history["train_acc"]) + 1)
    gap = [tr - vl for tr, vl in zip(history["train_acc"], history["val_acc"])]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(epochs, gap, alpha=0.3, color=PALETTE["warn"])
    ax.plot(epochs, gap, color=PALETTE["warn"], lw=2)
    ax.axhline(0, color=PALETTE["grid"], lw=1)
    ax.set_title("Разрыв между точностью обучения и валидации")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Разница, %")
    ax.grid(True, alpha=0.5)
    _savefig(fig, f"{save_dir}/04_overfitting_gap.png")


def plot_lr_curve(lr_history: list[float], save_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lr_history, color=PALETTE["train"], lw=1.5)
    ax.set_title("Изменение скорости обучения")
    ax.set_xlabel("Шаг батча")
    ax.set_ylabel("Скорость обучения")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.5)
    _savefig(fig, f"{save_dir}/05_learning_rate.png")


def plot_grad_norm(grad_history: list[float], save_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(grad_history, color=PALETTE["val"], lw=1, alpha=0.7, label="Норма градиента")
    window = max(1, len(grad_history) // 50)
    smoothed = np.convolve(grad_history, np.ones(window) / window, mode="valid")
    ax.plot(
        range(window - 1, len(grad_history)),
        smoothed,
        color=PALETTE["train"],
        lw=2,
        label=f"Скользящее среднее ({window})",
    )
    ax.axhline(1.0, color=PALETTE["warn"], ls="--", lw=1.2, label="Порог клиппирования")
    ax.set_title("Норма градиента по шагам")
    ax.set_xlabel("Шаг батча")
    ax.set_ylabel("L2-норма")
    ax.legend()
    ax.grid(True, alpha=0.5)
    _savefig(fig, f"{save_dir}/06_grad_norm.png")


def plot_confusion_matrix_absolute(y_true, y_pred, class_names: list[str], save_dir: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.3,
        linecolor=PALETTE["grid"],
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Матрица ошибок: абсолютные значения")
    ax.set_ylabel("Истинный класс")
    ax.set_xlabel("Предсказанный класс")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    _savefig(fig, f"{save_dir}/07_confusion_matrix_absolute.png")


def plot_confusion_matrix_normalized(y_true, y_pred, class_names: list[str], save_dir: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.3,
        linecolor=PALETTE["grid"],
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Матрица ошибок: нормализованные значения")
    ax.set_ylabel("Истинный класс")
    ax.set_xlabel("Предсказанный класс")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    _savefig(fig, f"{save_dir}/08_confusion_matrix_normalized.png")


def plot_per_class_metrics(y_true, y_pred, class_names: list[str], save_dir: str) -> None:
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    x = np.arange(len(class_names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(14, len(class_names) * 0.6), 6))
    ax.bar(x - w, precision, w, label="Точность", color=PALETTE["train"], alpha=0.85)
    ax.bar(x, recall, w, label="Полнота", color=PALETTE["val"], alpha=0.85)
    ax.bar(x + w, f1, w, label="F1-мера", color=PALETTE["accent"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Значение")
    ax.set_title("Метрики по каждому классу")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.5)
    _savefig(fig, f"{save_dir}/09_per_class_metrics.png")


def plot_top_confusions(y_true, y_pred, class_names: list[str], save_dir: str, top_n: int = 15) -> None:
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)
    flat = cm.flatten()
    indices = np.argsort(flat)[::-1][:top_n]
    errors = []
    for idx in indices:
        r, c = divmod(idx, len(class_names))
        if cm[r, c] > 0:
            errors.append((f"{class_names[r]} → {class_names[c]}", cm[r, c]))

    if not errors:
        return

    labels, counts = zip(*errors)
    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.4)))
    bars = ax.barh(labels[::-1], counts[::-1], color=PALETTE["val"], alpha=0.85)
    ax.bar_label(bars, padding=3)
    ax.set_xlabel("Количество ошибок")
    ax.set_title(f"Топ-{len(errors)} самых частых ошибок классификации")
    ax.grid(True, axis="x", alpha=0.5)
    _savefig(fig, f"{save_dir}/10_top_confusions.png")


def plot_class_accuracy(y_true, y_pred, class_names: list[str], save_dir: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    per_class = cm.diagonal() / cm.sum(axis=1)
    order = np.argsort(per_class)
    sorted_names = [class_names[i] for i in order]
    sorted_acc = per_class[order]
    colors = [
        PALETTE["val"] if a < 0.7 else PALETTE["warn"] if a < 0.9 else PALETTE["accent"]
        for a in sorted_acc
    ]

    fig, ax = plt.subplots(figsize=(10, max(5, len(class_names) * 0.35)))
    bars = ax.barh(sorted_names, sorted_acc, color=colors, alpha=0.88)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Точность")
    ax.set_title("Точность по классам")
    ax.axvline(0.9, color=PALETTE["warn"], ls="--", lw=1, label="Порог 0.9")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.5)
    _savefig(fig, f"{save_dir}/11_per_class_accuracy.png")


def plot_roc_curves(y_true, y_proba, class_names: list[str], save_dir: str, max_classes: int = 10) -> None:
    n_cls = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_cls)))
    aucs = []
    for i in range(n_cls):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        aucs.append(auc(fpr, tpr))
    top_idx = np.argsort(aucs)[:max_classes]
    cmap = plt.cm.get_cmap("tab10", max_classes)

    fig, ax = plt.subplots(figsize=(9, 7))
    for k, i in enumerate(top_idx):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, lw=1.5, color=cmap(k), label=f"{class_names[i]} (AUC={aucs[i]:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Доля ложноположительных")
    ax.set_ylabel("Доля истинноположительных")
    ax.set_title(f"ROC-кривые для {max_classes} самых сложных классов")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.4)
    _savefig(fig, f"{save_dir}/12_roc_curves.png")


def plot_pr_curves(y_true, y_proba, class_names: list[str], save_dir: str, max_classes: int = 10) -> None:
    n_cls = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_cls)))
    aucs = []
    for i in range(n_cls):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        aucs.append(auc(fpr, tpr))
    top_idx = np.argsort(aucs)[:max_classes]
    cmap = plt.cm.get_cmap("tab10", max_classes)

    fig, ax = plt.subplots(figsize=(9, 7))
    for k, i in enumerate(top_idx):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, lw=1.5, color=cmap(k), label=f"{class_names[i]} (AP={pr_auc:.2f})")
    ax.set_xlabel("Полнота")
    ax.set_ylabel("Точность")
    ax.set_title(f"PR-кривые для {max_classes} самых сложных классов")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.4)
    _savefig(fig, f"{save_dir}/13_pr_curves.png")


def plot_prediction_confidence(y_true, y_proba, y_pred, save_dir: str) -> None:
    max_prob = y_proba.max(axis=1)
    correct = np.array(y_pred) == np.array(y_true)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(max_prob[correct], bins=30, alpha=0.7, color=PALETTE["accent"], label=f"Верные ({correct.sum()})", density=True)
    ax.hist(max_prob[~correct], bins=30, alpha=0.7, color=PALETTE["val"], label=f"Ошибочные ({(~correct).sum()})", density=True)
    ax.set_xlabel("Максимальная вероятность softmax")
    ax.set_ylabel("Плотность")
    ax.set_title("Распределение уверенности модели")
    ax.legend()
    ax.grid(True, alpha=0.5)
    _savefig(fig, f"{save_dir}/14_confidence_distribution.png")


def plot_final_metrics(evaluation: dict, history: dict, save_dir: str) -> None:
    metrics = {
        "Top-1 Accuracy": evaluation["top1_accuracy"],
        "F1 weighted": evaluation["f1_weighted"],
        "F1 macro": evaluation["f1_macro"],
        "Лучшая val accuracy": max(history["val_acc"]) / 100.0,
    }
    labels = list(metrics.keys())
    values = list(metrics.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=[PALETTE["train"], PALETTE["accent"], PALETTE["val"], PALETTE["warn"]], alpha=0.85)
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Значение")
    ax.set_title("Итоговые метрики модели")
    ax.grid(True, axis="y", alpha=0.5)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    _savefig(fig, f"{save_dir}/15_final_metrics.png")


def save_all_plots(history_path: str = "results/training_history.json", evaluation_path: str = "results/evaluation_history.json", save_dir: str = "results") -> None:
    Path(save_dir).mkdir(exist_ok=True)
    history = _load_json(history_path)
    evaluation = _load_json(evaluation_path)

    y_true = evaluation["y_true"]
    y_pred = evaluation["y_pred"]
    y_proba = np.asarray(evaluation["y_proba"], dtype=float)
    class_names = evaluation["class_names"]

    print("\nПостроение графиков...")
    plot_loss(history, save_dir)
    plot_accuracy(history, save_dir)
    plot_f1(history, save_dir)
    plot_overfitting_gap(history, save_dir)
    plot_lr_curve(history["lr_history"], save_dir)
    plot_grad_norm(history["grad_norm_history"], save_dir)
    plot_confusion_matrix_absolute(y_true, y_pred, class_names, save_dir)
    plot_confusion_matrix_normalized(y_true, y_pred, class_names, save_dir)
    plot_per_class_metrics(y_true, y_pred, class_names, save_dir)
    plot_top_confusions(y_true, y_pred, class_names, save_dir)
    plot_class_accuracy(y_true, y_pred, class_names, save_dir)
    plot_roc_curves(y_true, y_proba, class_names, save_dir)
    plot_pr_curves(y_true, y_proba, class_names, save_dir)
    plot_prediction_confidence(y_true, y_proba, y_pred, save_dir)
    plot_final_metrics(evaluation, history, save_dir)
    print(f"Все графики сохранены в папке '{save_dir}/'.")


if __name__ == "__main__":
    save_all_plots()