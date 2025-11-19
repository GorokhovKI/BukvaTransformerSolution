import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import json

from model import SignLanguageTransformer
from constants import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SignDataset(Dataset):
    def __init__(self, paths, labels, seq_len):
        self.paths = paths
        self.labels = labels
        self.seq_len = seq_len

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        features = np.load(self.paths[idx])
        label = self.labels[idx]

        curr_len = features.shape[0]
        if curr_len > self.seq_len:
            start = (curr_len - self.seq_len) // 2
            features = features[start: start + self.seq_len]
            mask = np.zeros(self.seq_len, dtype=bool)
        else:
            pad_len = self.seq_len - curr_len
            features = np.pad(features, ((0, pad_len), (0, 0)), mode='constant')
            mask = np.concatenate([np.zeros(curr_len, dtype=bool), np.ones(pad_len, dtype=bool)])

        return torch.FloatTensor(features), torch.LongTensor([label]), torch.BoolTensor(mask)


def get_model_info(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    total_params = sum(p.numel() for p in model.parameters())

    return size_all_mb, total_params

def save_plots(history, save_dir='results'):
    Path(save_dir).mkdir(exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label='Потери на обучающей выборке')
    plt.plot(epochs, history['val_loss'], label='Потери на тестовой выборке')
    plt.title('Потери на обучающей и тестовой выборке')
    plt.xlabel('Эпохи')
    plt.ylabel('Потери')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/loss_plot.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_acc'], label='Точность на обучающей выборке')
    plt.plot(epochs, history['val_acc'], label='Точность на тестовой выборке')
    plt.title('Точность на обучающей и тестовой выборке')
    plt.xlabel('Эпохи')
    plt.ylabel('Точность (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/accuracy_plot.png')
    plt.close()


def save_confusion_matrix(y_true, y_pred, class_names, save_dir='results'):
    Path(save_dir).mkdir(exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Матрица ошибок')
    plt.ylabel('Правильные классы')
    plt.xlabel('Предсказанные классы')
    plt.savefig(f'{save_dir}/confusion_matrix.png')
    plt.close()

    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(f'{save_dir}/classification_report.txt', 'w') as f:
        f.write(report)
    print("\nClassification Report saved.")


def train_model():
    meta_path = Path(FEATURES_DIR) / "metadata.npy"
    if not meta_path.exists():
        print(f"❌ Metadata не найдена по пути {meta_path}. Запустите preprocess.py")
        return

    meta = np.load(meta_path, allow_pickle=True).item()
    X_paths = meta['paths']
    y_labels = meta['labels']

    if 'label_map' in meta:
        label_map = meta['label_map']
        idx_to_class = {v: k for k, v in label_map.items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    else:
        unique_labels = np.unique(y_labels)
        class_names = [str(l) for l in unique_labels]

    X_train, X_val, y_train, y_val = train_test_split(
        X_paths, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )

    train_ds = SignDataset(X_train, y_train, SEQUENCE_LENGTH)
    val_ds = SignDataset(X_val, y_val, SEQUENCE_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SignLanguageTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_acc = 0.0
    print(f"Старт обучения. Train size: {len(X_train)}, Val size: {len(X_val)}")
    print(f"Device: {device}")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.squeeze().to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(x, src_key_padding_mask=mask)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(output, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()

        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100 * train_correct / train_total

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(device), y.squeeze().to(device), mask.to(device)
                output = model(x, src_key_padding_mask=mask)
                loss = criterion(output, y)

                val_loss += loss.item()
                _, predicted = torch.max(output, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * val_correct / val_total

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f"Epoch {epoch + 1}/{EPOCHS} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save({'model_state_dict': model.state_dict()}, CHECKPOINT_PATH)
            print("Model Saved (New Best)")

    print("\nГенерация графиков обучения...")
    save_plots(history)

    with open('training_history.json', 'w') as f:
        json.dump(history, f)

    print("\nСбор данных для матрицы ошибок...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH)['model_state_dict'])
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y, mask in val_loader:
            x, y, mask = x.to(device), y.squeeze().to(device), mask.to(device)
            output = model(x, src_key_padding_mask=mask)
            _, predicted = torch.max(output, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    save_confusion_matrix(all_labels, all_preds, class_names)

    top1_acc = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    model_size_mb, num_params = get_model_info(model)
    num_params_m = num_params / 1_000_000

    print(f"{'Model Size (MB)':<18} | {'Parameters (M)':<15} | {'Top-1 Acc':<10} | {'F1 Score':<10}")
    print(f"{model_size_mb:<18.2f} | {num_params_m:<15.2f} | {top1_acc:<10.2%} | {f1_weighted:<10.2f}")

    print("Все готово! Результаты сохранены в папку 'results/'.")


if __name__ == "__main__":
    train_model()