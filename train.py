import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
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

        # Приводим к фиксированной длине (Padding или Truncating)
        if curr_len > self.seq_len:
            # Если длиннее, берем центр
            start = (curr_len - self.seq_len) // 2
            features = features[start: start + self.seq_len]
            mask = np.zeros(self.seq_len, dtype=bool)  # False = не паддинг
        else:
            # Если короче, добиваем нулями
            pad_len = self.seq_len - curr_len
            features = np.pad(features, ((0, pad_len), (0, 0)), mode='constant')
            # Маска: 0 для данных, 1 для паддинга
            mask = np.concatenate([np.zeros(curr_len, dtype=bool), np.ones(pad_len, dtype=bool)])

        return torch.FloatTensor(features), torch.LongTensor([label]), torch.BoolTensor(mask)


def train_model():
    meta_path = Path(FEATURES_DIR) / "metadata.npy"
    if not meta_path.exists():
        print("Metadata не найдена. Запустите preprocess.py")
        return

    meta = np.load(meta_path, allow_pickle=True).item()
    X_paths = meta['paths']
    y_labels = meta['labels']

    # Stratify гарантирует, что редкие буквы попадут и в трейн, и в тест
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

    best_acc = 0.0

    print(f"Старт обучения. Train size: {len(X_train)}, Val size: {len(X_val)}")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.squeeze().to(device), mask.to(device)

            optimizer.zero_grad()
            output = model(x, src_key_padding_mask=mask)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Валидация
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(device), y.squeeze().to(device), mask.to(device)
                output = model(x, src_key_padding_mask=mask)
                _, predicted = torch.max(output, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        acc = 100 * correct / total
        print(f"Epoch {epoch + 1}: Loss={train_loss / len(train_loader):.4f}, Val Acc={acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save({'model_state_dict': model.state_dict()}, CHECKPOINT_PATH)
            print("  --> Model Saved")


if __name__ == "__main__":
    train_model()