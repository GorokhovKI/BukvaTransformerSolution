import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

from constants import (
    ANNOTATIONS_PATH,
    CLASS_TO_IDX,
    FEATURES_DIR,
    MAX_TRACKED_HANDS,
    NUM_LANDMARKS,
    TRIMMED_VIDEOS_DIR,
)


def normalize_landmarks(landmarks, mirror_x: bool = False) -> np.ndarray:
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    wrist = coords[0]
    centered = coords - wrist
    scale = np.linalg.norm(centered[9])
    if scale < 1e-6:
        scale = 1.0
    normalized = centered / scale
    if mirror_x:
        # каноническая ПРАВАЯ рука: левую зеркалим по x
        normalized[:, 0] = -normalized[:, 0]
    return np.clip(normalized, -5.0, 5.0).flatten()


def _motion(entries: list[tuple[int, np.ndarray]]) -> float:
    """Мера активности руки — суммарное изменение между последовательными
    обнаруженными кадрами. Жестикулирующая рука меняет конфигурацию пальцев,
    покоящаяся — нет. <2 кадров → -1 (нет данных)."""
    if len(entries) < 2:
        return -1.0
    return sum(float(np.abs(f1 - f0).sum()) for (_, f0), (_, f1) in zip(entries, entries[1:]))


def process_video(video_path: Path) -> np.ndarray | None:
    """Обрабатывает одно видео в одну каноническую (правую) руку.

    Дактиль одноручный: отслеживаем до MAX_TRACKED_HANDS рук, выбираем активную
    (с максимальным межкадровым движением), левую руку зеркалим к правой.
    Кадры без выбранной руки заполняются нулями (как в demo) — временная
    структура сохраняется, что важно для динамических букв (Ё, Й).
    """
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_TRACKED_HANDS,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        # ключ = label руки, значение = [(номер_кадра, канонические_признаки), ...]
        detected: dict[str, list[tuple[int, np.ndarray]]] = {}
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # selfie-ориентация как в demo.py — иначе MediaPipe путает Left/Right
            frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                seen: set[str] = set()
                handedness = results.multi_handedness or []
                for i, lms in enumerate(results.multi_hand_landmarks):
                    label = "Right"
                    if i < len(handedness) and handedness[i].classification:
                        label = handedness[i].classification[0].label
                    if label in seen:
                        continue
                    seen.add(label)
                    feat = normalize_landmarks(lms.landmark, mirror_x=(label == "Left"))
                    detected.setdefault(label, []).append((frame_idx, feat))
            frame_idx += 1
        cap.release()

    if not detected or frame_idx == 0:
        return None

    # Активная рука — максимум движения; при равенстве — правая, затем больше кадров.
    def key(label: str) -> tuple[float, int, int]:
        entries = detected[label]
        return (_motion(entries), 1 if label == "Right" else 0, len(entries))
    best_label = max(detected, key=key)

    # Восстанавливаем временную структуру: нули там, где выбранной руки нет.
    by_frame = {i: f for i, f in detected[best_label]}
    features = np.array(
        [by_frame.get(i, np.zeros(NUM_LANDMARKS * 3, dtype=np.float32)) for i in range(frame_idx)],
        dtype=np.float32,
    )

    # Фильтрация видео с >50% нулевых кадров (рука не обнаружена)
    if (features.sum(axis=1) == 0).mean() > 0.5:
        return None

    return features


def _worker(args: tuple) -> tuple[str, int, bool] | None:
    """Воркер для ProcessPoolExecutor: обрабатывает одно видео и сохраняет .npy."""
    video_path, save_path, label_idx, is_train = args
    features = process_video(Path(video_path))
    if features is None:
        return None
    np.save(save_path, features)
    return str(save_path), label_idx, is_train


def run_preprocessing(num_workers: int = 4) -> None:
    print(f"Читаем аннотации из {ANNOTATIONS_PATH}...")
    try:
        df = pd.read_csv(ANNOTATIONS_PATH, sep="\t")
    except Exception as e:
        print(f"Ошибка чтения TSV: {e}")
        return

    required_cols = {"attachment_id", "text", "train"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        print(f"Ошибка: в TSV не найдены колонки: {missing}")
        print("Доступные колонки:", df.columns.tolist())
        return

    save_dir = Path(FEATURES_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = Path(TRIMMED_VIDEOS_DIR)

    # Формируем список задач с сохранением официального train/test сплита
    tasks: list[tuple] = []
    skipped_label = 0
    skipped_file = 0

    for _, row in df.iterrows():
        video_id = str(row["attachment_id"])
        label_char = str(row["text"]).upper().strip()
        is_train_flag = bool(row["train"])

        if label_char not in CLASS_TO_IDX:
            skipped_label += 1
            continue

        video_path = videos_dir / f"{video_id}.mp4"
        if not video_path.exists():
            video_path = videos_dir / f"{video_id}.avi"
            if not video_path.exists():
                skipped_file += 1
                continue

        save_path = save_dir / f"{video_id}.npy"
        tasks.append((str(video_path), str(save_path), CLASS_TO_IDX[label_char], is_train_flag))

    print(f"Найдено {len(tasks)} видео (пропущено: {skipped_label} по метке, {skipped_file} по файлу)")
    print(f"Запуск обработки (workers={num_workers})...")

    dataset_paths: list[str] = []
    dataset_labels: list[int] = []
    dataset_is_train: list[bool] = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_worker, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                path, label, is_train_flag = result
                dataset_paths.append(path)
                dataset_labels.append(label)
                dataset_is_train.append(is_train_flag)

    if not dataset_paths:
        print("Предупреждение: ни одно видео не было успешно обработано.")
        return

    # Перемешиваем только train-часть, test оставляем как есть
    train_indices = [i for i, t in enumerate(dataset_is_train) if t]
    test_indices  = [i for i, t in enumerate(dataset_is_train) if not t]
    random.shuffle(train_indices)
    order = train_indices + test_indices

    dataset_paths    = [dataset_paths[i]    for i in order]
    dataset_labels   = [dataset_labels[i]   for i in order]
    dataset_is_train = [dataset_is_train[i] for i in order]

    meta_data = {
        "paths":     dataset_paths,
        "labels":    dataset_labels,
        "is_train":  dataset_is_train,
        "label_map": CLASS_TO_IDX,
    }
    np.save(save_dir / "metadata.npy", meta_data)

    n_train = sum(dataset_is_train)
    n_test  = len(dataset_is_train) - n_train
    print(
        f"Готово! Обработано: {len(dataset_paths)} видео "
        f"(train={n_train}, test={n_test}). "
        f"Сохранено в {FEATURES_DIR}"
    )


if __name__ == "__main__":
    run_preprocessing()