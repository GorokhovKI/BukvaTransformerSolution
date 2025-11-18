import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
from constants import *


def normalize_landmarks(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = coords[0]
    centered = coords - wrist
    scale = np.linalg.norm(centered[9])
    if scale < 1e-6: scale = 1.0
    return (centered / scale).flatten()


def process_video(video_path, hands_detector):
    cap = cv2.VideoCapture(str(video_path))
    frames_data = []
    if not cap.isOpened():
        return None

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(frame_rgb)
        frame_features = np.zeros((NUM_HANDS, NUM_LANDMARKS, 3))
        if results.multi_hand_landmarks:
            for idx, landmarks in enumerate(results.multi_hand_landmarks):
                if idx >= NUM_HANDS: break
                norm_lm = normalize_landmarks(landmarks.landmark)
                frame_features[idx] = norm_lm.reshape(NUM_LANDMARKS, 3)

        frames_data.append(frame_features.flatten())
    cap.release()
    return np.array(frames_data)


def run_preprocessing():
    print(f"Читаем аннотации из {ANNOTATIONS_PATH}...")
    try:
        df = pd.read_csv(ANNOTATIONS_PATH, sep='\t')
    except Exception as e:
        print(f"Ошибка чтения TSV: {e}")
        return
    if 'attachment_id' not in df.columns or 'text' not in df.columns:
        print("Ошибка: В TSV файле не найдены колонки 'attachment_id' или 'text'")
        print("Доступные колонки:", df.columns)
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=NUM_HANDS,
        min_detection_confidence=0.5
    )

    save_dir = Path(FEATURES_DIR)
    save_dir.mkdir(exist_ok=True)
    videos_dir = Path(TRIMMED_VIDEOS_DIR)

    dataset_paths = []
    dataset_labels = []

    print(f"Начинаем обработку {len(df)} записей...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_id = str(row['attachment_id'])
        label_char = str(row['text']).upper()

        if label_char not in CLASS_TO_IDX:
            continue

        video_path = videos_dir / f"{video_id}.mp4"
        if not video_path.exists():
            video_path = videos_dir / f"{video_id}.avi"
            if not video_path.exists():
                continue
        features = process_video(video_path, hands)
        if features is not None and len(features) > 0:
            save_name = f"{video_id}.npy"
            save_path = save_dir / save_name
            np.save(save_path, features)
            dataset_paths.append(str(save_path))
            dataset_labels.append(CLASS_TO_IDX[label_char])

    meta_data = {
        "paths": dataset_paths,
        "labels": dataset_labels
    }
    np.save(save_dir / "metadata.npy", meta_data)
    print(f"Готово! Обработано видео: {len(dataset_paths)}. Сохранено в {FEATURES_DIR}")


if __name__ == "__main__":
    run_preprocessing()