import os

import cv2
import torch
import mediapipe as mp
import numpy as np
import platform
from collections import deque
from PIL import Image, ImageDraw, ImageFont
from model import SignLanguageTransformer
from constants import *
from preprocess import normalize_landmarks

STABILITY_FRAMES = 40
CONFIDENCE_THRESH = 0.70

def get_font(size=32):
    system = platform.system()
    font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
    try:
        if font_path:
            return ImageFont.truetype(font_path, size)
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        print("Шрифт не найден. Кириллица может не отображаться.")
        return ImageFont.load_default()


class SubtitleManager:
    def __init__(self):
        self.text = ""
        self.current_stable_char = None
        self.counter = 0
        self.is_committed = False

    def update(self, predicted_char, confidence):
        if confidence < CONFIDENCE_THRESH:
            self.counter = 0
            self.is_committed = False
            self.current_stable_char = None
            return

        if predicted_char == self.current_stable_char:
            self.counter += 1
        else:
            self.counter = 0
            self.current_stable_char = predicted_char
            self.is_committed = False

        if self.counter > STABILITY_FRAMES and not self.is_committed:
            self.text += predicted_char
            self.is_committed = True

    def add_space(self):
        if self.text and self.text[-1] != " ":
            self.text += " "

    def backspace(self):
        self.text = self.text[:-1]

    def clear(self):
        self.text = ""


def draw_interface_pil(img_cv2, text_pred, conf, subtitles, font_ui, font_sub):

    img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    W, H = img_pil.size
    draw.rectangle([(0, 0), (350, 60)], fill=(0, 0, 0, 200))

    color_ui = (0, 255, 0) if conf > CONFIDENCE_THRESH else (255, 255, 0)
    ui_text = f"Жест: {text_pred} ({conf:.2f})"
    draw.text((10, 15), ui_text, font=font_ui, fill=color_ui)

    sub_height = 80
    draw.rectangle([(0, H - sub_height), (W, H)], fill=(0, 0, 0, 180))

    if subtitles:
        bbox = draw.textbbox((0, 0), subtitles, font=font_sub)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        display_sub = subtitles
        while text_w > W - 40:
            display_sub = display_sub[1:]  # Откусываем начало
            bbox = draw.textbbox((0, 0), display_sub, font=font_sub)
            text_w = bbox[2] - bbox[0]

        x_pos = (W - text_w) // 2
        y_pos = H - (sub_height + text_h) // 2 - 5
        draw.text((x_pos, y_pos), display_sub, font=font_sub, fill=(255, 255, 255))
    else:
        draw.text((20, H - 55), "Начните показывать жесты...", font=font_sub, fill=(150, 150, 150))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def run_demo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Загрузка модели на {device}...")
    model = SignLanguageTransformer().to(device)
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Ошибка: Файл {CHECKPOINT_PATH} не найден.")
        return
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Ошибка весов: {e}")
        return
    model.eval()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=NUM_HANDS, min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    font_ui = get_font(32)
    font_sub = get_font(40)

    sub_manager = SubtitleManager()

    cap = cv2.VideoCapture(0)
    buffer = deque(maxlen=SEQUENCE_LENGTH)

    current_pred_text = "..."
    current_conf = 0.0

    print("УПРАВЛЕНИЕ:")
    print("Q - Выход")
    print("SPACE - Добавить пробел")
    print("BACKSPACE - Удалить последний символ")
    print("C - Очистить все субтитры")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        features_frame = np.zeros((NUM_HANDS, NUM_LANDMARKS, 3))
        has_hands = False

        if results.multi_hand_landmarks:
            has_hands = True
            for idx, lms in enumerate(results.multi_hand_landmarks):
                if idx >= NUM_HANDS: break
                mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)
                norm = normalize_landmarks(lms.landmark)
                features_frame[idx] = norm.reshape(NUM_LANDMARKS, 3)

        if has_hands:
            buffer.append(features_frame.flatten())
        else:
            if len(buffer) > 0:
                buffer.append(np.zeros(INPUT_SIZE))
                sub_manager.current_stable_char = None
                sub_manager.is_committed = False

        if len(buffer) >= SEQUENCE_LENGTH // 2:
            data = np.array(buffer)
            curr_len = len(data)
            if curr_len < SEQUENCE_LENGTH:
                pad = SEQUENCE_LENGTH - curr_len
                data = np.pad(data, ((0, pad), (0, 0)), mode='constant')
            if len(data) > SEQUENCE_LENGTH:
                data = data[-SEQUENCE_LENGTH:]

            tensor = torch.FloatTensor(data).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(tensor)
                probs = torch.softmax(out, dim=1)
                max_prob, idx = torch.max(probs, 1)

                current_conf = max_prob.item()
                current_pred_text = IDX_TO_CLASS[idx.item()]

                sub_manager.update(current_pred_text, current_conf)

        frame = draw_interface_pil(
            frame,
            current_pred_text,
            current_conf,
            sub_manager.text,
            font_ui,
            font_sub
        )

        cv2.imshow('RSL Subtitles Demo', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:
            sub_manager.add_space()
        elif key == 8:
            sub_manager.backspace()
        elif key == ord('c') or key == ord('с'):
            sub_manager.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_demo()