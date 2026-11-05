# BukvaTransformerSolution — Спецификация проекта

> Назначение документа: единый источник правды для будущей разработки. Фиксирует назначение,
> архитектуру, данные, модель, достигнутые метрики и известные риски. Обновлять при каждом
> значимом изменении кода или данных.

---

## 1. Назначение проекта

Распознавание **русской дактильной азбуки** (жестового алфавита РЖЯ) по видеопотоку.

- Вход: видеокадры (веб-камера или файл `.mp4`).
- Выход: буква русского алфавита `А`–`Я` (33 класса).
- Применение: субтитры в реальном времени для жестового общения (см. `demo.py`),
  экспорт модели в Unity (`rsl_dactyl.onnx`), персонализация под конкретного пользователя (`personalization.py`).

Ключевые артефакты:
| Файл | Роль |
|---|---|
| `best_model_bukva.pth` | Лучший чекпойнт (эпоха 93, F1≈0.9126) |
| `rsl_dactyl.onnx` | ONNX-экспорт для Unity (opset 17) |
| `rsl_dactyl_config.json` | Рантайм-конфиг ONNX-модели (вход/выход/классы/нормализация) |
| `features_data/` | Извлечённые признаки (`.npy`) + `metadata.npy` |
| `data/original/`, `data/trimmed/` | Исходные и обрезанные видео |

---

## 2. Данные

- Аннотации: `data/annotations.tsv` (таб-разделённый). Колонки:
  `attachment_id, user_id, text, begin, end, height, width, train, length`.
- **3 862** размеченных клипа, **155** уникальных пользователей.
- **34** метки: 33 буквы русского алфавита + служебный класс **`no_event`** (жест не распознан / нет жеста).
- Официальное разбиение (`train` колонка):
  - train = **3 182** (≈82%)
  - test  = **680** (≈18%), сбалансированный: **по 20 сэмплов на каждый из 34 классов**.
- Классы букв по ~101–125 клипов, из них 20 в test (у букв test = 33×20 = **660**).
- Видео: `data/original/` = 3 757 файлов, `data/trimmed/` = 3 862 файла (обрезка по `begin`–`end`).

> ⚠️ Класс `no_event` **отбрасывается** в `preprocess.py`, т.к. отсутствует в `CLASSES`
> (там только 33 буквы). Для реального времени `no_event` — это полезный rejection-класс,
> см. раздел «Риски».

---

## 3. Пайплайн

```
data/trimmed/*.mp4 ──► preprocess.py ──► features_data/*.npy + metadata.npy
                                              │
                                              ▼
                                       train.py ──► best_model_bukva.pth + results/
                                              │
                        ┌─────────────────────┼─────────────────────┐
                        ▼                     ▼                     ▼
                  demo.py (realtime)   pth_to_onnx.py         personalization.py
                                       (→ rsl_dactyl.onnx)    (→ profiles/*.pt)
```

### 3.1 `preprocess.py` — извлечение признаков
- MediaPipe Hands (`min_detection_confidence=0.5`, `min_tracking_confidence=0.5`, `max_num_hands=2`).
- Каждый кадр → тензор `[2 руки × 21 ландмарк × 3 (x,y,z)] = 126` чисел.
- Нормализация руки в `normalize_landmarks()`:
  1. центр в запястье (ландмарк 0) → `coords - wrist`;
  2. масштаб = норма вектора до ландмарка 9 (MCP среднего пальца);
  3. деление на масштаб, clip в `[-5.0, 5.0]`, flatten.
- Фильтр: видео с >50% пустых кадров (рука не найдена) отбрасывается.
- Распараллеливание: `ProcessPoolExecutor` (отдельный экземпляр `Hands` на воркер — требование MediaPipe).
- Результат: `features_data/<id>.npy` (форма `[T, 126]`) + `features_data/metadata.npy`
  (`paths, labels, is_train, label_map`). Train-часть перемешивается, test — нет.

### 3.2 `train.py` — обучение
- `SignDataset`: загрузка `.npy`, аугментация (гаусс-шум σ=0.005, масштаб ×[0.9, 1.1]);
  обрезка по центру / паддинг до `SEQUENCE_LENGTH=80` + булева padding-маска.
- Лосс: `CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)` (классовые веса через `bincount`).
- Оптимизатор: `AdamW`, weight decay 1e-2 только для ненормирующих параметров
  (`norm`, `bias`, `cls_token` → decay 0).
- Шедулер: `OneCycleLR` (pct_start=0.1, cos, div_factor=10, final_div_factor=100).
- Gradient clipping: `max_norm=1.0`. Early stopping: `patience=20` по **weighted F1** на валидации.
- Чекпойнт сохраняет: `model_state_dict, model_config, epoch, best_f1`.
- Результаты: `results/training_history.json`, `results/evaluation_history.json`,
  `results/classification_report.txt`, 15 PNG (`plot.py`).

### 3.3 `demo.py` — реальное время
- Веб-камера `cv2.VideoCapture(1)` (⚠️ индекс захардкожен), зеркалирование кадра.
- Скользящее окно `deque(maxlen=80)`, инференс при ≥40 кадрах.
- `SubtitleManager`: буква фиксируется после `STABILITY_FRAMES=40` одинаковых предсказаний
  при уверенности `CONFIDENCE_THRESH=0.70`.
- Управление: `Q` выход, `SPACE` пробел, `BACKSPACE` удалить символ, `C` очистить.

### 3.4 `pth_to_onnx.py` — экспорт для Unity
- `InferenceWrapper`: вход `[1, 80, 126]`, **без padding-маски** (Unity заполняет окно
  80 реальными кадрами). Выход `[1, 33]` = logits.
- `torch.onnx.export` opset **17**, `do_constant_folding=True`.
- Проверка эквивалентности PyTorch↔ONNX через `onnxruntime` (top-1 класс и max |Δlogits|).
- Пишет `rsl_dactyl_config.json` (input/output имена, форма, классы, параметры нормализации).

### 3.5 `personalization.py` — персональная адаптация
- Базовая модель замораживается. `UserAdapter`: `down(128→16) → GELU → up(16→128)` (инициализация up в 0)
  + обучаемый `bias` размера 33. Residual: `emb + adapter(emb)`, logits = `classify(...) + bias`.
- Калибровка признаков: `feat_mean` = среднее смещение кадров пользователя ×0.5 (демпфирование).
- `fit()`: адаптер обучается на буфере сэмплов (AdamW lr=1e-3, L2-якорь 1e-2, label_smoothing 0.1).
- Профили: `profiles/<user>.pt` + `profiles/<user>_mean.npy`.

### 3.6 `model_adapter.py` — мобильный экспорт
- `torch.jit.script` + `mobile_optimizer.optimize_for_mobile` → `.ptl` (Lite Interpreter).
- Опциональная dynamic quantization int8 для `Linear` (по умолчанию выключена).

---

## 4. Архитектура модели (`model.py`)

`SignLanguageTransformer`:

1. **TemporalStem**: конкатенация кадров и их межкадровой дельты (`delta`), линейная проекция
   `input_size*2 → hidden`, depthwise Conv1d(kernel=5, groups=hidden) + pointwise Conv1d,
   LayerNorm + GELU + Dropout.
2. **Позиционное кодирование**: синусоидальное (max_len = sequence_length+1 под CLS).
3. **CLS-токен**: `nn.Parameter`, `trunc_normal_(std=0.02)` (не нулевая инициализация — избегает
   симметричного градиента через головы внимания).
4. **Энкодер**: 2 блока `TransformerBlock` (`MultiheadAttention` + FFN `hidden→4·hidden→hidden`, pre-norm).
5. **Голова**: `LayerNorm → Linear(128→128) → GELU → Dropout → Linear(128→33)`.
6. Классификация по позиции CLS (`x[:, 0]`).

Гиперпараметры (`constants.py`):
```
SEQUENCE_LENGTH=80  NUM_LANDMARKS=21  NUM_HANDS=2  INPUT_SIZE=126
HIDDEN_SIZE=128  NUM_LAYERS=2  NUM_HEADS=4  DROPOUT=0.3
BATCH_SIZE=16  LEARNING_RATE=1e-4  EPOCHS=150
ff_multiplier=4  temporal_kernel_size=5
```

**Размер**: 467 617 параметров ≈ **1.82 МБ**.

---

## 5. Достигнутые результаты

Источник: `best_model_bukva.pth` (эпоха 93) / `results_gradclip7/`.

| Метрика | Значение |
|---|---|
| Top-1 Accuracy | **91.21%** |
| F1 weighted | **91.26%** |
| F1 macro | **91.26%** |
| Параметры | 467 617 (1.82 МБ) |

Слабые классы (F1 < 0.9):
| Класс | F1 | Комментарий |
|---|---|---|
| Ш | 0.6154 | худший, путается с Щ |
| Щ | 0.6190 | худший |
| Т | 0.7907 | |
| Ь | 0.8444 | |
| Л / М | 0.8649 | путаются между собой |

Остальные 27 классов — F1 ≥ 0.87, большинство > 0.92.

---

## 6. Конвенции и соглашения

- **Python 3.11**, venv: `.venv` (проверено: `torch`, `pandas`, `numpy` в `.venv`).
- Зависимости в `requirements.txt` (⚠️ нет `onnx`/`onnxruntime` — нужны только для `pth_to_onnx.py`).
- Код и сообщения — на русском; комментарии — на русском.
- Константы конфигурации — в `constants.py`, не в отдельных модулях.
- Классы заданы порядком `А..Я`; маппинг `CLASS_TO_IDX`/`IDX_TO_CLASS`/`NUM_CLASSES=33`.
- Чекпойнт всегда словарь с ключом `model_state_dict` (плюс `model_config`, `epoch`, `best_f1`).
- `model_config` сериализуется в чекпойнт (dataclass `ModelConfig`) и восстанавливается при экспорте.

---

## 7. Риски и известные проблемы (⚠️ важно для будущей разработки)

### 7.1 КРИТИЧНО: сломанный `forward()` в рабочей копии `model.py`
В рабочем дереве (`git diff model.py`) строка `return self.head(x[:, 0])` стоит **ДО** цикла энкодера:

```python
    return self.head(x[:, 0])          # ← ранний выход
    for block in self.encoder:          # ← мёртвый код
        x = block(x, padding_mask=src_key_padding_mask)
```

Последствия:
- `forward()` классифицирует **CLS-токен (константу)**, т.е. выход не зависит от входа, а энкодер не выполняется.
- Ломает **`train.py`** (переобучение даст мусор), **`demo.py`** и **`pth_to_onnx.py`** (экспорт сломанной модели).
- Закоммиченная версия (`HEAD:model.py`) корректна: цикл энкодера → затем `return self.head(x[:, 0])`.
- `personalization.py` использует `embed()`/`classify()` (корректны), поэтому формально работает, но
  **зависит от этой рабочей версии** `model.py` — закоммиченная версия методов `embed()/classify()` не содержит.

**Фикс**: перенести `return self.head(x[:, 0])` **после** цикла `for block in self.encoder`.
Артефакты `best_model_bukva.pth` (июнь) и `rsl_dactyl.onnx` (сентябрь) созданы **до** регрессии (октябрь) и корректны.

### 7.2 Класс `no_event` не используется
`CLASSES` содержит только 33 буквы; `no_event` (105 клипов) отбрасывается в `preprocess.py`.
Для продакшена realtime-системы это потерянный rejection-сигнал («жест не распознан / нет руки»).

### 7.3 Захардкоженные значения
- `demo.py`: `cv2.VideoCapture(1)` — индекс камеры; шрифт `/usr/share/fonts/...` — путь Linux,
  на Windows fallback-шрифт может не отрисовать кириллицу.
- Пути/константы частично в `constants.py`, частично захардкожены в модулях (напр. `CHECKPOINT_PATH`
  в `model_adapter.py`, `demo.py`).

### 7.4 `personalization.py` — незавершённые места
- `PersonalizedModel.load()` не возвращает признак успеха/ошибки, если профиля нет.
- Нет CLI/точки входа `if __name__ == "__main__"` — модуль только импортируется.
- `json` импортирован, но не используется.

### 7.5 Требования неполны
`requirements.txt` не включает `onnx`, `onnxruntime`, `scikit-learn`, `seaborn` — они нужны
для `pth_to_onnx.py` и `plot.py`/`train.py`.

---

## 8. Быстрый старт

```bash
# активация venv
.venv/Scripts/activate            # Windows

# 1) признаки (при отсутствии features_data/metadata.npy)
python preprocess.py

# 2) обучение
python train.py

# 3) реальное время
python demo.py

# 4) экспорт ONNX
python pth_to_onnx.py

# 5) мобильный экспорт
python model_adapter.py
```

> Перед любым переобучением/инференсом/экспортом — **исправить баг `forward()` из раздела 7.1**.
