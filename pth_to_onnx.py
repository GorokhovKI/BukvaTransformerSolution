from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

CHECKPOINT_PATH = Path("best_model_bukva.pth")
MODEL_PATH = Path("model.py")  # Можно указать Path("model.py")
ONNX_PATH = Path("rsl_dactyl.onnx")
OPSET_VERSION = 17


# В checkpoint сохранён объект ModelConfig.
def load_model_module(model_path: Path):
    if not model_path.is_file():
        raise FileNotFoundError(f"Не найден файл архитектуры: {model_path.resolve()}")

    spec = importlib.util.spec_from_file_location("model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Не удалось импортировать модуль: {model_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["model"] = module
    spec.loader.exec_module(module)
    return module


class InferenceWrapper(nn.Module):
    """Экспортирует модель с одним входом [batch, 80, INPUT_SIZE].

    В Unity будет использоваться скользящее окно из 80 настоящих кадров,
    поэтому маска padding не нужна и не становится входом ONNX-модели.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, landmark_sequence: torch.Tensor) -> torch.Tensor:
        return self.model(landmark_sequence, src_key_padding_mask=None)


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Не найден checkpoint: {checkpoint_path.resolve()}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint должен быть словарём с ключом 'model_state_dict'.")

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "В checkpoint отсутствует 'model_state_dict'. "
            f"Доступные ключи: {list(checkpoint.keys())}"
        )

    return checkpoint


def save_runtime_config(onnx_path: Path, constants: Any) -> Path:
    config = {
        "model_file": onnx_path.name,
        "input_name": "landmark_sequence",
        "output_name": "logits",
        "input_dtype": "float32",
        "input_shape": [1, constants.SEQUENCE_LENGTH, constants.INPUT_SIZE],
        "sequence_length": constants.SEQUENCE_LENGTH,
        "num_hands": constants.NUM_HANDS,
        "num_landmarks": constants.NUM_LANDMARKS,
        "coordinates_per_landmark": 3,
        "features_per_frame": constants.INPUT_SIZE,
        "normalization": {
            "wrist_landmark_index": 0,
            "scale_landmark_index": 9,
            "clip_min": -5.0,
            "clip_max": 5.0,
        },
        "classes": {
            str(index): label
            for index, label in enumerate(constants.CLASSES)
        },
        "note": (
            "Input shape is [batch, frames, features]. "
            "For Unity, fill a sliding window with exactly 80 real frames; "
            "the ONNX model has no padding-mask input."
        ),
    }

    config_path = onnx_path.with_name(f"{onnx_path.stem}_config.json")
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return config_path


def verify_with_onnxruntime(
    model: nn.Module,
    onnx_path: Path,
    example_input: torch.Tensor,
) -> None:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("Проверка пропущена. Для неё установите: pip install onnx onnxruntime")
        return

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    with torch.inference_mode():
        pytorch_logits = model(example_input).cpu().numpy()

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    onnx_logits = session.run(
        ["logits"],
        {"landmark_sequence": example_input.cpu().numpy()},
    )[0]

    pytorch_class = int(np.argmax(pytorch_logits, axis=1)[0])
    onnx_class = int(np.argmax(onnx_logits, axis=1)[0])
    max_abs_error = float(np.max(np.abs(pytorch_logits - onnx_logits)))

    print("\nПроверка ONNX Runtime: OK")
    print(f"PyTorch top-1 class index: {pytorch_class}")
    print(f"ONNX top-1 class index:    {onnx_class}")
    print(f"Максимальная разница logits: {max_abs_error:.8f}")

    if pytorch_class != onnx_class:
        raise RuntimeError(
            "Top-1 классы PyTorch и ONNX не совпали. Не используйте этот ONNX."
        )


def main() -> None:
    # constants.py должен лежать рядом со скриптом.
    import constants

    model_module = load_model_module(MODEL_PATH)
    checkpoint = load_checkpoint(CHECKPOINT_PATH)

    saved_config = checkpoint.get("model_config")
    if saved_config is None:
        saved_config = model_module.ModelConfig()

    model = model_module.SignLanguageTransformer(saved_config).cpu()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    inference_model = InferenceWrapper(model).eval()

    sequence_length = int(constants.SEQUENCE_LENGTH)  # 80
    features_per_frame = int(constants.INPUT_SIZE)
    num_classes = int(constants.NUM_CLASSES)           # 33

    example_input = torch.randn(
        1,
        sequence_length,
        features_per_frame,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        test_logits = inference_model(example_input)

    expected_shape = (1, num_classes)
    if tuple(test_logits.shape) != expected_shape:
        raise RuntimeError(
            f"Неожиданная форма выхода: {tuple(test_logits.shape)}. "
            f"Ожидалось: {expected_shape}."
        )

    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {CHECKPOINT_PATH.resolve()}")
    print(f"Архитектура: {MODEL_PATH.resolve()}")
    print(f"ONNX:       {ONNX_PATH.resolve()}")
    print(f"Вход:       [1, {sequence_length}, {features_per_frame}]")
    print(f"Выход:      [1, {num_classes}]")

    torch.onnx.export(
        inference_model,
        (example_input,),
        str(ONNX_PATH),
        input_names=["landmark_sequence"],
        output_names=["logits"],
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
    )

    if not ONNX_PATH.is_file() or ONNX_PATH.stat().st_size == 0:
        raise RuntimeError("ONNX-файл не создан или имеет нулевой размер.")

    config_path = save_runtime_config(ONNX_PATH, constants)

    print(f"\nONNX успешно создан: {ONNX_PATH.stat().st_size / 1024 / 1024:.2f} MiB")
    print(f"Конфигурация создана: {config_path}")

    verify_with_onnxruntime(inference_model, ONNX_PATH, example_input)


if __name__ == "__main__":
    main()