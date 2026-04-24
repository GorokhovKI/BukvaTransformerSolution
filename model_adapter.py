import torch
import torch.utils.mobile_optimizer

from model import SignLanguageTransformer

CHECKPOINT_PATH = "best_model_bukva.pth"
OUTPUT_PATH = "rsl_model_mobile.ptl"
USE_DYNAMIC_QUANTIZATION = False


def export() -> None:
    device = torch.device("cpu")
    print(f"Инициализация модели на {device}...")
    model = SignLanguageTransformer().to(device)

    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("Веса загружены.")
    except Exception as error:
        print(f"Ошибка загрузки весов: {error}")
        return

    model.eval()

    export_model = model
    if USE_DYNAMIC_QUANTIZATION:
        export_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        print("Применена dynamic quantization (int8 Linear).")

    print("Экспорт через torch.jit.script...")
    try:
        scripted_model = torch.jit.script(export_model)
        optimized_model = torch.utils.mobile_optimizer.optimize_for_mobile(scripted_model)
        optimized_model._save_for_lite_interpreter(OUTPUT_PATH)
        print(f"Успех! Модель сохранена в: {OUTPUT_PATH}")

    except Exception as error:
        print(f"Критическая ошибка экспорта: {error}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    export()
