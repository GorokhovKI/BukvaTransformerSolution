import torch
import torch.utils.mobile_optimizer
from model import SignLanguageTransformer
CHECKPOINT_PATH = "best_model_bukva.pth"
OUTPUT_PATH = "rsl_model_mobile.ptl"


def export():
    device = torch.device('cpu')
    print(f"Инициализация модели на {device}...")
    model = SignLanguageTransformer().to(device)

    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Веса загружены.")
    except Exception as e:
        print(f"Ошибка загрузки весов: {e}")
        return

    model.eval()

    print("Экспорт через torch.jit.script...")
    try:
        scripted_model = torch.jit.script(model)
        optimized_model = torch.utils.mobile_optimizer.optimize_for_mobile(scripted_model)
        optimized_model._save_for_lite_interpreter(OUTPUT_PATH)
        print(f"Успех! Модель сохранена в: {OUTPUT_PATH}")

    except Exception as e:
        print(f"Критическая ошибка экспорта: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    export()