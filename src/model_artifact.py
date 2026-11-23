"""Store and load the CV-selected pipeline used by evaluation and prediction."""

import os
import tempfile

import joblib

from config import BEST_MODEL_PATH


def save_best_model(model, model_name):
    """Publish a complete pipeline atomically and return its reloaded instance."""
    model.selected_model_name_ = model_name
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        dir=BEST_MODEL_PATH.parent, suffix=".joblib",
    )
    os.close(descriptor)
    try:
        joblib.dump(model, temporary_path, compress=3)
        restored = joblib.load(temporary_path)
        os.replace(temporary_path, BEST_MODEL_PATH)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    return restored


def load_best_model():
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy mô hình CV tại {BEST_MODEL_PATH}. "
            "Hãy chạy python src/tune_models.py hoặc python src/compare_models.py."
        )
    return joblib.load(BEST_MODEL_PATH)
