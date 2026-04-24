import joblib
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
META_PATH = BASE_DIR / "models" / "churn_model_meta.joblib"

def save_churn_model(pipeline, metrics: dict) -> None:
    """Сохраняет обученный pipeline и метаданные на диск"""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # сохраняем сам joblib
    joblib.dump(pipeline, MODEL_PATH)

    # сохраняем метаданные: время обучения и метрики
    meta = {
        "trained_at": datetime.now().isoformat(),
        "metrics": metrics,
    }
    joblib.dump(meta, META_PATH)

def load_churn_model() -> dict | None:
    """
    Загружает pipeline и метаданные с диска
    Возвращает None если файлы не найдены
    """
    if not MODEL_PATH.exists() or not META_PATH.exists():
        return None
    
    pipeline = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)

    return {"pipeline": pipeline, "meta": meta}

def get_model_status() -> dict:
    """
    Возвращает статус модели: обучена ли, когда, с какими метриками
    """
    if not MODEL_PATH.exists() or not META_PATH.exists():
        return {
            "is_trained": False,
            "trained_at": None,
            "metrics": None,
            "model_path": str(MODEL_PATH),
        }
    meta = joblib.load(META_PATH)

    return {
        "is_trained": True,
        "trained_at": meta['trained_at'],
        "metrics": meta['metrics'],
        "model_path": str(MODEL_PATH),
    }