import json
from datetime import datetime
from pathlib import Path

HISTORY_PATH = Path("models/training_history.json")

def _load_history() -> list:
    """Загружает историю из JSON, возвращает пустой список если файл не найден"""
    if not HISTORY_PATH.exists():
        return []
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_history(history: list) -> None:
    """Сохраняет историю в JSON"""
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def append_training_history(
        model_type: str, 
        hyperparameters: dict, 
        metrics: dict
) -> dict:
    record = {
        "trained_at": datetime.now().isoformat(),
        "model_type": model_type,
        "hyperparameters": hyperparameters,
        "metrics": metrics,
    }
    """Добавляет запись об обучении модели в историю"""
    history = _load_history()
    history.append(record)
    _save_history(history)

    return record

def get_history(model_type: str | None = None, limit: int = 10) -> list[dict]:
    """Возвращает историю обучений, можно фильтровать по model_type и ограничивать количество записей"""
    history = _load_history()
    if model_type:
        history = [record for record in history if record["model_type"] == model_type]
    return history[-limit:]

def get_last_record(model_type: str | None = None) -> dict | None:
    """Возвращает последнюю запись об обучении модели, можно фильтровать по model_type"""
    records = get_history(model_type=model_type, limit=1)
    return records[0] if records else None