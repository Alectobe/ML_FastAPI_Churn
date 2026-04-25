from fastapi import APIRouter, HTTPException, Query
from schemas.churn import TrainingConfigChurn
from services.trainer import train_churn_model
from services.model_score import save_churn_model, load_churn_model, get_model_status
from services.feature_schema import FEATURE_ORDER, FEATURE_SCHEMA, EXAMPLE_REQUEST

router = APIRouter(prefix="/model", tags=["model"])

# глобальное хранилище обученного Pipeline в памяти приложения
# на следующих днях заменим на сохранение в файл
trained_pipeline = None

def init_model() -> None:
    """
    Пробует загрузить модель с диска при старте приложения
    Вызывается из lifespan в main.py
    """
    global trained_pipeline
    result = load_churn_model()
    if result is not None:
        trained_pipeline = result['pipeline']
        print(f"[model] Модель полностью загружена с диска: {result['meta']['trained_at']}")
    else:
        print("[model] Сохранённая модель не найдена, треюуется обучение.")


@router.post("/train")
def train_model(
    config: TrainingConfigChurn = TrainingConfigChurn(),
    test_size: float = Query(default=0.2, ge=0.1, le=0.5),
    random_state: int = Query(default=42, ge=0),
):
    """
    Обучает модель на churn_dataset.csv и сохраняет на диск.

    Возможные ошибки:
    - `DATASET_NOT_FOUND` (404) — файл churn_dataset.csv не найден
    - `DATASET_EMPTY` (422) — датасет пуст
    - `UNKNOWN_MODEL_TYPE` (422) — передан неизвестный model_type
    - `VALIDATION_ERROR` (422) — некорректные типы параметров
    """
    global trained_pipeline

    result = train_churn_model(config=config, test_size=test_size, random_state=random_state)

    trained_pipeline = result["pipeline"]

    save_churn_model(
        pipeline=result["pipeline"],
        metrics=result["metrics"],
        model_type=result["model_type"],
        hyperparameters=result["hyperparameters"],
    )

    return {
        "status": "model trained successfully",
        "model_type": result["model_type"],
        "hyperparameters": result["hyperparameters"],
        "metrics": result["metrics"],
    }

@router.get("/status")
def model_status():
    """Показывает статус модели: обучена ли, когда и с какими метриками."""
    return get_model_status()

@router.get("/schema")
def model_schema():
    """Показывает JSON-схему для объекта TrainingConfigChurn, который нужно отправлять на /train"""
    return {
        "feature_order": FEATURE_ORDER,
        "features": FEATURE_SCHEMA,
        "example_request": EXAMPLE_REQUEST,
    }