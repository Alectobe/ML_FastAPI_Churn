from fastapi import APIRouter, Query
from schemas.churn import TrainingConfigChurn
from services.trainer import train_churn_model
from services.model_score import save_churn_model, load_churn_model, get_model_status
from services.feature_schema import FEATURE_SCHEMA, FEATURE_ORDER, EXAMPLE_REQUEST, MODEL_TYPES
from services.history_store import append_training_history, get_history, get_last_record
from core.logger import setup_logger

logger = setup_logger("model")
router = APIRouter(prefix="/model", tags=["model"])

trained_pipeline = None


def init_model() -> None:
    global trained_pipeline
    result = load_churn_model()
    if result is not None:
        trained_pipeline = result["pipeline"]
        logger.info(f"Модель загружена с диска: {result['meta']['trained_at']}")
    else:
        logger.warning("Сохранённая модель не найдена, требуется обучение.")


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

    logger.info(f"Запуск обучения: model_type={config.model_type}, test_size={test_size}")

    result = train_churn_model(
        config=config,
        test_size=test_size,
        random_state=random_state,
    )

    trained_pipeline = result["pipeline"]

    save_churn_model(
        pipeline=result["pipeline"],
        metrics=result["metrics"],
        model_type=result["model_type"],
        hyperparameters=result["hyperparameters"],
    )

    append_training_history(
        model_type=result["model_type"],
        hyperparameters=result["hyperparameters"],
        metrics=result["metrics"],
    )

    logger.info(
        f"Обучение завершено: model_type={result['model_type']}, "
        f"accuracy={result['metrics']['accuracy']}, "
        f"f1={result['metrics']['f1_score']}, "
        f"roc_auc={result['metrics']['roc_auc']}"
    )

    return {
        "status": "model trained successfully",
        "model_type": result["model_type"],
        "hyperparameters": result["hyperparameters"],
        "metrics": result["metrics"],
    }


@router.get("/status")
def model_status():
    return get_model_status()


@router.get("/schema")
def model_schema():
    return {
        "feature_order": FEATURE_ORDER,
        "features": FEATURE_SCHEMA,
        "example_request": EXAMPLE_REQUEST,
    }


@router.get("/metrics")
def model_metrics(
    model_type: str | None = Query(default=None, description=f"Фильтр по типу модели: {MODEL_TYPES}"),
    limit: int = Query(default=10, ge=1, le=100),
):
    last = get_last_record(model_type=model_type)
    history = get_history(model_type=model_type, limit=limit)
    return {
        "last_training": last,
        "history": history,
        "total_shown": len(history),
    }