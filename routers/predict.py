import pandas as pd
from fastapi import APIRouter
from schemas.churn import FeatureVectorChurn
from services.feature_schema import FEATURE_ORDER
from core.errors import ModelNotTrainedError, InvalidFeaturesError
from core.logger import setup_logger
from routers import model as model_router

logger = setup_logger("predict")
router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("")
def predict(features: FeatureVectorChurn):
    """
    Принимает признаки клиента и возвращает предсказание churn и вероятность.

    Возможные ошибки:
    - `MODEL_NOT_TRAINED` (503) — модель ещё не обучена
    - `VALIDATION_ERROR` (422) — некорректные типы признаков
    """
    if model_router.trained_pipeline is None:
        logger.warning("Запрос предсказания без обученной модели")
        raise ModelNotTrainedError()

    try:
        # собираем DataFrame строго в порядке FEATURE_ORDER
        input_df = pd.DataFrame([features.model_dump()])[FEATURE_ORDER]
    except KeyError as e:
        logger.error(f"Ошибка формирования признаков: {e}")
        raise InvalidFeaturesError(
            message=f"Не удалось сформировать признаки для предсказания: {e}",
            details={"missing_feature": str(e)},
        )

    churn_proba = model_router.trained_pipeline.predict_proba(input_df)[0][1]
    churn_label = int(churn_proba >= 0.5)

    logger.info(
        f"Предсказание: churn={churn_label}, probability={round(churn_proba, 4)}"
    )

    return {
        "churn": churn_label,
        "probability": round(churn_proba, 4),
    }