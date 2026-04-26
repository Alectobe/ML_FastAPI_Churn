import pandas as pd
from typing import Union, List
from fastapi import APIRouter, Body
from schemas.churn import FeatureVectorChurn, PredictionResponseChurn
from services.feature_schema import FEATURE_ORDER
from core.errors import ModelNotTrainedError, InvalidFeaturesError
from core.logger import setup_logger
from routers import model as model_router

logger = setup_logger("predict")
router = APIRouter(prefix="/predict", tags=["predict"])

_SINGLE_EXAMPLE = {
    "monthly_fee": 75.5, "usage_hours": 120.0, "support_requests": 2,
    "account_age_months": 24, "failed_payments": 0, "region": "europe",
    "device_type": "mobile", "payment_method": "card", "autopay_enabled": 1,
}

_BATCH_EXAMPLE = [
    {"monthly_fee": 95.0, "usage_hours": 30.0, "support_requests": 5,
     "account_age_months": 6, "failed_payments": 3, "region": "asia",
     "device_type": "desktop", "payment_method": "paypal", "autopay_enabled": 0},
    {"monthly_fee": 45.0, "usage_hours": 200.0, "support_requests": 0,
     "account_age_months": 48, "failed_payments": 0, "region": "america",
     "device_type": "tablet", "payment_method": "crypto", "autopay_enabled": 1},
]


def _predict_items(items: List[FeatureVectorChurn]) -> List[PredictionResponseChurn]:
    """Прогоняет список объектов через обученный pipeline."""
    try:
        df = pd.DataFrame([item.model_dump() for item in items])[FEATURE_ORDER]
    except KeyError as e:
        logger.error(f"Ошибка формирования признаков: {e}")
        raise InvalidFeaturesError(
            message=f"Не удалось сформировать признаки: {e}",
            details={"missing_feature": str(e)},
        )

    probabilities = model_router.trained_pipeline.predict_proba(df)

    results = []
    for proba in probabilities:
        prob_stayed = round(float(proba[0]), 4)
        prob_churned = round(float(proba[1]), 4)
        prediction = int(prob_churned >= 0.5)
        results.append(PredictionResponseChurn(
            churn_prediction=prediction,
            churn_label="churned" if prediction == 1 else "stayed",
            probability_stayed=prob_stayed,
            probability_churned=prob_churned,
        ))
        logger.info(
            f"Предсказание: churn={prediction}, "
            f"prob_stayed={prob_stayed}, prob_churned={prob_churned}"
        )

    return results


@router.post(
    "",
    response_model=Union[PredictionResponseChurn, List[PredictionResponseChurn]],
    summary="Предсказание оттока клиента",
    description=(
        "Принимает **один объект** `FeatureVectorChurn` или **список** — "
        "возвращает соответственно один ответ или список `PredictionResponseChurn`.\n\n"
        "Возможные ошибки:\n"
        "- `MODEL_NOT_TRAINED` (503) — модель ещё не обучена\n"
        "- `VALIDATION_ERROR` (422) — некорректные типы признаков"
    ),
)
def predict(
    features: Union[FeatureVectorChurn, List[FeatureVectorChurn]] = Body(
        ...,
        openapi_examples={
            "single": {"summary": "Один клиент", "value": _SINGLE_EXAMPLE},
            "batch": {"summary": "Батч (список)", "value": _BATCH_EXAMPLE},
        },
    ),
) -> Union[PredictionResponseChurn, List[PredictionResponseChurn]]:
    if model_router.trained_pipeline is None:
        logger.warning("Запрос предсказания без обученной модели")
        raise ModelNotTrainedError()

    if isinstance(features, list):
        return _predict_items(features)
    return _predict_items([features])[0]
