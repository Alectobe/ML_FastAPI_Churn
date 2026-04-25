from contextlib import asynccontextmanager
from typing import Union, List

import pandas as pd
from fastapi import FastAPI, HTTPException, Body

from schemas.churn import FeatureVectorChurn, PredictionResponseChurn
from routers import data, model
from routers.model import init_model
import routers.model as model_router
from core.errors import register_error_handlers


_PREDICT_FEATURES = [
    "monthly_fee", "usage_hours", "support_requests",
    "account_age_months", "failed_payments", "autopay_enabled",
    "region", "device_type", "payment_method",
]

_SINGLE_EXAMPLE = {
    "monthly_fee": 75.5,
    "usage_hours": 120.0,
    "support_requests": 2,
    "account_age_months": 24,
    "failed_payments": 0,
    "region": "North",
    "device_type": "Mobile",
    "payment_method": "Credit Card",
    "autopay_enabled": 1,
}

_BATCH_EXAMPLE = [
    {
        "monthly_fee": 95.0,
        "usage_hours": 30.0,
        "support_requests": 5,
        "account_age_months": 6,
        "failed_payments": 3,
        "region": "South",
        "device_type": "Desktop",
        "payment_method": "Debit Card",
        "autopay_enabled": 0,
    },
    {
        "monthly_fee": 45.0,
        "usage_hours": 200.0,
        "support_requests": 0,
        "account_age_months": 48,
        "failed_payments": 0,
        "region": "East",
        "device_type": "Tablet",
        "payment_method": "PayPal",
        "autopay_enabled": 1,
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_model()
    yield


app = FastAPI(
    title="ML Churn Service",
    description="Сервис для предсказания оттока клиентов",
    version="0.1.0",
    lifespan=lifespan,
)

register_error_handlers(app)

app.include_router(data.router)
app.include_router(model.router)


@app.get("/")
def root():
    return {"message": "ML churn service is running"}


def _run_predictions(items: List[FeatureVectorChurn]) -> List[PredictionResponseChurn]:
    """Runs model inference on a list of feature vectors."""
    if model_router.trained_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Модель ещё не обучена. "
                "Сначала выполните POST /model/train."
            ),
        )

    df = pd.DataFrame([item.model_dump() for item in items])[_PREDICT_FEATURES]
    predictions = model_router.trained_pipeline.predict(df)
    probabilities = model_router.trained_pipeline.predict_proba(df)

    return [
        PredictionResponseChurn(
            churn_prediction=int(pred),
            churn_label="churned" if pred == 1 else "stayed",
            probability_stayed=round(float(proba[0]), 4),
            probability_churned=round(float(proba[1]), 4),
        )
        for pred, proba in zip(predictions, probabilities)
    ]


@app.post(
    "/predict",
    summary="Предсказание оттока клиента",
    description=(
        "Принимает **один объект** `FeatureVectorChurn` или **список** таких объектов. "
        "Возвращает предсказанный класс (`0` — остался, `1` — ушёл) "
        "и вероятности обоих классов.\n\n"
        "Если модель ещё не обучена — возвращает **503** с описанием ошибки."
    ),
    response_model=Union[PredictionResponseChurn, List[PredictionResponseChurn]],
    tags=["predict"],
)
def predict(
    features: Union[FeatureVectorChurn, List[FeatureVectorChurn]] = Body(
        ...,
        openapi_examples={
            "single_client": {
                "summary": "Один клиент",
                "description": "Передаём один объект — получаем один ответ",
                "value": _SINGLE_EXAMPLE,
            },
            "batch_clients": {
                "summary": "Несколько клиентов (батч)",
                "description": "Передаём список — получаем список ответов",
                "value": _BATCH_EXAMPLE,
            },
        },
    ),
) -> Union[PredictionResponseChurn, List[PredictionResponseChurn]]:
    if isinstance(features, list):
        return _run_predictions(features)
    return _run_predictions([features])[0]
