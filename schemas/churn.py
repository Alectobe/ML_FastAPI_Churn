from pydantic import BaseModel, Field


class FeatureVectorChurn(BaseModel):
    monthly_fee: float = Field(..., examples=[75.5])
    usage_hours: float = Field(..., examples=[120.0])
    support_requests: int = Field(..., examples=[2])
    account_age_months: int = Field(..., examples=[24])
    failed_payments: int = Field(..., examples=[0])
    region: str = Field(..., examples=["North"])
    device_type: str = Field(..., examples=["Mobile"])
    payment_method: str = Field(..., examples=["Credit Card"])
    autopay_enabled: int = Field(..., examples=[1])


class DatasetRowChurn(FeatureVectorChurn):
    churn: int


class PredictionResponseChurn(BaseModel):
    churn_prediction: int = Field(
        ..., description="0 — клиент остался, 1 — клиент ушёл"
    )
    churn_label: str = Field(
        ..., description="'stayed' или 'churned'"
    )
    probability_stayed: float = Field(
        ..., description="Вероятность что клиент останется (класс 0)"
    )
    probability_churned: float = Field(
        ..., description="Вероятность что клиент уйдёт (класс 1)"
    )
