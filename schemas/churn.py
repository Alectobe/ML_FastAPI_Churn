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
    churn_prediction: int = Field(..., description="0 — остался, 1 — ушёл")
    churn_label: str = Field(..., description="'stayed' или 'churned'")
    probability_stayed: float = Field(..., description="Вероятность класса 0")
    probability_churned: float = Field(..., description="Вероятность класса 1")


class TrainingConfigChurn(BaseModel):
    model_type: str = "logreg"
    hyperparameters: dict = {}