from fastapi import FastAPI
from schemas.churn import FeatureVectorChurn
from routers import data

app = FastAPI(
    title="ML Churn Service",
    description="Сервис для предсказания оттока клиентов",
    version="0.1.0"
)

app.include_router(data.router)

@app.get("/")
def root():
    return {"message": "ML churn service is running"} 

@app.post("/predict")
def predict(features: FeatureVectorChurn):
    return features