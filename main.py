from fastapi import FastAPI

app = FastAPI(
    title="ML Churn Service",
    description="Сервис для предсказания оттока клиентов",
    version="0.1.0"
)

@app.get("/")
def root():
    return {"message": "ML churn service is running"} 