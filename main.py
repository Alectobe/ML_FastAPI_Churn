from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from routers import data, model, predict
from routers.model import init_model
from core.errors import register_error_handlers
from core.logger import setup_logger
from services.model_score import get_model_status

logger = setup_logger("main")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "churn_dataset.csv"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Запуск ML Churn Service...")
    init_model()
    logger.info("Сервис готов к работе.")
    yield
    logger.info("Остановка ML Churn Service.")


app = FastAPI(
    title="ML Churn Service",
    description="Сервис для предсказания оттока клиентов (churn)",
    version="0.1.0",
    lifespan=lifespan,
)

register_error_handlers(app)

app.include_router(data.router)
app.include_router(model.router)
app.include_router(predict.router)


@app.get("/")
def root():
    return {"message": "ML churn service is running"}


@app.get("/health", tags=["health"])
def health():
    """
    Проверка состояния сервиса.
    Возвращает доступность модели и датасета.
    """
    model_status = get_model_status()
    dataset_exists = DATA_PATH.exists()

    status = "ok" if model_status["is_trained"] else "degraded"

    logger.info(f"Health check: status={status}, dataset={dataset_exists}, model={model_status['is_trained']}")

    return {
        "status": status,
        "model": {
            "is_trained": model_status["is_trained"],
            "trained_at": model_status["trained_at"],
            "model_type": model_status["model_type"],
        },
        "dataset": {
            "available": dataset_exists,
            "path": str(DATA_PATH),
        },
    }