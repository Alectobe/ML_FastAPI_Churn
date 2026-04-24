from fastapi import APIRouter, HTTPException, Query
from services.trainer import train_churn_model

router = APIRouter(prefix="/model", tags=["model"])

# глобальное хранилище обученного Pipeline в памяти приложения
# на следующих днях заменим на сохранение в файл
trained_pipeline = None

@router.post("/train")
def train_model(
    test_size: float = Query(default=0.2, ge=0.1, le=0.5),
    random_state: int = Query(default=42, ge=0),
):
    """Обучает модель на churn_dataset.csv и возвращает метрики"""
    global trained_pipeline

    try:
        result = train_churn_model(test_size=test_size, random_state=random_state)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    trained_pipeline = result['pipeline']

    return {
        "status": "model trained successfully",
        "metrics": result['metrics'],
    }