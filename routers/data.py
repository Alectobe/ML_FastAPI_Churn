import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pathlib import Path

router = APIRouter(prefix="/dataset", tags=["dataset"])

DATA_PATH = Path("data/churn_dataset.csv")

def load_dataframe() -> pd.DataFrame:
    """Читаем CSV и возвращаем DataFrame, выкидываем 404, если файл не найден"""

    if not DATA_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Файл датасета не найден: {DATA_PATH}"
        )
    return pd.read_csv(DATA_PATH)

@router.get("/preview")
def preview_dataset(n: int = Query(default=5, ge=1, le=100)):
    """Возвращает первые N строк датасета в виде JSON"""
    df = load_dataframe()
    return df.head(n).to_dict(orient="records")

@router.get("/info")
def dataset_info():
    """Возвращает базовую информацию о датасете."""
    df = load_dataframe()

    # распределение целевой переменной churn по классам
    churn_distribution = df["churn"].value_counts().to_dict()

    return {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": df.columns.tolist(),
        "churn_distribution": {
            "stayed": churn_distribution.get(0, 0),   # churn == 0, клиент остался
            "churned": churn_distribution.get(1, 0),  # churn == 1, клиент ушёл
        }
    }
