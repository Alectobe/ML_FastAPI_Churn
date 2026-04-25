import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
from services.preprocessing import get_split_info
from core.logger import setup_logger

logger = setup_logger("data")
router = APIRouter(prefix="/dataset", tags=["dataset"])

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "churn_dataset.csv"


def load_dataframe() -> pd.DataFrame:
    if not DATA_PATH.exists():
        logger.error(f"Файл датасета не найден: {DATA_PATH}")
        raise HTTPException(
            status_code=404,
            detail=f"Файл датасета не найден: {DATA_PATH}"
        )
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Датасет загружен: {len(df)} строк, {len(df.columns)} столбцов")
    return df


@router.get("/preview")
def preview_dataset(n: int = Query(default=5, ge=1, le=100)):
    logger.info(f"Запрос превью датасета: n={n}")
    df = load_dataframe()
    return df.head(n).to_dict(orient="records")


@router.get("/info")
def dataset_info():
    logger.info("Запрос информации о датасете")
    df = load_dataframe()
    churn_distribution = df["churn"].value_counts().to_dict()
    return {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": df.columns.tolist(),
        "churn_distribution": {
            "stayed": churn_distribution.get(0, 0),
            "churned": churn_distribution.get(1, 0),
        }
    }


@router.get("/split-info")
def split_info(
    test_size: float = Query(default=0.2, ge=0.1, le=0.5),
    random_state: int = Query(default=42, ge=0),
):
    logger.info(f"Запрос split-info: test_size={test_size}, random_state={random_state}")
    return get_split_info(test_size=test_size, random_state=random_state)