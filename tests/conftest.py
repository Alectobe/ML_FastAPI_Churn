import pytest
import services.model_score
from fastapi.testclient import TestClient
from main import app
from routers import model as model_router
from tests.fixtures import SYNTHETIC_DATA


@pytest.fixture
def client(tmp_path, monkeypatch):
    """
    TestClient с чистым состоянием — модель не загружается с диска.
    Подменяем пути к файлам модели на пустую tmp_path.
    """
    # указываем на несуществующие файлы — init_model найдёт что грузить нечего
    monkeypatch.setattr("services.model_score.MODEL_PATH", tmp_path / "churn_model.joblib")
    monkeypatch.setattr("services.model_score.META_PATH", tmp_path / "churn_model_meta.joblib")

    model_router.trained_pipeline = None

    with TestClient(app) as c:
        yield c


@pytest.fixture
def trained_client(tmp_path, monkeypatch):
    """
    TestClient с обученной моделью на синтетических данных.
    Подменяет пути к файлам через monkeypatch — не трогает реальные файлы.
    """
    csv_path = tmp_path / "churn_dataset.csv"
    SYNTHETIC_DATA.to_csv(csv_path, index=False)

    model_path = tmp_path / "churn_model.joblib"
    meta_path = tmp_path / "churn_model_meta.joblib"

    monkeypatch.setattr("services.trainer.DATA_PATH", csv_path)
    monkeypatch.setattr("services.model_score.MODEL_PATH", model_path)
    monkeypatch.setattr("services.model_score.META_PATH", meta_path)

    model_router.trained_pipeline = None

    with TestClient(app) as c:
        response = c.post("/model/train")
        assert response.status_code == 200, f"Обучение не прошло: {response.json()}"
        yield c