import pytest
from tests.fixtures import VALID_PREDICT_PAYLOAD


# --- тесты базового состояния ---

def test_root(client):
    """GET / возвращает статус сервиса."""
    response = client.get("/")
    assert response.status_code == 200
    assert "ml churn service is running" in response.json()["message"].lower()


def test_model_status_not_trained(client):
    """GET /model/status возвращает is_trained=False если модель не обучена."""
    response = client.get("/model/status")
    assert response.status_code == 200
    assert response.json()["is_trained"] is False


# --- тесты обучения ---

def test_train_default_config(trained_client):
    """POST /model/train с дефолтным конфигом возвращает метрики."""
    response = trained_client.post("/model/train", json={})
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "model trained successfully"
    assert "metrics" in body
    assert "accuracy" in body["metrics"]
    assert "roc_auc" in body["metrics"]


def test_train_random_forest(trained_client):
    """POST /model/train с random_forest отрабатывает без ошибок."""
    response = trained_client.post("/model/train", json={
        "model_type": "random_forest",
        "hyperparameters": {"n_estimators": 10},
    })
    assert response.status_code == 200
    assert response.json()["model_type"] == "random_forest"


def test_train_unknown_model_type(trained_client):
    """POST /model/train с неизвестным model_type возвращает 422."""
    response = trained_client.post("/model/train", json={"model_type": "xgboost"})
    assert response.status_code == 422
    assert response.json()["code"] == "UNKNOWN_MODEL_TYPE"


# --- тесты статуса после обучения ---

def test_model_status_after_train(trained_client):
    """GET /model/status возвращает is_trained=True после обучения."""
    response = trained_client.get("/model/status")
    assert response.status_code == 200

    body = response.json()
    assert body["is_trained"] is True
    assert body["trained_at"] is not None
    assert body["metrics"] is not None


# --- тесты предсказания ---

def test_predict_single(trained_client):
    """POST /predict с одним объектом возвращает PredictionResponseChurn."""
    response = trained_client.post("/predict", json=VALID_PREDICT_PAYLOAD)
    assert response.status_code == 200

    body = response.json()
    assert body["churn_prediction"] in [0, 1]
    assert body["churn_label"] in ["stayed", "churned"]
    assert 0.0 <= body["probability_stayed"] <= 1.0
    assert 0.0 <= body["probability_churned"] <= 1.0
    assert round(body["probability_stayed"] + body["probability_churned"], 2) == 1.0


def test_predict_batch(trained_client):
    """POST /predict со списком объектов возвращает список PredictionResponseChurn."""
    payload = [VALID_PREDICT_PAYLOAD, VALID_PREDICT_PAYLOAD]
    response = trained_client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert isinstance(body, list)
    assert len(body) == 2
    for item in body:
        assert item["churn_prediction"] in [0, 1]
        assert item["churn_label"] in ["stayed", "churned"]
        assert 0.0 <= item["probability_stayed"] <= 1.0
        assert 0.0 <= item["probability_churned"] <= 1.0


def test_predict_without_trained_model(client):
    """POST /predict без обученной модели возвращает 503 MODEL_NOT_TRAINED."""
    response = client.post("/predict", json=VALID_PREDICT_PAYLOAD)
    assert response.status_code == 503
    assert response.json()["code"] == "MODEL_NOT_TRAINED"


def test_predict_invalid_payload(trained_client):
    """POST /predict с неверным типом поля возвращает 422 VALIDATION_ERROR."""
    bad_payload = {**VALID_PREDICT_PAYLOAD, "monthly_fee": "дорого"}
    response = trained_client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"


def test_predict_missing_field(trained_client):
    """POST /predict с отсутствующим обязательным полем возвращает 422."""
    incomplete = {k: v for k, v in VALID_PREDICT_PAYLOAD.items() if k != "region"}
    response = trained_client.post("/predict", json=incomplete)
    assert response.status_code == 422


# --- тесты датасета ---

def test_dataset_preview(client):
    """GET /dataset/preview возвращает список строк."""
    response = client.get("/dataset/preview?n=3")
    # если датасет есть — 200, если нет — 404, оба варианта корректны
    assert response.status_code in [200, 404]

    if response.status_code == 200:
        body = response.json()
        assert isinstance(body, list)
        assert len(body) <= 3


def test_dataset_preview_invalid_n(client):
    """GET /dataset/preview?n=0 возвращает 422."""
    response = client.get("/dataset/preview?n=0")
    assert response.status_code == 422


# --- тесты метрик и схемы ---

def test_model_metrics_empty_history(client):
    """GET /model/metrics без истории возвращает корректную структуру."""
    response = client.get("/model/metrics")
    assert response.status_code == 200

    body = response.json()
    assert "last_training" in body
    assert "history" in body
    assert isinstance(body["history"], list)


def test_model_schema(client):
    """GET /model/schema возвращает feature_order и example_request."""
    response = client.get("/model/schema")
    assert response.status_code == 200

    body = response.json()
    assert "feature_order" in body
    assert "example_request" in body
    assert len(body["feature_order"]) > 0