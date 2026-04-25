import pytest
import pandas as pd
from unittest.mock import patch
from schemas.churn import TrainingConfigChurn
from services.trainer import train_churn_model, build_pipeline
from core.errors import DatasetNotFoundError, DatasetEmptyError, UnknownModelTypeError
from tests.fixtures import SYNTHETIC_DATA


def _train_on_synthetic(model_type="logreg", hyperparameters=None, tmp_path=None):
    """Вспомогательная функция: обучает модель на синтетических данных."""
    csv_path = tmp_path / "churn_dataset.csv"
    SYNTHETIC_DATA.to_csv(csv_path, index=False)

    config = TrainingConfigChurn(
        model_type=model_type,
        hyperparameters=hyperparameters or {},
    )

    with patch("services.trainer.DATA_PATH", csv_path):
        return train_churn_model(config=config, test_size=0.3, random_state=42)


def test_train_logreg_returns_pipeline(tmp_path):
    """Обучение logreg возвращает pipeline и метрики."""
    result = _train_on_synthetic(tmp_path=tmp_path)

    assert "pipeline" in result
    assert "metrics" in result
    assert result["model_type"] == "logreg"


def test_train_random_forest(tmp_path):
    """Обучение random_forest завершается без ошибок."""
    result = _train_on_synthetic(model_type="random_forest", tmp_path=tmp_path)

    assert result["model_type"] == "random_forest"


def test_metrics_keys_present(tmp_path):
    """В метриках присутствуют все ожидаемые ключи."""
    result = _train_on_synthetic(tmp_path=tmp_path)
    metrics = result["metrics"]

    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert "train_size" in metrics
    assert "test_size" in metrics


def test_metrics_values_in_range(tmp_path):
    """Все метрики находятся в допустимом диапазоне [0, 1]."""
    result = _train_on_synthetic(tmp_path=tmp_path)
    metrics = result["metrics"]

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_dataset_not_found_raises():
    """Если файл датасета не существует — бросается DatasetNotFoundError."""
    from pathlib import Path
    config = TrainingConfigChurn()

    with patch("services.trainer.DATA_PATH", Path("/nonexistent/path.csv")):
        with pytest.raises(DatasetNotFoundError):
            train_churn_model(config=config)


def test_empty_dataset_raises(tmp_path):
    """Если датасет пустой — бросается DatasetEmptyError."""
    csv_path = tmp_path / "empty.csv"
    # пишем только заголовок без строк
    pd.DataFrame(columns=SYNTHETIC_DATA.columns).to_csv(csv_path, index=False)

    config = TrainingConfigChurn()

    with patch("services.trainer.DATA_PATH", csv_path):
        with pytest.raises(DatasetEmptyError):
            train_churn_model(config=config)


def test_unknown_model_type_raises(tmp_path):
    """Неизвестный model_type бросает UnknownModelTypeError."""
    csv_path = tmp_path / "churn_dataset.csv"
    SYNTHETIC_DATA.to_csv(csv_path, index=False)

    config = TrainingConfigChurn(model_type="xgboost")

    with patch("services.trainer.DATA_PATH", csv_path):
        with pytest.raises(UnknownModelTypeError):
            train_churn_model(config=config)


def test_pipeline_predict_after_train(tmp_path):
    """Обученный pipeline принимает X и возвращает предсказания правильной длины."""
    from services.feature_schema import FEATURE_ORDER
    result = _train_on_synthetic(tmp_path=tmp_path)

    pipeline = result["pipeline"]
    X = SYNTHETIC_DATA[FEATURE_ORDER]
    predictions = pipeline.predict(X)

    assert len(predictions) == len(X)
    assert set(predictions).issubset({0, 1})