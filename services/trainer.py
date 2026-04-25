import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from schemas.churn import TrainingConfigChurn  
from services.feature_schema import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, FEATURE_ORDER

DATA_PATH = Path("data/churn_dataset.csv")

# реестр доступных моделей — добавлять новые сюда
MODEL_REGISTRY = {
    "logreg": LogisticRegression,
    "random_forest": RandomForestClassifier,
}

# дефолтные гиперпараметры для каждой модели
DEFAULT_HYPERPARAMETERS = {
    "logreg": {"max_iter": 1000, "random_state": 42},
    "random_forest": {"n_estimators": 100, "random_state": 42},
}

def build_pipeline(config: TrainingConfigChurn) -> Pipeline:
    """Собирает sklearn Pipeline с предобработанной моделью"""
    if config.model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {config.model_type}. Supported types: {list(MODEL_REGISTRY.keys())}")

    params = {
        **DEFAULT_HYPERPARAMETERS[config.model_type],
        **config.hyperparameters,
    }
    classifier = MODEL_REGISTRY[config.model_type](**params)

    # трансформер для числовых признаков - масштабирование (StandarcScaler)
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    # трансформер для категориальных признаков - OHE
    categorical_transformer = Pipeline(steps=[
        ("scaler", OneHotEncoder(handle_unknown="ignore")),
    ])

    # объединяем обра трансформера через ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    # итоговый пайплайн: предобработка + лог регрессия
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=100, random_state=42),)
    ])

    return pipeline

def train_churn_model(
        config: TrainingConfigChurn,
        test_size: float = 0.2,
        random_state: int = 42,
) -> dict:
    """
    Читает датасет, обучает Pipeline считает метрики на тестовой выборке.
    Возвращает обученный pipeline и словарь с метриками и конфигом
    """
    # проверяем наличие файла
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Файл не найден: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)

    # проверяем что датасет не пустой
    if df.empty:
        raise ValueError("Датасет пуст")
    
    # проверяем наличие всех нужных столбцов
    required_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"В датасете отсутствуют столбцы: {missing}")
    
    X = df[FEATURE_ORDER]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline = build_pipeline(config)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    final_params = {
        **DEFAULT_HYPERPARAMETERS[config.model_type],
        **config.hyperparameters,
    }

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    return {"pipeline": pipeline, "metrics": metrics, "model_type": config.model_type, "hyperparameters": final_params}