import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA_PATH = Path("data/churn_dataset.csv")

# явно задаём типы признаков на основе описания задачи
NUMERIC_FEATURES = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",
]

CATEGORICAL_FEATURES = [
    "region",
    "device_type",
    "payment_method",
]

TARGET = "churn"

def load_and_prepare(
        test_size: float = 0.2,
        random_state: int = 42,
) -> dict:
    """
    Читает датасет, обрабатывает пропуски, разбивает на train и test.
    Возвращает словарь с X_train, X_test, y_train, y_test и мета-информацией.
    """
    df = pd.read_csv(DATA_PATH)

    #Обработка пропусков
    #Числовые заполняем медианой, а категориальные - модой
    for col in NUMERIC_FEATURES:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_FEATURES:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=False)

    X = df_encoded.drop(columns=TARGET)
    y = df_encoded[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": X.columns.tolist(),
    }

def get_split_info(test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Возвращает мета-информацию о разбиении: размеры выборок
    и распределение классов churn в train и test.
    """
    result = load_and_prepare(test_size=test_size, random_state=random_state)

    y_train = result["y_train"]
    y_test = result["y_test"]

    def class_distribution(y: pd.Series) -> dict:
        counts = y.value_counts().to_dict()
        total = len(y)
        return {
            "stayed": {
                "count": counts.get(0, 0),
                "share": round(counts.get(0, 0) / total, 4),
            },
            "churned": {
                "count": counts.get(1, 0),
                "share": round(counts.get(1, 0) / total, 4),
            },
        }

    return {
        "total_rows": len(y_train) + len(y_test),
        "test_size": test_size,
        "random_state": random_state,
        "train": {
            "num_rows": len(y_train),
            "churn_distribution": class_distribution(y_train),
        },
        "test": {
            "num_rows": len(y_test),
            "churn_distribution": class_distribution(y_test),
        },
        "num_features": len(result["feature_names"]),
        "feature_names": result["feature_names"],
    }