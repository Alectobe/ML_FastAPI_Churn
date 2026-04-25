import pytest
import pandas as pd
from services.feature_schema import NUMERIC_FEATURES, CATEGORICAL_FEATURES, FEATURE_ORDER, TARGET
from tests.fixtures import SYNTHETIC_DATA


def test_feature_order_completeness():
    """Все признаки из NUMERIC + CATEGORICAL входят в FEATURE_ORDER."""
    assert set(FEATURE_ORDER) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)


def test_feature_order_no_target():
    """Целевая переменная не должна входить в FEATURE_ORDER."""
    assert TARGET not in FEATURE_ORDER


def test_synthetic_data_has_all_columns():
    """Синтетический датасет содержит все нужные столбцы."""
    required = set(FEATURE_ORDER + [TARGET])
    assert required.issubset(set(SYNTHETIC_DATA.columns))


def test_synthetic_data_no_nulls():
    """В синтетическом датасете нет пропусков."""
    assert SYNTHETIC_DATA.isnull().sum().sum() == 0


def test_synthetic_data_churn_classes():
    """В синтетическом датасете присутствуют оба класса churn."""
    classes = set(SYNTHETIC_DATA[TARGET].unique())
    assert classes == {0, 1}


def test_x_y_split():
    """Разделение на X и y работает корректно."""
    X = SYNTHETIC_DATA[FEATURE_ORDER]
    y = SYNTHETIC_DATA[TARGET]

    assert list(X.columns) == FEATURE_ORDER
    assert len(X) == len(y)
    assert TARGET not in X.columns