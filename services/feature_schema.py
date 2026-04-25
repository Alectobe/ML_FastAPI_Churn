# описание всех признаков churn модели
# этот файл — единственный источник правды о схеме признаков
# trainer.py, predictor.py и эндпоинт /model/schema импортируют отсюда

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

# порядок признаков для подачи в pipeline при предсказании
# должен строго совпадать с порядком при обучении
FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# описание типов и допустимых значений для каждого признака
# используется в эндпоинте /model/schema
FEATURE_SCHEMA = {
    "monthly_fee": {
        "type": "float",
        "description": "Ежемесячная стоимость тарифа клиента",
        "example": 29.99,
    },
    "usage_hours": {
        "type": "float",
        "description": "Количество часов использования сервиса за последний месяц",
        "example": 45.5,
    },
    "support_requests": {
        "type": "int",
        "description": "Количество обращений в техподдержку",
        "example": 2,
    },
    "account_age_months": {
        "type": "int",
        "description": "Возраст аккаунта в месяцах",
        "example": 12,
    },
    "failed_payments": {
        "type": "int",
        "description": "Количество неудачных платежей",
        "example": 0,
    },
    "autopay_enabled": {
        "type": "int",
        "description": "Включено ли автосписание",
        "allowed_values": [0, 1],
        "example": 1,
    },
    "region": {
        "type": "str",
        "description": "Регион клиента",
        "allowed_values": ["europe", "asia", "america", "africa"],
        "example": "europe",
    },
    "device_type": {
        "type": "str",
        "description": "Основной тип устройства",
        "allowed_values": ["mobile", "desktop", "tablet"],
        "example": "mobile",
    },
    "payment_method": {
        "type": "str",
        "description": "Способ оплаты",
        "allowed_values": ["card", "paypal", "crypto"],
        "example": "card",
    },
}

# пример валидного запроса — собирается автоматически из схемы
EXAMPLE_REQUEST = {field: meta["example"] for field, meta in FEATURE_SCHEMA.items()}

MODEL_TYPES = ["logreg", "random_forest"]