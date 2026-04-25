# ML Churn Service

FastAPI-сервис для предсказания оттока клиентов (churn). Принимает набор признаков клиента и возвращает вероятность того, что клиент уйдёт.

## Структура проекта

```
├── main.py                  # FastAPI приложение, /health эндпоинт
├── routers/
│   ├── data.py              # GET /dataset/...
│   ├── model.py             # POST /model/train, GET /model/status, /schema, /metrics
│   └── predict.py           # POST /predict
├── schemas/
│   └── churn.py             # Pydantic модели запросов
├── services/
│   ├── feature_schema.py    # Единый источник правды о признаках
│   ├── trainer.py           # Обучение sklearn Pipeline
│   ├── model_score.py       # Сохранение/загрузка модели (joblib)
│   ├── history_store.py     # История обучений (JSON)
│   └── preprocessing.py     # Анализ разбиения датасета
├── core/
│   ├── errors.py            # Кастомные исключения и обработчики ошибок
│   └── logger.py            # Настройка логгера
├── tests/                   # pytest тесты
├── data/
│   └── churn_dataset.csv    # Датасет (2000 строк)
├── models/                  # Сохранённые модели (joblib, создаётся автоматически)
├── Dockerfile
└── requirements.txt
```

## Формат датасета

Файл `data/churn_dataset.csv`, 2000 строк. Обязательные столбцы:

| Столбец | Тип | Описание |
|---|---|---|
| `monthly_fee` | float | Ежемесячная стоимость тарифа |
| `usage_hours` | float | Часов использования сервиса за месяц |
| `support_requests` | int | Количество обращений в поддержку |
| `account_age_months` | int | Возраст аккаунта в месяцах |
| `failed_payments` | int | Количество неудачных платежей |
| `autopay_enabled` | int | Автосписание включено (0 или 1) |
| `region` | str | Регион: `europe`, `asia`, `america`, `africa` |
| `device_type` | str | Тип устройства: `mobile`, `desktop`, `tablet` |
| `payment_method` | str | Способ оплаты: `card`, `paypal`, `crypto` |
| `churn` | int | Целевая переменная: 0 — остался, 1 — ушёл |

## Запуск локально

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn main:app --reload
```

Сервис доступен на `http://localhost:8000`. Документация: `http://localhost:8000/docs`.

## Запуск в Docker

```bash
docker build -t ml-churn-service .

docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  ml-churn-service
```

Тома `-v` монтируют датасет и директорию моделей с хоста, чтобы обученная модель сохранялась между перезапусками контейнера.

## Примеры запросов

### Обучить модель

```bash
curl -X POST "http://localhost:8000/model/train" \
  -H "Content-Type: application/json" \
  -d '{"model_type": "logreg", "hyperparameters": {}}'
```

С параметрами через query:
```bash
curl -X POST "http://localhost:8000/model/train?test_size=0.2&random_state=42" \
  -H "Content-Type: application/json" \
  -d '{"model_type": "random_forest", "hyperparameters": {"n_estimators": 200}}'
```

Ответ:
```json
{
  "status": "model trained successfully",
  "model_type": "logreg",
  "hyperparameters": {"max_iter": 1000, "random_state": 42},
  "metrics": {
    "accuracy": 0.8125,
    "f1_score": 0.2857,
    "roc_auc": 0.7431,
    "train_size": 1600,
    "test_size": 400
  }
}
```

### Предсказать отток

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "monthly_fee": 95.0,
    "usage_hours": 30.0,
    "support_requests": 5,
    "account_age_months": 6,
    "failed_payments": 3,
    "region": "europe",
    "device_type": "mobile",
    "payment_method": "card",
    "autopay_enabled": 0
  }'
```

Ответ:
```json
{
  "churn": 1,
  "probability": 0.7341
}
```

### Проверить состояние сервиса

```bash
curl http://localhost:8000/health
```

Ответ:
```json
{
  "status": "ok",
  "model": {
    "is_trained": true,
    "trained_at": "2026-04-25T10:00:00.000000",
    "model_type": "logreg"
  },
  "dataset": {
    "available": true,
    "path": "/app/data/churn_dataset.csv"
  }
}
```

## Тесты

```bash
pytest tests/ -v
```

## Эндпоинты

| Метод | Путь | Описание |
|---|---|---|
| GET | `/` | Статус сервиса |
| GET | `/health` | Состояние модели и датасета |
| POST | `/predict` | Предсказание churn |
| POST | `/model/train` | Обучение модели |
| GET | `/model/status` | Статус обученной модели |
| GET | `/model/metrics` | История обучений |
| GET | `/model/schema` | Схема признаков |
| GET | `/dataset/preview` | Первые N строк датасета |
| GET | `/dataset/info` | Статистика датасета |
| GET | `/dataset/split-info` | Информация о разбиении train/test |
