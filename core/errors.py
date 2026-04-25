# core/errors.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel


# --- схема ответа при ошибке ---

class ErrorResponse(BaseModel):
    code: str
    message: str
    details: dict | None = None


# --- кастомные исключения сервиса ---

class ModelNotTrainedError(Exception):
    """Модель ещё не обучена — предсказание невозможно."""
    pass


class DatasetNotFoundError(Exception):
    """Файл датасета не найден на диске."""
    pass


class DatasetEmptyError(Exception):
    """Датасет загружен но пустой."""
    pass


class InvalidFeaturesError(Exception):
    """Переданы некорректные признаки для предсказания."""
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


class UnknownModelTypeError(Exception):
    """Передан неизвестный тип модели при обучении."""
    def __init__(self, model_type: str, available: list):
        super().__init__(f"Неизвестный тип модели: '{model_type}'")
        self.model_type = model_type
        self.available = available


# --- регистрация глобальных обработчиков ---

def register_error_handlers(app: FastAPI) -> None:
    """Регистрирует все глобальные обработчики ошибок в приложении."""

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Перехватывает стандартные HTTP ошибки и оборачивает в единый формат."""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                code=f"HTTP_{exc.status_code}",
                message=exc.detail,
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """
        Перехватывает ошибки валидации Pydantic (422).
        Возвращает список конкретных полей с проблемами.
        """
        field_errors = {}
        for error in exc.errors():
            # loc — путь к полю, например ("body", "monthly_fee")
            field = " -> ".join(str(part) for part in error["loc"])
            field_errors[field] = error["msg"]

        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                code="VALIDATION_ERROR",
                message="Переданы некорректные данные. Проверьте типы и наличие обязательных полей.",
                details={"field_errors": field_errors},
            ).model_dump(),
        )

    @app.exception_handler(ModelNotTrainedError)
    async def model_not_trained_handler(request: Request, exc: ModelNotTrainedError):
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                code="MODEL_NOT_TRAINED",
                message="Модель ещё не обучена. Сначала вызовите POST /model/train.",
            ).model_dump(),
        )

    @app.exception_handler(DatasetNotFoundError)
    async def dataset_not_found_handler(request: Request, exc: DatasetNotFoundError):
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                code="DATASET_NOT_FOUND",
                message=str(exc),
            ).model_dump(),
        )

    @app.exception_handler(DatasetEmptyError)
    async def dataset_empty_handler(request: Request, exc: DatasetEmptyError):
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                code="DATASET_EMPTY",
                message="Датасет загружен, но не содержит строк — обучение невозможно.",
            ).model_dump(),
        )

    @app.exception_handler(InvalidFeaturesError)
    async def invalid_features_handler(request: Request, exc: InvalidFeaturesError):
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                code="INVALID_FEATURES",
                message=str(exc),
                details=exc.details,
            ).model_dump(),
        )

    @app.exception_handler(UnknownModelTypeError)
    async def unknown_model_type_handler(request: Request, exc: UnknownModelTypeError):
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                code="UNKNOWN_MODEL_TYPE",
                message=str(exc),
                details={
                    "passed": exc.model_type,
                    "available": exc.available,
                },
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        """
        Ловит все необработанные исключения.
        Скрывает технические детали от клиента, логирует в консоль.
        """
        print(f"[error] Необработанное исключение: {type(exc).__name__}: {exc}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                code="INTERNAL_ERROR",
                message="Внутренняя ошибка сервера. Обратитесь к администратору.",
            ).model_dump(),
        )