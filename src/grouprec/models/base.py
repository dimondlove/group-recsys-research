"""Интерфейс индивидуального рекомендателя."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..datasets.movielens import Ratings

_REGISTRY: dict[str, type["Recommender"]] = {}


class Recommender(ABC):
    name: str = ""

    @abstractmethod
    def fit(self, train: Ratings) -> "Recommender":
        """Обучение на обучающей части рейтингов. Возвращает self."""

    @abstractmethod
    def predict(self, users: np.ndarray) -> np.ndarray:
        """Предсказанные оценки всех элементов для users: форма (len(users), n_items)."""

    @abstractmethod
    def predict_pairs(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Предсказания для пар (пользователь, элемент); нужны для RMSE на тесте."""

    def rmse(self, data: Ratings) -> float:
        pred = self.predict_pairs(data.user, data.item)
        return float(np.sqrt(np.mean((pred - data.rating) ** 2)))


def register_model(cls):
    if not cls.name:
        raise ValueError("У модели должно быть имя (name).")
    _REGISTRY[cls.name] = cls
    return cls


def build_model(name: str, **params) -> Recommender:
    try:
        return _REGISTRY[name](**params)
    except KeyError:
        raise KeyError(f"Неизвестная модель '{name}'. Доступны: {', '.join(_REGISTRY)}") from None