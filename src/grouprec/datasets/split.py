"""Разбиение рейтингов на обучающую и тестовую части для каждого пользователя."""
from __future__ import annotations

import numpy as np

from .movielens import Ratings


def split_ratings(r: Ratings, test_frac: float = 0.2, method: str = "random",
                  seed: int = 0, min_train: int = 5) -> tuple[Ratings, Ratings]:
    """
    Отложенная выборка по каждому пользователю.

    method='random'  случайные test_frac оценок каждого пользователя;
    method='time'    самые поздние test_frac оценок каждого пользователя.
    Каждому пользователю остаётся не менее min_train обучающих оценок; если оценок
    мало, пользователь целиком попадает в обучение (в тесте его нет).
    """
    if method not in ("random", "time"):
        raise ValueError("method должен быть 'random' или 'time'.")
    if not 0 < test_frac < 1:
        raise ValueError("test_frac должен лежать в (0, 1).")
    rng = np.random.default_rng(seed)
    n = r.n_ratings
    tie = rng.random(n)
    order = np.lexsort((tie, r.timestamp, r.user)) if method == "time" else np.lexsort((tie, r.user))

    counts = np.bincount(r.user, minlength=r.n_users)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    u_sorted = r.user[order]
    pos = np.arange(n) - starts[u_sorted]
    n_test = np.minimum(np.floor(test_frac * counts).astype(int), np.maximum(counts - min_train, 0))
    is_test = pos >= (counts[u_sorted] - n_test[u_sorted])
    return r.subset(np.sort(order[~is_test])), r.subset(np.sort(order[is_test]))