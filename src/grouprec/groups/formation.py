"""Формирование групп пользователей заданного размера и степени сходства."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import scipy.sparse as sp

from ..datasets.movielens import Ratings

KINDS = ("random", "similar", "dissimilar")


@dataclass
class Group:
    members: np.ndarray   # глобальные индексы пользователей
    kind: str
    similarity: float     # средняя попарная схожесть по обучающим данным


def similarity_matrix(train: Ratings, users: np.ndarray) -> np.ndarray:
    """
    Косинус между центрированными по среднему пользователя векторами оценок
    (отсутствующие оценки считаются нулём). Форма (len(users), len(users)), по обучающим данным.
    Замечание: у малоактивных пользователей косинус «шумнее», они чаще попадают в крайние
    значения, поэтому пул пользователей стоит ограничивать диапазоном активности
    (groups.activity_band) и проверять диагностику mean_train_ratings.
    """
    m = train.to_csr()[users].astype(np.float64)
    counts = np.diff(m.indptr)
    sums = np.asarray(m.sum(axis=1)).ravel()
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    centered = m.copy()
    centered.data = centered.data - np.repeat(means, counts)
    norms = np.sqrt(np.asarray(centered.multiply(centered).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    centered = sp.diags(1.0 / norms) @ centered
    s = (centered @ centered.T).toarray()
    np.fill_diagonal(s, 1.0)
    return s


def mean_pairwise(S: np.ndarray, positions) -> float:
    pos = list(positions)
    return float(np.mean([S[a, b] for a, b in combinations(pos, 2)]))


def form_groups(users: np.ndarray, S: np.ndarray, size: int, n_groups: int, kind: str,
                rng: np.random.Generator, neighborhood: int = 50) -> list[Group]:
    """
    users          глобальные индексы кандидатов (строки/столбцы S в том же порядке)
    kind='random'      случайные участники;
    kind='similar'     случайный «затравочный» пользователь, затем жадно добавляются
                       участники из его `neighborhood` самых похожих соседей: каждый
                       следующий максимизирует среднюю схожесть с уже набранными;
    kind='dissimilar'  то же, но из `neighborhood` самых непохожих и с минимизацией схожести.
    Реально полученная схожесть всегда записывается в Group.similarity. Различия между
    группами обеспечивает случайный выбор затравки (при n_groups больше числа кандидатов
    возможны повторы групп).
    """
    if kind not in KINDS:
        raise ValueError(f"kind должен быть одним из {KINDS}.")
    n = len(users)
    if size < 2 or size > n:
        raise ValueError(f"Размер группы должен быть от 2 до {n} (число кандидатов).")
    if kind != "random" and neighborhood < size - 1:
        raise ValueError("neighborhood должен быть не меньше size - 1.")

    groups = []
    for _ in range(n_groups):
        if kind == "random":
            pos = list(rng.choice(n, size=size, replace=False))
        else:
            seed = int(rng.integers(n))
            order = np.argsort(-S[seed] if kind == "similar" else S[seed], kind="stable")
            pool = list(order[order != seed][:neighborhood])
            pos = [seed]
            while len(pos) < size:
                mean_sim = S[np.ix_(pool, pos)].mean(axis=1)
                best = int(np.argmax(mean_sim) if kind == "similar" else np.argmin(mean_sim))
                pos.append(pool.pop(best))
        groups.append(Group(users[np.array(pos)], kind, mean_pairwise(S, pos)))
    return groups