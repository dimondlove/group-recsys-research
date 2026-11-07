"""Метрики: удовлетворённость участников, справедливость, согласованность группы."""
from __future__ import annotations

import numpy as np


def member_satisfaction(ratings: np.ndarray, selected) -> np.ndarray:
    """Средняя оценка выбранных элементов каждым участником (форма (n_members,))."""
    return ratings[:, np.asarray(selected, dtype=int)].mean(axis=1)


def gini(values) -> float:
    """Коэффициент Джини (0 = полное равенство). Для неотрицательных значений."""
    v = np.sort(np.asarray(values, dtype=float))
    if v.size == 0 or v.sum() == 0:
        return 0.0
    n = v.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))


def fairness_summary(sat: np.ndarray) -> dict[str, float]:
    """Сводка по удовлетворённости участников: среднее, минимум, std, Джини."""
    return {
        "mean": float(sat.mean()),
        "min": float(sat.min()),
        "std": float(sat.std()),
        "gini": gini(sat - min(0.0, sat.min())),  # сдвиг, если есть отрицательные значения
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Ранги (1 = наименьшее значение) с усреднением при совпадениях."""
    s = np.sort(values)
    less = np.searchsorted(s, values, side="left")
    leq = np.searchsorted(s, values, side="right")
    return less + (leq - less + 1) / 2


def group_cohesion(ratings: np.ndarray) -> float:
    """
    Согласованность группы: средняя попарная корреляция Спирмена между участниками
    (1 = одинаковые вкусы, около 0 = несвязанные). Участник с постоянными оценками
    даёт корреляцию 0.
    """
    ranks = np.array([average_ranks(r) for r in ratings])
    centered = ranks - ranks.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    corr = np.zeros((len(ranks), len(ranks)))
    ok = norms > 0
    if ok.sum() >= 2:
        z = centered[ok] / norms[ok, None]
        corr[np.ix_(ok, ok)] = z @ z.T
    iu = np.triu_indices(len(ranks), k=1)
    return float(corr[iu].mean())


def topk_overlap(a, b) -> float:
    """Доля общих элементов в двух списках (индекс Жаккара)."""
    sa, sb = set(map(int, a)), set(map(int, b))
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0