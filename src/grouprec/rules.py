"""
Правила агрегации предпочтений с единым интерфейсом.

Каждое правило получает матрицу оценок ratings формы (участники, элементы)
и возвращает вектор очков по элементам: чем больше очков, тем лучше.

Чтобы добавить своё правило, достаточно описать класс и пометить его @register:

    @register
    class MyRule(AggregationRule):
        name = "my_rule"
        title = "Моё правило"
        def scores(self, ratings):
            return ratings.mean(axis=0)

Совпадающие оценки (слабые порядки) поддерживаются всеми правилами.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

_REGISTRY: dict[str, "AggregationRule"] = {}


class AggregationRule(ABC):
    name: str = ""    # короткий идентификатор (для командной строки и конфигов)
    title: str = ""   # название для отчётов

    @abstractmethod
    def scores(self, ratings: np.ndarray) -> np.ndarray:
        """Очки элементов (форма (n_items,)); больше = лучше."""


def register(cls):
    """Декоратор: добавляет правило в реестр."""
    if not cls.name:
        raise ValueError("У правила должно быть задано имя (name).")
    _REGISTRY[cls.name] = cls()
    return cls


def unregister(name: str) -> None:
    """Удаляет правило из реестра (полезно в тестах и экспериментах)."""
    _REGISTRY.pop(name, None)


def get_rule(name: str) -> AggregationRule:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Неизвестное правило '{name}'. Доступны: {', '.join(available_rules())}") from None


def available_rules() -> list[str]:
    return list(_REGISTRY)


# --------------------------------------------------------------------------
# Вспомогательные функции
# --------------------------------------------------------------------------
def pairwise_preferences(ratings: np.ndarray) -> np.ndarray:
    """
    P[a, b] = число участников, строго предпочитающих элемент a элементу b.
    Равные оценки не дают голоса ни одной из сторон (так учитываются слабые порядки).
    """
    n_items = ratings.shape[1]
    pref = np.zeros((n_items, n_items), dtype=np.int32)
    for r in ratings:
        pref += r[:, None] > r[None, :]
    return pref


def rank_items(scores, tiebreak=None, k: int | None = None) -> np.ndarray:
    """
    Индексы элементов по убыванию очков.
    Равные очки разрешаются сначала по tiebreak (если задан), затем по номеру элемента
    (детерминированно). Реальные ничьи определяйте через top_ties().
    """
    s = np.round(np.asarray(scores, dtype=float), 10)
    keys = [np.arange(len(s))]
    if tiebreak is not None:
        keys.append(-np.round(np.asarray(tiebreak, dtype=float), 10))
    keys.append(-s)
    order = np.lexsort(tuple(keys))  # последний ключ - главный
    return order if k is None else order[:k]


def top_ties(scores, tiebreak=None) -> np.ndarray:
    """Индексы всех элементов, нераздельно делящих первое место."""
    s = np.round(np.asarray(scores, dtype=float), 10)
    mask = s == s.max()
    if tiebreak is not None:
        t = np.round(np.asarray(tiebreak, dtype=float), 10)
        mask &= t == t[mask].max()
    return np.flatnonzero(mask)


# --------------------------------------------------------------------------
# Правила
# --------------------------------------------------------------------------
@register
class Average(AggregationRule):
    name = "average"
    title = "Среднее (Average)"

    def scores(self, ratings):
        return ratings.mean(axis=0)


@register
class LeastMisery(AggregationRule):
    name = "least_misery"
    title = "Наименьшее недовольство (Least Misery)"

    def scores(self, ratings):
        return ratings.min(axis=0)


@register
class MostPleasure(AggregationRule):
    name = "most_pleasure"
    title = "Наибольшее удовольствие (Most Pleasure)"

    def scores(self, ratings):
        return ratings.max(axis=0)


@register
class Borda(AggregationRule):
    """
    Борда: у каждого участника элемент получает столько очков, сколько элементов он
    строго превосходит, плюс 0,5 за каждый элемент с равной оценкой. Очки суммируются.
    """
    name = "borda"
    title = "Борда (Borda)"

    def scores(self, ratings):
        total = np.zeros(ratings.shape[1])
        for r in ratings:
            s = np.sort(r)
            less = np.searchsorted(s, r, side="left")
            leq = np.searchsorted(s, r, side="right")
            total += less + 0.5 * (leq - less - 1)
        return total


@register
class Copeland(AggregationRule):
    """
    Копленд: очки = число побед минус число поражений в попарных сравнениях
    (побеждает тот, кого строго предпочитает больше участников; при равенстве ничья).
    Память: матрица n_items x n_items, для тысяч элементов лучше предварительно
    сократить кандидатов (см. data.top_n_union).
    """
    name = "copeland"
    title = "Копленд (Copeland)"

    def scores(self, ratings):
        pref = pairwise_preferences(ratings)
        wins = (pref > pref.T).sum(axis=1)
        losses = (pref < pref.T).sum(axis=1)
        return (wins - losses).astype(float)