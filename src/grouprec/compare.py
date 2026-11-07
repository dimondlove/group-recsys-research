"""Сравнение правил агрегации на одной группе."""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .data import top_n_union, validate_ratings
from .metrics import fairness_summary, group_cohesion, member_satisfaction, topk_overlap
from .rules import available_rules, get_rule, rank_items, top_ties


def compare_rules(ratings, rule_names=None, k: int = 5, tiebreak: str | None = None,
                  prefilter_top_n: int | None = None, limits=None) -> dict:
    """
    Применяет правила к матрице оценок и считает метрики по первоначальным оценкам.

    rule_names       список имён правил (по умолчанию все зарегистрированные)
    k                длина рекомендательного списка
    tiebreak         имя правила для разрешения ничьих (например, 'borda')
    prefilter_top_n  если задан, правила применяются только к объединению top-n каждого
                     участника (сокращение кандидатов); метрики считаются по исходным оценкам
    Возвращает словарь с индексами элементов (названия подставляет вызывающий код).
    """
    r = validate_ratings(ratings, limits)
    names = list(rule_names) if rule_names else available_rules()
    cand = top_n_union(r, prefilter_top_n) if prefilter_top_n else np.arange(r.shape[1])
    sub = r[:, cand]
    k = min(k, len(cand))
    tb = get_rule(tiebreak).scores(sub) if tiebreak else None

    out: dict = {"n_members": r.shape[0], "n_items": r.shape[1], "n_candidates": len(cand),
                 "k": k, "tiebreak": tiebreak, "cohesion": group_cohesion(r), "rules": {}}
    tops: dict[str, np.ndarray] = {}
    for name in names:
        sc = get_rule(name).scores(sub)
        top = cand[rank_items(sc, tb, k)]
        tops[name] = top
        sat = member_satisfaction(r, top)
        out["rules"][name] = {
            "top": top.tolist(),
            "top_scores": sc[rank_items(sc, tb, k)].tolist(),
            "winners": cand[top_ties(sc, tb)].tolist(),
            "satisfaction": sat.tolist(),
            "fairness": fairness_summary(sat),
        }
    out["overlap"] = {f"{a}|{b}": topk_overlap(tops[a], tops[b]) for a, b in combinations(names, 2)}
    return out