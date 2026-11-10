"""Оценка рекомендательного списка по отложенным оценкам участников группы."""
from __future__ import annotations

import numpy as np

from ..metrics import gini


def evaluate_list(rel: np.ndarray, rec_cols: np.ndarray) -> dict[str, float]:
    """
    rel       булева матрица (участники x кандидаты): релевантен ли кандидат участнику
              (отложенная оценка >= порога)
    rec_cols  индексы столбцов rel, попавших в список (в порядке ранжирования)

    Считает NDCG@k и попадание (hit) для каждого участника, у которого среди
    кандидатов есть хотя бы один релевантный элемент (иначе участник не оценивается),
    затем сводные показатели по группе.
    """
    k = len(rec_cols)
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    n_rel = rel.sum(axis=1)
    valid = n_rel > 0
    if not valid.any():
        return {"n_valid": 0, "ndcg_mean": np.nan, "ndcg_min": np.nan, "ndcg_gini": np.nan,
                "share_zero": np.nan, "hit_mean": np.nan}

    top = rel[:, rec_cols]
    dcg = (top * discount).sum(axis=1)
    ideal_prefix = np.cumsum(discount)
    idcg = ideal_prefix[np.minimum(n_rel, k).clip(min=1) - 1]
    ndcg = (dcg / idcg)[valid]
    hit = top.any(axis=1)[valid]
    return {
        "n_valid": int(valid.sum()),
        "ndcg_mean": float(ndcg.mean()),
        "ndcg_min": float(ndcg.min()),
        "ndcg_gini": gini(ndcg),
        "share_zero": float((ndcg == 0).mean()),
        "hit_mean": float(hit.mean()),
    }