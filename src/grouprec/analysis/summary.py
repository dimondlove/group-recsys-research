"""Сводные таблицы результатов эксперимента."""
from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = ["ndcg_mean", "ndcg_min", "ndcg_gini", "share_zero", "hit_mean"]


def summarize(df: pd.DataFrame, by=("kind", "size", "rule"), metrics=METRICS) -> pd.DataFrame:
    """Среднее по группам и половина ширины 95% доверительного интервала (нормальное приближение)."""
    g = df.groupby(list(by))
    out = g[metrics].mean()
    n = g.size()
    for m in metrics:
        out[m + "_ci95"] = 1.96 * g[m].std(ddof=1) / np.sqrt(n)
    out["n_groups"] = n
    return out.reset_index()