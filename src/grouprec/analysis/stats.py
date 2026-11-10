"""Парные статистические тесты для сравнения правил агрегации."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def _wide(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Таблица группа x правило для одной метрики (группы - те же для всех правил)."""
    return df.pivot(index="group_id", columns="rule", values=metric)


def paired_vs_reference(df: pd.DataFrame, metric: str, reference: str,
                        by=("kind", "size"), alpha: float = 0.05) -> pd.DataFrame:
    """
    Парный критерий Уилкоксона каждого правила против эталонного внутри каждой конфигурации
    (тип группы x размер) и поправка Холма на все сравнения для данной метрики.
    Возвращает: среднюю разность (правило - эталон), медиану, dz Коэна, p и p с поправкой.
    """
    rows = []
    for key, part in df.groupby(list(by), sort=False):
        w = _wide(part, metric)
        if reference not in w:
            raise KeyError(f"Эталонное правило '{reference}' отсутствует в результатах.")
        for rule in w.columns.drop(reference):
            d = (w[rule] - w[reference]).dropna()
            if len(d) < 2 or np.allclose(d, 0):
                p = 1.0
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p = float(stats.wilcoxon(d).pvalue)
            sd = d.std(ddof=1) if len(d) > 1 else np.nan
            rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))),
                         "metric": metric, "rule": rule, "reference": reference, "n": len(d),
                         "mean_diff": float(d.mean()) if len(d) else np.nan,
                         "median_diff": float(d.median()) if len(d) else np.nan,
                         "dz": float(d.mean() / sd) if sd and sd > 0 else np.nan,
                         "p": p})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = multipletests(out["p"].to_numpy(), alpha=alpha, method="holm")[1]
    out["significant"] = out["p_holm"] < alpha
    return out


def friedman_by_config(df: pd.DataFrame, metric: str, by=("kind", "size")) -> pd.DataFrame:
    """Критерий Фридмана по всем правилам внутри конфигурации; W Кендалла как размер эффекта."""
    rows = []
    for key, part in df.groupby(list(by), sort=False):
        w = _wide(part, metric).dropna()
        k = w.shape[1]
        base = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        if k < 3 or len(w) < 2 or np.allclose(w.to_numpy(), w.to_numpy()[:, [0]]):
            rows.append({**base, "metric": metric, "n": len(w), "chi2": np.nan, "p": np.nan, "kendall_w": np.nan})
            continue
        chi2, p = stats.friedmanchisquare(*[w[c] for c in w.columns])
        rows.append({**base, "metric": metric, "n": len(w), "chi2": float(chi2), "p": float(p),
                     "kendall_w": float(chi2 / (len(w) * (k - 1)))})
    return pd.DataFrame(rows)


def equivalence_vs_reference(df: pd.DataFrame, metric: str, reference: str, margin: float,
                             by=("kind", "size"), alpha: float = 0.05) -> pd.DataFrame:
    """
    Парный тест эквивалентности TOST (две односторонние t-проверки): отличается ли правило
    от эталона не больше чем на +-margin по данной метрике. Допуск margin нужно обосновать ДО анализа
    (например, доля от среднего значения метрики или наименьшая практически значимая разность).
    Поправка Холма применяется ко всем сравнениям. equivalent = True, если эквивалентность доказана.
    Также возвращается 90% доверительный интервал разности (соответствует TOST при alpha = 0.05).
    """
    rows = []
    for key, part in df.groupby(list(by), sort=False):
        w = _wide(part, metric)
        for rule in w.columns.drop(reference):
            d = (w[rule] - w[reference]).dropna()
            n = len(d)
            mean = float(d.mean()) if n else np.nan
            sd = float(d.std(ddof=1)) if n > 1 else np.nan
            if n < 3:
                p, lo, hi = np.nan, np.nan, np.nan
            elif sd == 0:
                p, lo, hi = (0.0 if abs(mean) < margin else 1.0), mean, mean
            else:
                se = sd / np.sqrt(n)
                p = float(max(stats.t.sf((mean + margin) / se, n - 1), stats.t.cdf((mean - margin) / se, n - 1)))
                tcrit = stats.t.ppf(1 - alpha, n - 1)
                lo, hi = mean - tcrit * se, mean + tcrit * se
            rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))),
                         "metric": metric, "rule": rule, "reference": reference, "n": n,
                         "mean_diff": mean, "ci90_lo": lo, "ci90_hi": hi, "margin": margin, "p_tost": p})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    p = out["p_tost"].fillna(1.0).to_numpy()
    out["p_holm"] = multipletests(p, alpha=alpha, method="holm")[1]
    out["equivalent"] = out["p_holm"] < alpha
    return out