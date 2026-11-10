"""Зависит ли разница между правилами от согласованности группы (для H4)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def by_cohesion(df: pd.DataFrame, metric: str, reference: str, cohesion_col: str = "cohesion_pred",
                n_bins: int = 4, alpha: float = 0.05) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Для каждого размера группы делит группы на n_bins равных по численности корзин по возрастанию
    согласованности (Q1 = наименее согласованные) и сравнивает каждое правило с эталоном внутри корзины.

    Возвращает (по корзинам, корреляции):
      по корзинам: средняя разность (правило - эталон), 95% ДИ, p Уилкоксона, p с поправкой Холма
                   (на все сравнения для метрики), среднее пересечение списков с эталоном;
      корреляции:  ранговая корреляция Спирмена между согласованностью и разностью (по всем группам размера).
    """
    labels = [f"Q{i + 1}" for i in range(n_bins)]
    rows, corr_rows = [], []
    for size, part in df.groupby("size", sort=True):
        w = part.pivot(index="group_id", columns="rule", values=metric)
        ov = part.pivot(index="group_id", columns="rule", values="overlap_ref")
        coh = part[part["rule"] == reference].set_index("group_id")[cohesion_col].reindex(w.index)
        bins = pd.qcut(coh.rank(method="first"), n_bins, labels=labels)
        for rule in w.columns.drop(reference):
            diff_all = (w[rule] - w[reference])
            ok = diff_all.notna() & coh.notna()
            rho, p_rho = stats.spearmanr(coh[ok], diff_all[ok]) if ok.sum() > 2 else (np.nan, np.nan)
            corr_rows.append({"size": size, "metric": metric, "rule": rule, "n": int(ok.sum()),
                              "spearman_rho": float(rho), "p": float(p_rho)})
            for b in labels:
                idx = (bins == b).to_numpy()
                d = diff_all[idx].dropna()
                if len(d) < 2 or np.allclose(d, 0):
                    p = 1.0
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        p = float(stats.wilcoxon(d).pvalue)
                sd = d.std(ddof=1) if len(d) > 1 else np.nan
                rows.append({"size": size, "bin": b, "metric": metric, "rule": rule, "n": len(d),
                             "cohesion_lo": float(coh[idx].min()), "cohesion_hi": float(coh[idx].max()),
                             "mean_diff": float(d.mean()), "ci95": float(1.96 * sd / np.sqrt(len(d))),
                             "overlap_ref": float(ov.loc[idx, rule].mean()), "p": p})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = multipletests(out["p"].to_numpy(), alpha=alpha, method="holm")[1]
        out["significant"] = out["p_holm"] < alpha
    corr = pd.DataFrame(corr_rows)
    if not corr.empty:
        corr["p_holm"] = multipletests(corr["p"].fillna(1.0).to_numpy(), alpha=alpha, method="holm")[1]
    return out, corr