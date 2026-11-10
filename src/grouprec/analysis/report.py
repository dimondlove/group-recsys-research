"""Разбор результатов запуска: диагностика, пересечение списков, парные тесты."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .stats import equivalence_vs_reference, friedman_by_config, paired_vs_reference
from .stratify import by_cohesion

DEFAULT_METRICS = ["ndcg_mean", "ndcg_min", "ndcg_gini", "share_zero"]


def analyze_run(run_dir, reference: str | None = None, metrics=DEFAULT_METRICS,
                margin: float = 0.01) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    df = pd.read_csv(run_dir / "results.csv")
    cfg_ref = None
    try:
        import yaml
        cfg_ref = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))["aggregation"].get("reference_rule")
    except Exception:  # noqa: BLE001
        pass
    reference = reference or cfg_ref or "average"

    first_rule = df["rule"].iloc[0]
    diag = (df[df["rule"] == first_rule]
            .groupby(["kind", "size"], sort=False)[
                ["cohesion_sim", "cohesion_pred", "n_candidates", "mean_train_ratings",
                 "mean_relevant_cands", "n_valid"]].mean().round(3))
    overlap = df.groupby(["kind", "size", "rule"], sort=False)["overlap_ref"].mean().unstack("rule").round(3)
    tests = pd.concat([paired_vs_reference(df, m, reference) for m in metrics], ignore_index=True)
    fried = pd.concat([friedman_by_config(df, m) for m in metrics], ignore_index=True)

    equiv = equivalence_vs_reference(df, "ndcg_mean", reference, margin)
    equiv.to_csv(run_dir / "stats_equivalence.csv", index=False)
    strat, corr = by_cohesion(df, "ndcg_mean", reference)
    strat_g, corr_g = by_cohesion(df, "ndcg_gini", reference)
    strat_all = pd.concat([strat, strat_g], ignore_index=True)
    corr_all = pd.concat([corr, corr_g], ignore_index=True)
    strat_all.to_csv(run_dir / "stats_by_cohesion.csv", index=False)
    corr_all.to_csv(run_dir / "corr_cohesion.csv", index=False)
    diag.to_csv(run_dir / "diagnostics.csv")
    overlap.to_csv(run_dir / "overlap.csv")
    tests.to_csv(run_dir / "stats_paired.csv", index=False)
    fried.to_csv(run_dir / "stats_friedman.csv", index=False)
    return {"reference": reference, "diagnostics": diag, "overlap": overlap, "paired": tests, "friedman": fried,
            "equivalence": equiv, "by_cohesion": strat_all, "corr_cohesion": corr_all}


def print_analysis(res: dict) -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)
    ref = res["reference"]
    print("\n=== Диагностика групп (средние по группам) ===")
    print("cohesion_sim: схожесть по обучению; cohesion_pred: согласованность предсказаний;")
    print("mean_train_ratings: активность участников; mean_relevant_cands: релевантных кандидатов на участника")
    print(res["diagnostics"].to_string())
    print(f"\n=== Пересечение списков с эталоном '{ref}' (индекс Жаккара, 1 = списки совпадают) ===")
    print(res["overlap"].to_string())
    fr = res["friedman"]
    print("\n=== Критерий Фридмана: различаются ли правила в конфигурации (p < 0.05) ===")
    print(fr[fr["metric"] == "ndcg_mean"].round(4).to_string(index=False))
    t = res["paired"]
    print(f"\n=== Парные тесты против '{ref}' (Уилкоксон, поправка Холма) ===")
    sig = t[t["significant"]]
    print(f"Сравнений: {len(t)}, значимых после поправки: {len(sig)}")
    if len(sig):
        print(sig.round(4).to_string(index=False))
    else:
        print("Значимых различий нет. Самые заметные по p с поправкой:")
        print(t.sort_values("p_holm").head(5).round(4).to_string(index=False))
    eq = res["equivalence"]
    if len(eq):
        m = eq["margin"].iloc[0]
        print(f"\n=== Эквивалентность по ndcg_mean (TOST, допуск +-{m:g}, поправка Холма) ===")
        print(f"Сравнений: {len(eq)}, эквивалентность доказана в {int(eq['equivalent'].sum())}")
        bad = eq[~eq["equivalent"]]
        if len(bad):
            print("Эквивалентность НЕ доказана (ДИ выходит за допуск или эффект больше допуска):")
            print(bad.round(4)[["kind", "size", "rule", "n", "mean_diff", "ci90_lo", "ci90_hi", "p_holm"]].to_string(index=False))
        print(f"Диапазон 90% ДИ разности по всем сравнениям: от {eq['ci90_lo'].min():+.4f} до {eq['ci90_hi'].max():+.4f}")
    print("\n=== Зависимость от согласованности группы (для гипотезы H4) ===")
    for metric in ("ndcg_mean", "ndcg_gini"):
        _print_by_cohesion(res["by_cohesion"], res["corr_cohesion"], ref, metric)


def _print_by_cohesion(strat: pd.DataFrame, corr: pd.DataFrame, ref: str, metric: str) -> None:
    sub = strat[strat["metric"] == metric]
    if sub.empty:
        return
    print(f"\n--- {metric}: разность (правило - '{ref}') по квартилям согласованности предсказаний; "
          "* = значимо после поправки Холма ---")
    for size, part in sub.groupby("size", sort=True):
        cells = part.assign(cell=lambda d: d["mean_diff"].map("{:+.4f}".format) + d["significant"].map({True: "*", False: ""}))
        table = cells.pivot(index="bin", columns="rule", values="cell")
        rng = part.drop_duplicates("bin").set_index("bin")[["cohesion_lo", "cohesion_hi"]].round(2)
        table.insert(0, "cohesion", rng["cohesion_lo"].astype(str) + ".." + rng["cohesion_hi"].astype(str))
        ov = part.pivot(index="bin", columns="rule", values="overlap_ref").round(2).add_prefix("overlap_")
        print(f"\nразмер группы {size}")
        print(table.join(ov).to_string())
    c = corr[corr["metric"] == metric]
    print(f"\nКорреляция Спирмена: согласованность группы vs разность {metric} (rho < 0: чем менее согласована группа, тем лучше правило относительно '{ref}')")
    print(c.assign(sig=c["p_holm"] < 0.05).round(4)[["size", "rule", "n", "spearman_rho", "p", "p_holm", "sig"]].to_string(index=False))