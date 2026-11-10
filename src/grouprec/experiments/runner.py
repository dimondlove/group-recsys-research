"""
Запуск эксперимента: данные -> разбиение -> индивидуальная модель -> группы ->
правила агрегации -> метрики по отложенным оценкам -> результаты на диск.

Протокол для каждой группы:
  1. Модель предсказывает оценки всех элементов участникам.
  2. Кандидаты: элементы, которых никто из группы не видел в обучении
     (режим 'unseen_test_union' дополнительно требует отложенной оценки хотя бы у одного участника).
  3. Правило агрегирует предсказания участников по кандидатам, берётся список из k элементов.
  4. Релевантным для участника считается кандидат с отложенной оценкой >= порога;
     считаются NDCG@k, hit и показатели справедливости по участникам группы.
"""
from __future__ import annotations

import json
import platform
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import yaml

from ..analysis.summary import METRICS, summarize
from ..data import top_n_union
from ..datasets import load_movielens, split_ratings
from ..groups import form_groups, similarity_matrix
from ..metrics import group_cohesion, topk_overlap
from ..models import build_model
from ..rules import get_rule, rank_items
from .config import load_config
from .evaluation import evaluate_list


def _progress(iterable, enabled: bool, desc: str):
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, leave=False)
    except ImportError:
        return iterable


def run_experiment(cfg: dict, base_dir=".", progress: bool = True) -> Path:
    t0 = time.time()
    base = Path(base_dir)
    data_cfg, g_cfg, a_cfg = cfg["data"], cfg["groups"], cfg["aggregation"]

    n_streams = 3 + len(g_cfg["sizes"]) * len(g_cfg["kinds"])
    streams = np.random.SeedSequence(cfg["seed"]).spawn(n_streams)
    int_seed = lambda ss: int(ss.generate_state(1)[0])  # noqa: E731

    ratings = load_movielens(base / data_cfg["path"], data_cfg["min_user_ratings"], data_cfg["min_item_ratings"])
    train, test = split_ratings(ratings, cfg["split"]["test_frac"], cfg["split"]["method"],
                                int_seed(streams[0]), cfg["split"]["min_train"])
    model = build_model(**{**cfg["model"], "seed": int_seed(streams[1])}).fit(train)
    rmse_test = model.rmse(test)

    train_csr, test_csr = train.to_csr(), test.to_csr()
    thr = g_cfg["relevance_threshold"]
    rel_counts = np.bincount(test.user[test.rating >= thr], minlength=train.n_users)
    eligible = np.flatnonzero(rel_counts >= g_cfg["min_test_relevant"])
    if g_cfg["activity_band"]:   # ограничиваем активность участников, чтобы не смешивать её со схожестью
        activity = np.bincount(train.user, minlength=train.n_users)[eligible]
        lo, hi = np.quantile(activity, g_cfg["activity_band"])
        eligible = eligible[(activity >= lo) & (activity <= hi)]
    if len(eligible) < max(g_cfg["sizes"]):
        raise ValueError(f"Подходящих пользователей слишком мало ({len(eligible)}) для групп размера "
                         f"{max(g_cfg['sizes'])}. Уменьшите groups.min_test_relevant или размеры групп.")
    pool_rng = np.random.default_rng(streams[2])
    pool = np.sort(pool_rng.choice(eligible, min(len(eligible), g_cfg["pool_size"]), replace=False))
    S = similarity_matrix(train, pool)

    rules = [get_rule(n) for n in a_cfg["rules"]]
    tiebreak_rule = get_rule(a_cfg["tiebreak"]) if a_cfg["tiebreak"] else None
    k = a_cfg["k"]
    rows, skipped = [], 0
    stream_i = 3
    for size in g_cfg["sizes"]:
        for kind in g_cfg["kinds"]:
            rng = np.random.default_rng(streams[stream_i])
            stream_i += 1
            groups = form_groups(pool, S, size, g_cfg["n_groups"], kind, rng, g_cfg["neighborhood"])
            for gi, grp in enumerate(_progress(groups, progress, f"{kind} x{size}")):
                members = grp.members
                P = model.predict(members)
                cand_mask = train_csr[members].getnnz(axis=0) == 0
                if a_cfg["candidates"] == "unseen_test_union":
                    cand_mask &= test_csr[members].getnnz(axis=0) > 0
                cand = np.flatnonzero(cand_mask)
                if len(cand) < k:
                    skipped += 1
                    continue
                R = P[:, cand].astype(np.float64)
                rel = test_csr[members][:, cand].toarray() >= thr
                if a_cfg["prefilter_top_n"]:
                    sub = top_n_union(R, a_cfg["prefilter_top_n"])
                    sub = sub if len(sub) >= k else np.arange(len(cand))
                else:
                    sub = np.arange(len(cand))
                Rs = R[:, sub]
                tb = tiebreak_rule.scores(Rs) if tiebreak_rule else None
                coh_pred = group_cohesion(R)
                tops = {rule.name: sub[rank_items(rule.scores(Rs), tb, k)] for rule in rules}
                ref_top = tops[a_cfg["reference_rule"]]
                activity = float(train_csr[members].getnnz(axis=1).mean())
                for rule in rules:
                    top = tops[rule.name]
                    rows.append({"group_id": f"{kind}-{size}-{gi}", "kind": kind, "size": size,
                                 "rule": rule.name, "cohesion_sim": grp.similarity,
                                 "cohesion_pred": coh_pred, "n_candidates": len(cand),
                                 "mean_train_ratings": activity,
                                 "mean_relevant_cands": float(rel.sum(axis=1).mean()),
                                 "overlap_ref": topk_overlap(top, ref_top),
                                 **evaluate_list(rel, top)})

    df = pd.DataFrame(rows)
    summary = summarize(df) if len(df) else pd.DataFrame()

    run_id = cfg["output"]["run_id"] or f"{cfg['name']}_s{cfg['seed']}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = base / cfg["output"]["dir"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(run_dir / "results.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    meta = {
        "run_id": run_id, "seed": cfg["seed"], "seconds": round(time.time() - t0, 1),
        "n_users": train.n_users, "n_items": train.n_items,
        "n_train": train.n_ratings, "n_test": test.n_ratings,
        "n_eligible_users": int(len(eligible)), "n_pool": int(len(pool)),
        "model_train_rmse": getattr(model, "train_rmse_", None), "model_test_rmse": rmse_test,
        "groups_skipped_few_candidates": skipped,
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    log = base / cfg["output"]["log"]
    log.parent.mkdir(parents=True, exist_ok=True)
    new = not log.exists()
    with log.open("a", encoding="utf-8") as f:
        if new:
            f.write("run_id,config_name,seed,test_rmse,seconds,run_dir\n")
        f.write(f"{run_id},{cfg['name']},{cfg['seed']},{rmse_test:.4f},{meta['seconds']},{run_dir.as_posix()}\n")
    return run_dir


def print_summary(run_dir: Path) -> None:
    summary = pd.read_csv(Path(run_dir) / "summary.csv")
    meta = json.loads((Path(run_dir) / "meta.json").read_text(encoding="utf-8"))
    print(f"\nЗапуск {meta['run_id']}: пользователей {meta['n_users']}, элементов {meta['n_items']}, "
          f"RMSE модели на тесте {meta['model_test_rmse']:.3f}, время {meta['seconds']} с")
    if summary.empty:
        print("Нет результатов (слишком мало кандидатов?).")
        return
    pd.set_option("display.width", 140)
    for (kind, size), part in summary.groupby(["kind", "size"], sort=False):
        print(f"\n[{kind}, размер группы {size}, групп: {int(part['n_groups'].iloc[0])}]")
        print(part.set_index("rule")[["ndcg_mean", "ndcg_min", "ndcg_gini", "share_zero"]].round(3).to_string())


def main(config_path, base_dir=".", progress: bool = True, seed: int | None = None) -> Path:
    cfg = load_config(config_path)
    if seed is not None:
        cfg["seed"] = seed
    run_dir = run_experiment(cfg, base_dir, progress)
    print_summary(run_dir)
    print(f"\nРезультаты сохранены в {run_dir}")
    return run_dir