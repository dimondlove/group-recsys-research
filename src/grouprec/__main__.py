"""Командная строка: python -m grouprec ФАЙЛ [параметры]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .compare import compare_rules
from .data import TASK_LIMITS, InputError, load_ratings_csv, parse_rankings
from .rules import available_rules, get_rule


def format_report(res: dict, members: list[str], items: list[str]) -> str:
    out = [f"Участников: {res['n_members']}, элементов: {res['n_items']}"
           + (f" (кандидатов после отбора: {res['n_candidates']})" if res["n_candidates"] != res["n_items"] else ""),
           f"Согласованность группы (средний Спирмен): {res['cohesion']:.3f}",
           f"Длина списка: {res['k']}"
           + (f", ничьи разрешаются по правилу '{res['tiebreak']}'" if res["tiebreak"] else ""),
           ""]
    w = max(len(items[i]) for r in res["rules"].values() for i in r["top"]) + 2
    for name, r in res["rules"].items():
        out.append(f"{get_rule(name).title}")
        for place, (i, s) in enumerate(zip(r["top"], r["top_scores"]), start=1):
            out.append(f"  {place}. {items[i].ljust(w)} очки: {s:g}")
        if len(r["winners"]) > 1:
            out.append("  Ничья за первое место: " + ", ".join(items[i] for i in r["winners"]))
        f = r["fairness"]
        out.append(f"  Удовлетворённость: среднее {f['mean']:.2f}, минимум {f['min']:.2f}, "
                   f"Джини {f['gini']:.3f}")
        out.append("")
    if res["overlap"]:
        out.append("Совпадение списков (индекс Жаккара):")
        for pair, v in res["overlap"].items():
            a, b = pair.split("|")
            out.append(f"  {a} / {b}: {v:.2f}")
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="grouprec",
                                description="Сравнение правил агрегации предпочтений группы.")
    p.add_argument("file", help="файл с данными: .csv (матрица оценок) или текст (ранжирования)")
    p.add_argument("--format", choices=["auto", "ratings", "rankings"], default="auto",
                   help="формат файла (auto: .csv = ratings, иначе rankings)")
    p.add_argument("--rules", nargs="+", metavar="ПРАВИЛО", default=None,
                   help=f"какие правила применять (доступны: {', '.join(available_rules())}); по умолчанию все")
    p.add_argument("--top", type=int, default=5, help="длина списка (по умолчанию 5)")
    p.add_argument("--tiebreak", default=None, metavar="ПРАВИЛО", help="правило для разрешения ничьих")
    p.add_argument("--prefilter", type=int, default=None, metavar="N",
                   help="отбор кандидатов: объединение top-N каждого участника")
    p.add_argument("--limits", choices=["none", "task"], default="none",
                   help="task: ограничения учебного задания (3-5 элементов, 3-20 экспертов)")
    p.add_argument("--json", metavar="FILE", dest="json_out", help="сохранить результат в JSON")
    args = p.parse_args(argv)

    limits = TASK_LIMITS if args.limits == "task" else None
    fmt = args.format
    if fmt == "auto":
        fmt = "ratings" if Path(args.file).suffix.lower() == ".csv" else "rankings"
    try:
        if fmt == "ratings":
            members, items, ratings = load_ratings_csv(args.file, limits)
        else:
            members, items, ratings = parse_rankings(Path(args.file).read_text(encoding="utf-8"), limits)
        res = compare_rules(ratings, args.rules, k=args.top, tiebreak=args.tiebreak,
                            prefilter_top_n=args.prefilter, limits=limits)
    except (InputError, KeyError, OSError) as exc:
        print(f"Ошибка: {exc.args[0] if isinstance(exc, KeyError) else exc}", file=sys.stderr)
        return 1

    print(format_report(res, members, items))
    if args.json_out:
        res_named = dict(res, items=items, members=members)
        Path(args.json_out).write_text(json.dumps(res_named, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nРезультат сохранён в {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())