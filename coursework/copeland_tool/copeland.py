#!/usr/bin/env python3
"""
Инструмент анализа коллективного выбора: процедура Копленда.

Вход:  профиль предпочтений (ранжирование альтернатив каждым экспертом).
Выход: матрица попарных сравнений, очки Копленда, итоговое ранжирование,
       победитель (или победители), победитель/проигравший Кондорсе.

Ограничения: 3-5 альтернатив, 3-20 экспертов.
Используется только стандартная библиотека Python (3.9+).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

MIN_ALTERNATIVES, MAX_ALTERNATIVES = 3, 5
MIN_EXPERTS, MAX_EXPERTS = 3, 20


class ProfileError(ValueError):
    """Ошибка во входных данных (формат или ограничения)."""


# --------------------------------------------------------------------------
# 1. Чтение и проверка профиля
# --------------------------------------------------------------------------
def parse_profile(text: str) -> list[tuple[int, list[str]]]:
    """
    Разбирает текст профиля. Одна строка = ранжирование одного эксперта
    (от лучшей альтернативы к худшей), разделители: '>' или ','.
    Необязательный префикс 'N:' означает, что такое ранжирование дали N экспертов.
    Пустые строки и строки, начинающиеся с '#', игнорируются.
    """
    rows: list[tuple[int, list[str]]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        count = 1
        m = re.match(r"^(\d+)\s*:\s*(.+)$", line)
        if m:
            count = int(m.group(1))
            line = m.group(2)
            if count < 1:
                raise ProfileError(f"Строка {lineno}: число экспертов должно быть >= 1.")
        ranking = [a.strip() for a in re.split(r"[>,]", line) if a.strip()]
        if not ranking:
            raise ProfileError(f"Строка {lineno}: не найдено ни одной альтернативы.")
        rows.append((count, ranking))
    return rows


def validate_profile(rows: list[tuple[int, list[str]]]) -> tuple[list[str], int]:
    """Проверяет профиль. Возвращает (список альтернатив, число экспертов)."""
    if not rows:
        raise ProfileError("Профиль пуст: нет ни одного ранжирования.")

    alternatives = list(rows[0][1])  # порядок из первой строки используем для вывода
    alt_set = set(alternatives)

    if len(alternatives) != len(alt_set):
        raise ProfileError("В первом ранжировании есть повторяющиеся альтернативы.")
    if not MIN_ALTERNATIVES <= len(alternatives) <= MAX_ALTERNATIVES:
        raise ProfileError(
            f"Число альтернатив должно быть от {MIN_ALTERNATIVES} до {MAX_ALTERNATIVES}, "
            f"получено {len(alternatives)}."
        )

    for i, (_, ranking) in enumerate(rows, start=1):
        if len(ranking) != len(set(ranking)):
            raise ProfileError(f"Ранжирование №{i}: альтернативы повторяются.")
        if set(ranking) != alt_set:
            missing = alt_set - set(ranking)
            extra = set(ranking) - alt_set
            detail = []
            if missing:
                detail.append(f"отсутствуют: {', '.join(sorted(missing))}")
            if extra:
                detail.append(f"лишние: {', '.join(sorted(extra))}")
            raise ProfileError(
                f"Ранжирование №{i}: набор альтернатив не совпадает с первым ({'; '.join(detail)}). "
                "Каждый эксперт должен упорядочить все альтернативы."
            )

    n_experts = sum(count for count, _ in rows)
    if not MIN_EXPERTS <= n_experts <= MAX_EXPERTS:
        raise ProfileError(
            f"Число экспертов должно быть от {MIN_EXPERTS} до {MAX_EXPERTS}, получено {n_experts}."
        )
    return alternatives, n_experts


# --------------------------------------------------------------------------
# 2. Процедура Копленда
# --------------------------------------------------------------------------
@dataclass
class CopelandResult:
    alternatives: list[str]
    n_experts: int
    pairwise: dict[str, dict[str, int]]  # pairwise[a][b] = число экспертов, предпочитающих a над b
    wins: dict[str, int]
    ties: dict[str, int]
    losses: dict[str, int]
    scores: dict[str, int]               # очки Копленда = победы - поражения
    borda: dict[str, int]                # очки Борда (для справки и разрешения ничьих)
    ranking: list[list[str]]             # группы равных альтернатив от лучшей к худшей
    winners: list[str]
    condorcet_winner: str | None
    condorcet_loser: str | None
    tiebreak: str | None

    def to_dict(self) -> dict:
        return {
            "alternatives": self.alternatives,
            "n_experts": self.n_experts,
            "pairwise": self.pairwise,
            "wins": self.wins,
            "ties": self.ties,
            "losses": self.losses,
            "copeland_scores": self.scores,
            "borda_scores": self.borda,
            "ranking": self.ranking,
            "winners": self.winners,
            "condorcet_winner": self.condorcet_winner,
            "condorcet_loser": self.condorcet_loser,
            "tiebreak": self.tiebreak,
        }


def copeland(rows: list[tuple[int, list[str]]], tiebreak: str | None = None) -> CopelandResult:
    """
    Считает результат по правилу Копленда.

    Для каждой пары (a, b): если большинство экспертов предпочитает a, то у a +1 победа,
    у b +1 поражение; при равенстве голосов - ничья (0 очков обоим).
    Очки Копленда = победы - поражения.

    tiebreak: None (ничьи остаются) или 'borda' (равные по Копленду
    упорядочиваются по очкам Борда).
    """
    if tiebreak not in (None, "borda"):
        raise ValueError("tiebreak должен быть None или 'borda'.")

    alternatives, n_experts = validate_profile(rows)
    m = len(alternatives)

    pairwise = {a: {b: 0 for b in alternatives if b != a} for a in alternatives}
    borda = {a: 0 for a in alternatives}

    for count, ranking in rows:
        pos = {alt: i for i, alt in enumerate(ranking)}
        for a in alternatives:
            borda[a] += count * (m - 1 - pos[a])
        for a, b in combinations(alternatives, 2):
            if pos[a] < pos[b]:
                pairwise[a][b] += count
            else:
                pairwise[b][a] += count

    wins = {a: 0 for a in alternatives}
    ties = {a: 0 for a in alternatives}
    losses = {a: 0 for a in alternatives}
    for a, b in combinations(alternatives, 2):
        if pairwise[a][b] > pairwise[b][a]:
            wins[a] += 1
            losses[b] += 1
        elif pairwise[a][b] < pairwise[b][a]:
            wins[b] += 1
            losses[a] += 1
        else:
            ties[a] += 1
            ties[b] += 1

    scores = {a: wins[a] - losses[a] for a in alternatives}

    def key(a: str):
        return (scores[a], borda[a]) if tiebreak == "borda" else (scores[a],)

    ordered = sorted(alternatives, key=key, reverse=True)
    ranking: list[list[str]] = []
    for alt in ordered:
        if ranking and key(ranking[-1][0]) == key(alt):
            ranking[-1].append(alt)
        else:
            ranking.append([alt])

    condorcet_winner = next((a for a in alternatives if wins[a] == m - 1), None)
    condorcet_loser = next((a for a in alternatives if losses[a] == m - 1), None)

    return CopelandResult(
        alternatives=alternatives, n_experts=n_experts, pairwise=pairwise,
        wins=wins, ties=ties, losses=losses, scores=scores, borda=borda,
        ranking=ranking, winners=ranking[0],
        condorcet_winner=condorcet_winner, condorcet_loser=condorcet_loser,
        tiebreak=tiebreak,
    )


# --------------------------------------------------------------------------
# 3. Вывод результата
# --------------------------------------------------------------------------
def format_report(res: CopelandResult) -> str:
    alts = res.alternatives
    w = max(len(a) for a in alts) + 2  # ширина столбцов матрицы
    nw = max(w, 16)                    # ширина колонки с названиями в таблице очков
    out: list[str] = []

    out.append(f"Экспертов: {res.n_experts}, альтернатив: {len(alts)}")
    out.append("")

    out.append("Матрица попарных сравнений (строка предпочтена столбцу, число экспертов):")
    out.append("".ljust(w) + "".join(b.ljust(w) for b in alts))
    for a in alts:
        cells = ["-" if a == b else str(res.pairwise[a][b]) for b in alts]
        out.append(a.ljust(w) + "".join(c.ljust(w) for c in cells))
    out.append("")

    out.append("Попарные дуэли:")
    for a, b in combinations(alts, 2):
        x, y = res.pairwise[a][b], res.pairwise[b][a]
        verdict = a if x > y else b if y > x else "ничья"
        out.append(f"  {a} vs {b}: {x}:{y} -> {verdict}")
    out.append("")

    out.append(
        "Очки Копленда:".ljust(nw)
        + "".join(h.ljust(10) for h in ["Победы", "Ничьи", "Поражения", "Очки", "Борда"])
    )
    for group in res.ranking:
        for a in group:
            out.append(
                a.ljust(nw)
                + "".join(str(v).ljust(10) for v in
                          [res.wins[a], res.ties[a], res.losses[a], res.scores[a], res.borda[a]])
            )
    out.append("")

    out.append("Итоговое ранжирование (лучшая альтернатива первая):")
    for place, group in enumerate(res.ranking, start=1):
        out.append(f"  {place}. " + " ~ ".join(group) + ("  (равны)" if len(group) > 1 else ""))
    out.append("")

    if len(res.winners) == 1:
        out.append(f"Победитель по Копленду: {res.winners[0]}")
    else:
        out.append("Победители по Копленду (ничья): " + ", ".join(res.winners))
        if res.tiebreak is None:
            out.append("  Подсказка: запустите с --tiebreak borda, чтобы разрешить ничью по Борда.")
    out.append(
        "Победитель Кондорсе: "
        + (res.condorcet_winner if res.condorcet_winner else "нет (возможен цикл предпочтений)")
    )
    out.append(
        "Проигравший Кондорсе: " + (res.condorcet_loser if res.condorcet_loser else "нет")
    )
    return "\n".join(out)


# --------------------------------------------------------------------------
# 4. Командная строка
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Анализ коллективного выбора по процедуре Копленда "
                    "(3-5 альтернатив, 3-20 экспертов)."
    )
    parser.add_argument("profile", nargs="?",
                        help="путь к файлу профиля; если не указан, читается stdin")
    parser.add_argument("--tiebreak", choices=["borda"], default=None,
                        help="как разрешать ничьи по Копленду (по умолчанию не разрешать)")
    parser.add_argument("--json", metavar="FILE", dest="json_out",
                        help="дополнительно сохранить результат в JSON-файл")
    args = parser.parse_args(argv)

    try:
        text = Path(args.profile).read_text(encoding="utf-8") if args.profile else sys.stdin.read()
        result = copeland(parse_profile(text), tiebreak=args.tiebreak)
    except (ProfileError, OSError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    print(format_report(result))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nРезультат сохранён в {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())