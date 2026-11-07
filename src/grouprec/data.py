"""Загрузка и проверка входных данных, настраиваемые ограничения."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class InputError(ValueError):
    """Ошибка во входных данных."""


@dataclass(frozen=True)
class Limits:
    """Ограничения на размер задачи. None = без ограничения сверху."""
    min_items: int = 2
    max_items: int | None = None
    min_members: int = 2
    max_members: int | None = None


# Ограничения учебного задания: 3-5 альтернатив, 3-20 экспертов
TASK_LIMITS = Limits(min_items=3, max_items=5, min_members=3, max_members=20)


def validate_ratings(ratings, limits: Limits | None = None) -> np.ndarray:
    """Приводит вход к float-матрице (участники x элементы) и проверяет ограничения."""
    limits = limits or Limits()
    try:
        r = np.asarray(ratings, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InputError(f"Матрица оценок должна быть числовой: {exc}") from exc
    if r.ndim != 2:
        raise InputError("Матрица оценок должна быть двумерной: участники x элементы.")
    if not np.isfinite(r).all():
        raise InputError("В матрице оценок есть пропуски или нечисловые значения. "
                         "Заполните их (например, предсказаниями рекомендателя).")
    n_members, n_items = r.shape
    if n_members < limits.min_members or (limits.max_members and n_members > limits.max_members):
        hi = f" до {limits.max_members}" if limits.max_members else ""
        raise InputError(f"Число участников должно быть от {limits.min_members}{hi}, получено {n_members}.")
    if n_items < limits.min_items or (limits.max_items and n_items > limits.max_items):
        hi = f" до {limits.max_items}" if limits.max_items else ""
        raise InputError(f"Число элементов должно быть от {limits.min_items}{hi}, получено {n_items}.")
    return r


def load_ratings_csv(path, limits: Limits | None = None):
    """
    CSV: первая строка - названия элементов (первая ячейка игнорируется),
    далее по строке на участника: имя, оценки. Разделитель ',' или ';'
    (при ';' допускается десятичная запятая, как в русском Excel).
    Возвращает (имена участников, названия элементов, матрица).
    """
    text = Path(path).read_text(encoding="utf-8-sig")
    first = text.splitlines()[0] if text.strip() else ""
    delimiter = ";" if first.count(";") > first.count(",") else ","
    rows = [r for r in csv.reader(text.splitlines(), delimiter=delimiter) if any(c.strip() for c in r)]
    if len(rows) < 2:
        raise InputError("В CSV должна быть строка заголовка и хотя бы один участник.")
    items = [c.strip() for c in rows[0][1:]]
    members, data = [], []
    for lineno, row in enumerate(rows[1:], start=2):
        if len(row) - 1 != len(items):
            raise InputError(f"Строка {lineno}: ожидается {len(items)} оценок, получено {len(row) - 1}.")
        members.append(row[0].strip() or f"участник {lineno - 1}")
        try:
            vals = [float(c.strip().replace(",", ".") if delimiter == ";" else c.strip()) for c in row[1:]]
        except ValueError:
            raise InputError(f"Строка {lineno}: оценки должны быть числами (пропуски недопустимы).") from None
        data.append(vals)
    return members, items, validate_ratings(data, limits)


def parse_rankings(text: str, limits: Limits | None = None):
    """
    Профиль ранжирований (формат учебного задания, расширенный).
    Строка = один эксперт, от лучшей альтернативы к худшей.
      '>' или ','  - строгое предпочтение;
      '='          - равенство (слабый порядок): A > B = C > D;
      'N:' в начале - так ответили N экспертов.
    Возвращает (имена экспертов, альтернативы, матрица псевдооценок), где оценка =
    число альтернатив, которые данная альтернатива строго превосходит у эксперта.
    """
    parsed: list[tuple[int, list[list[str]]]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        count = 1
        m = re.match(r"^(\d+)\s*:\s*(.+)$", line)
        if m:
            count, line = int(m.group(1)), m.group(2)
            if count < 1:
                raise InputError(f"Строка {lineno}: число экспертов должно быть >= 1.")
        levels = [[a.strip() for a in lvl.split("=") if a.strip()]
                  for lvl in re.split(r"[>,]", line)]
        levels = [lvl for lvl in levels if lvl]
        if not levels:
            raise InputError(f"Строка {lineno}: не найдено ни одной альтернативы.")
        parsed.append((count, levels))
    if not parsed:
        raise InputError("Профиль пуст: нет ни одного ранжирования.")

    items = [a for lvl in parsed[0][1] for a in lvl]
    if len(items) != len(set(items)):
        raise InputError("В первом ранжировании есть повторяющиеся альтернативы.")
    index = {a: i for i, a in enumerate(items)}

    members, data = [], []
    for i, (count, levels) in enumerate(parsed, start=1):
        flat = [a for lvl in levels for a in lvl]
        if len(flat) != len(set(flat)):
            raise InputError(f"Ранжирование №{i}: альтернативы повторяются.")
        if set(flat) != set(items):
            raise InputError(f"Ранжирование №{i}: набор альтернатив не совпадает с первым. "
                             "Каждый эксперт должен упорядочить все альтернативы.")
        row = np.zeros(len(items))
        below = 0
        for lvl in reversed(levels):          # от худшего уровня к лучшему
            for a in lvl:
                row[index[a]] = below
            below += len(lvl)
        for c in range(count):
            data.append(row.copy())
            members.append(f"эксперт {len(members) + 1}")
    return members, items, validate_ratings(np.array(data), limits)


def top_n_union(ratings: np.ndarray, n: int) -> np.ndarray:
    """
    Предварительный отбор кандидатов: объединение top-n элементов каждого участника.
    Возвращает отсортированные индексы столбцов. Приём нужен для правил с
    квадратичной памятью (Копленд) при тысячах элементов; описывайте его в методике.
    """
    if n < 1:
        raise InputError("n должно быть >= 1.")
    cand: set[int] = set()
    for r in ratings:
        cand.update(np.argsort(-r, kind="stable")[:n].tolist())
    return np.array(sorted(cand))