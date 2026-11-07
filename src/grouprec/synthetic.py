"""Генератор синтетических групп для отладки и пилотных экспериментов."""
from __future__ import annotations

import numpy as np


def make_group_ratings(n_members: int, n_items: int, cohesion: float,
                       rng: np.random.Generator | int | None = None,
                       noise: float = 0.3, scale: tuple[float, float] = (1.0, 5.0),
                       round_to_int: bool = True) -> np.ndarray:
    """
    Оценки группы по модели: оценка = cohesion * (общее качество элемента)
                                      + (1 - cohesion) * (личный вкус участника) + шум.
    cohesion = 1 даёт почти одинаковые вкусы, 0 даёт независимые вкусы.
    Результат приводится к шкале scale (по умолчанию 1-5, целые - как в MovieLens).
    """
    if not 0.0 <= cohesion <= 1.0:
        raise ValueError("cohesion должен лежать в [0, 1].")
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    common = rng.normal(size=n_items)
    personal = rng.normal(size=(n_members, n_items))
    latent = cohesion * common + (1 - cohesion) * personal + noise * rng.normal(size=(n_members, n_items))
    # стандартизируем и растягиваем на шкалу: ±2 стандартных отклонения = края шкалы
    lo, hi = scale
    z = (latent - latent.mean()) / (latent.std() or 1.0)
    r = np.clip((z + 2) / 4, 0, 1) * (hi - lo) + lo
    return np.rint(r) if round_to_int else r