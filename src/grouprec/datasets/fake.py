"""
Генератор фиктивного набора в формате MovieLens-1M (ratings.dat) для проверки конвейера
без скачивания. Есть кластеры вкусов и популярность элементов, поэтому «похожие» и
«непохожие» группы действительно различаются. Научных выводов на нём делать нельзя.

    python -m grouprec.datasets.fake data/raw/fake-ml --users 500 --items 600
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_fake_movielens(path, n_users: int = 400, n_items: int = 500, n_clusters: int = 5,
                        n_factors: int = 4, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    centers = rng.normal(size=(n_clusters, n_factors))
    cluster = rng.integers(0, n_clusters, n_users)
    U = centers[cluster] + 0.4 * rng.normal(size=(n_users, n_factors))
    V = rng.normal(size=(n_items, n_factors))
    bu = 0.3 * rng.normal(size=n_users)
    bi = 0.5 * rng.normal(size=n_items)

    popularity = 1.0 / np.arange(1, n_items + 1) ** 0.8
    popularity = popularity[rng.permutation(n_items)]
    popularity /= popularity.sum()
    n_rated = np.clip(rng.lognormal(mean=4.0, sigma=0.6, size=n_users).astype(int), 20, n_items // 2)

    lines = []
    for u in range(n_users):
        items = rng.choice(n_items, size=n_rated[u], replace=False, p=popularity)
        raw = 3.5 + bu[u] + bi[items] + (U[u] @ V[items].T) * 0.6 + 0.5 * rng.normal(size=len(items))
        stars = np.clip(np.rint(raw), 1, 5).astype(int)
        ts = np.sort(rng.integers(956_000_000, 1_046_000_000, len(items)))
        lines += [f"{u + 1}::{i + 1}::{s}::{t}" for i, s, t in zip(items, stars, ts)]
    (path / "ratings.dat").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("path")
    ap.add_argument("--users", type=int, default=400)
    ap.add_argument("--items", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print("Готово:", make_fake_movielens(a.path, a.users, a.items, seed=a.seed))