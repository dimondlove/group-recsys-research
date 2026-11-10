"""Загрузка MovieLens (ml-100k, ml-1m, ml-latest-small) и контейнер Ratings."""
from __future__ import annotations

import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

BASE_URL = "https://files.grouplens.org/datasets/movielens/"
DATASETS = {"ml-100k": "ml-100k.zip", "ml-1m": "ml-1m.zip", "ml-latest-small": "ml-latest-small.zip"}


@dataclass(eq=False)
class Ratings:
    """
    Рейтинги в виде трёх параллельных массивов. Индексы пользователей и элементов
    переименованы в 0..n-1 (исходные идентификаторы хранятся в user_ids / item_ids).
    """
    user: np.ndarray
    item: np.ndarray
    rating: np.ndarray
    timestamp: np.ndarray
    n_users: int
    n_items: int
    user_ids: np.ndarray
    item_ids: np.ndarray

    @classmethod
    def from_arrays(cls, user_raw, item_raw, rating, timestamp=None) -> "Ratings":
        user_ids, u = np.unique(np.asarray(user_raw), return_inverse=True)
        item_ids, i = np.unique(np.asarray(item_raw), return_inverse=True)
        rating = np.asarray(rating, dtype=np.float32)
        ts = np.zeros(len(rating), dtype=np.int64) if timestamp is None else np.asarray(timestamp, dtype=np.int64)
        return cls(u.astype(np.int32), i.astype(np.int32), rating, ts,
                   len(user_ids), len(item_ids), user_ids, item_ids)

    @property
    def n_ratings(self) -> int:
        return len(self.rating)

    @property
    def rating_range(self) -> tuple[float, float]:
        return float(self.rating.min()), float(self.rating.max())

    def subset(self, idx) -> "Ratings":
        """Подмножество записей; размеры и идентификаторы сохраняются."""
        return Ratings(self.user[idx], self.item[idx], self.rating[idx], self.timestamp[idx],
                       self.n_users, self.n_items, self.user_ids, self.item_ids)

    def to_csr(self) -> sp.csr_matrix:
        """Разреженная матрица (пользователи x элементы) с оценками."""
        return sp.csr_matrix((self.rating, (self.user, self.item)), shape=(self.n_users, self.n_items))


def download_movielens(name: str = "ml-1m", dest="data/raw") -> Path:
    """Скачивает и распаковывает набор в dest/<name>. Требует доступа в интернет."""
    if name not in DATASETS:
        raise ValueError(f"Неизвестный набор '{name}'. Доступны: {', '.join(DATASETS)}")
    dest = Path(dest)
    target = dest / name
    if target.exists():
        return target
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / DATASETS[name]
    url = BASE_URL + DATASETS[name]
    print(f"Скачиваю {url} ...")
    try:
        urllib.request.urlretrieve(url, zip_path)
    except (urllib.error.URLError, OSError) as exc:
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Не удалось скачать {url}\nПричина: {exc}\n\n"
            "Скачайте архив вручную в браузере (https://grouplens.org/datasets/movielens/), "
            f"распакуйте его в {dest.as_posix()}/ так, чтобы существовал файл "
            f"{target.as_posix()}/{'ratings.csv' if name == 'ml-latest-small' else 'u.data' if name == 'ml-100k' else 'ratings.dat'}."
        ) from exc
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    zip_path.unlink()
    return target


_RATING_FILES = ("ratings.dat", "ratings.csv", "u.data")


def _find_ratings_file(directory: Path) -> Path | None:
    """Ищет файл рейтингов в папке или в её подпапках первого уровня (ml-1m/ml-1m/...)."""
    for folder in [directory] + sorted(p for p in directory.iterdir() if p.is_dir()):
        for fname in _RATING_FILES:
            if (folder / fname).exists():
                return folder / fname
    return None


def _read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден путь {path}.\nСкачайте MovieLens (python scripts/download_movielens.py) "
            "или распакуйте архив вручную так, чтобы внутри папки лежал файл ratings.dat "
            "(ml-1m), ratings.csv (ml-latest-small) или u.data (ml-100k). "
            "Проверьте параметр data.path в конфиге."
        )
    if path.is_dir():
        found = _find_ratings_file(path)
        if found is None:
            raise FileNotFoundError(f"В папке {path} (и её подпапках) не найдено ни одного из файлов: "
                                    f"{', '.join(_RATING_FILES)}. Распакован ли архив MovieLens?")
        path = found
    if path.name == "ratings.dat":      # ml-1m: user::item::rating::ts
        df = pd.read_csv(path, sep=":", header=None, usecols=[0, 2, 4, 6], engine="c")
    elif path.name == "ratings.csv":    # ml-latest-small: заголовок userId,movieId,rating,timestamp
        df = pd.read_csv(path, usecols=[0, 1, 2, 3])
    elif path.name == "u.data":         # ml-100k: табуляция
        df = pd.read_csv(path, sep="\t", header=None)
    else:
        raise ValueError(f"Неизвестный формат файла: {path.name}")
    df.columns = ["user", "item", "rating", "timestamp"]
    return df


def load_movielens(path, min_user_ratings: int = 0, min_item_ratings: int = 0) -> Ratings:
    """
    Читает MovieLens из папки набора или файла. Необязательная фильтрация
    (один проход: сначала пользователи, затем элементы).
    """
    df = _read_table(path)
    if min_user_ratings:
        df = df[df.groupby("user")["user"].transform("size") >= min_user_ratings]
    if min_item_ratings:
        df = df[df.groupby("item")["item"].transform("size") >= min_item_ratings]
    if df.empty:
        raise ValueError("После фильтрации не осталось данных.")
    return Ratings.from_arrays(df["user"].to_numpy(), df["item"].to_numpy(),
                               df["rating"].to_numpy(), df["timestamp"].to_numpy())