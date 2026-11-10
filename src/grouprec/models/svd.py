"""
Матричная факторизация (модель латентных факторов типа SVD), обучаемая методом ALS.

Оценка(u, i) = mu + b_u + b_i + p_u . q_i, значения ограничиваются шкалой рейтингов.
Смещения считаются с демпфированием, факторы подбираются чередующимися наименьшими
квадратами с регуляризацией, пропорциональной числу оценок (ALS-WR). Только numpy.
"""
from __future__ import annotations

import numpy as np

from ..datasets.movielens import Ratings
from .base import Recommender, register_model


def _csr_parts(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, n_rows: int):
    order = np.lexsort((cols, rows))
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=n_rows))])
    return indptr, cols[order], vals[order].astype(np.float64)


def _als_step(indptr, indices, data, fixed: np.ndarray, out: np.ndarray, reg: float) -> None:
    eye = np.eye(fixed.shape[1])
    for row in range(len(indptr) - 1):
        s, e = indptr[row], indptr[row + 1]
        if s == e:
            out[row] = 0.0
            continue
        f = fixed[indices[s:e]]
        out[row] = np.linalg.solve(f.T @ f + reg * (e - s) * eye, f.T @ data[s:e])


@register_model
class ALSFactorization(Recommender):
    name = "als"

    def __init__(self, n_factors: int = 32, reg: float = 0.1, n_iters: int = 10,
                 bias_reg: float = 10.0, seed: int = 0):
        self.n_factors, self.reg, self.n_iters = n_factors, reg, n_iters
        self.bias_reg, self.seed = bias_reg, seed
        self.fitted = False

    def fit(self, train: Ratings) -> "ALSFactorization":
        u, i, r = train.user, train.item, train.rating.astype(np.float64)
        nu, ni = train.n_users, train.n_items
        self.lo, self.hi = train.rating_range
        self.mu = float(r.mean())

        # смещения с демпфированием: сначала элементы, затем пользователи
        sum_i = np.bincount(i, weights=r - self.mu, minlength=ni)
        self.bi = sum_i / (np.bincount(i, minlength=ni) + self.bias_reg)
        sum_u = np.bincount(u, weights=r - self.mu - self.bi[i], minlength=nu)
        self.bu = sum_u / (np.bincount(u, minlength=nu) + self.bias_reg)

        resid = r - self.mu - self.bu[u] - self.bi[i]
        by_user = _csr_parts(u, i, resid, nu)
        by_item = _csr_parts(i, u, resid, ni)

        rng = np.random.default_rng(self.seed)
        self.P = np.zeros((nu, self.n_factors))
        self.Q = rng.normal(0.0, 0.1, size=(ni, self.n_factors))
        for _ in range(self.n_iters):
            _als_step(*by_user, self.Q, self.P, self.reg)
            _als_step(*by_item, self.P, self.Q, self.reg)
        self.n_items = ni
        self.fitted = True
        self.train_rmse_ = self.rmse(train)
        return self

    def _check(self):
        if not self.fitted:
            raise RuntimeError("Сначала вызовите fit().")

    def predict(self, users: np.ndarray) -> np.ndarray:
        self._check()
        users = np.asarray(users, dtype=int)
        out = self.mu + self.bu[users, None] + self.bi[None, :] + self.P[users] @ self.Q.T
        return np.clip(out, self.lo, self.hi).astype(np.float32)

    def predict_pairs(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        self._check()
        out = self.mu + self.bu[users] + self.bi[items] + np.einsum("nk,nk->n", self.P[users], self.Q[items])
        return np.clip(out, self.lo, self.hi)