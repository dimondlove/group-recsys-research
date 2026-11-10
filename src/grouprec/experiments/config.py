"""Чтение и проверка конфигов экспериментов (YAML)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "name": "experiment",
    "seed": 42,
    "data": {"path": "data/raw/ml-1m", "min_user_ratings": 0, "min_item_ratings": 0},
    "split": {"method": "random", "test_frac": 0.2, "min_train": 5},
    "model": {"name": "als", "n_factors": 32, "reg": 0.1, "n_iters": 10},
    "groups": {
        "sizes": [3, 5, 10],
        "kinds": ["random", "similar", "dissimilar"],
        "n_groups": 100,
        "pool_size": 2000,
        "neighborhood": 50,
        "activity_band": None,         # например [0.25, 0.75]: брать в пул пользователей из этого диапазона квантилей активности
        "min_test_relevant": 3,
        "relevance_threshold": 4.0,
    },
    "aggregation": {
        "rules": ["average", "least_misery", "borda", "copeland"],
        "k": 10,
        "tiebreak": "average",
        "prefilter_top_n": None,
        "candidates": "unseen",
        "reference_rule": "average",   # с ним сравниваются остальные правила (пересечение списков, парные тесты)
    },
    "output": {"dir": "results/runs", "log": "results/experiment_log.csv", "run_id": None},
}

CANDIDATE_MODES = ("unseen", "unseen_test_union")


class ConfigError(ValueError):
    pass


def _merge(base: dict, over: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for key, val in over.items():
        if key not in base:
            raise ConfigError(f"Неизвестный параметр конфига: '{path}{key}'")
        if isinstance(base[key], dict) and isinstance(val, dict):
            out[key] = _merge(base[key], val, f"{path}{key}.")
        else:
            out[key] = val
    return out


def validate(cfg: dict) -> dict:
    from ..groups import KINDS
    from ..rules import available_rules

    g, a = cfg["groups"], cfg["aggregation"]
    unknown = [r for r in a["rules"] if r not in available_rules()]
    if unknown:
        raise ConfigError(f"Неизвестные правила: {unknown}. Доступны: {available_rules()}")
    if a["tiebreak"] is not None and a["tiebreak"] not in available_rules():
        raise ConfigError(f"Неизвестное правило для ничьих: {a['tiebreak']}")
    if a["reference_rule"] not in a["rules"]:
        raise ConfigError(f"aggregation.reference_rule ('{a['reference_rule']}') должно входить в aggregation.rules")
    if a["candidates"] not in CANDIDATE_MODES:
        raise ConfigError(f"aggregation.candidates должен быть одним из {CANDIDATE_MODES}")
    bad = [k for k in g["kinds"] if k not in KINDS]
    if bad:
        raise ConfigError(f"Неизвестные типы групп: {bad}. Доступны: {list(KINDS)}")
    if not g["sizes"] or min(g["sizes"]) < 2:
        raise ConfigError("groups.sizes: размеры групп должны быть >= 2")
    band = g["activity_band"]
    if band is not None and not (len(band) == 2 and 0 <= band[0] < band[1] <= 1):
        raise ConfigError("groups.activity_band должен быть парой квантилей [low, high], 0 <= low < high <= 1")
    if g["n_groups"] < 1 or a["k"] < 1:
        raise ConfigError("groups.n_groups и aggregation.k должны быть >= 1")
    return cfg


def load_config(path) -> dict:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return validate(_merge(DEFAULTS, raw))