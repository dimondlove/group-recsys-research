import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from grouprec import available_rules, compare_rules, make_group_ratings  # noqa: E402

N_GROUPS, GROUP_SIZE, N_ITEMS, K, SEED = 200, 5, 100, 5, 42

rng = np.random.default_rng(SEED)
rules = available_rules()
print(
    f"Групп на конфигурацию: {N_GROUPS}, размер группы: {GROUP_SIZE}, элементов: {N_ITEMS}, k={K}, seed={SEED}\n"
)

for cohesion in (0.0, 0.5, 0.9):
    stats = {name: {"mean": [], "min": [], "gini": []} for name in rules}
    for _ in range(N_GROUPS):
        ratings = make_group_ratings(GROUP_SIZE, N_ITEMS, cohesion, rng)
        res = compare_rules(ratings, k=K, tiebreak="average")
        for name, r in res["rules"].items():
            for key in ("mean", "min", "gini"):
                stats[name][key].append(r["fairness"][key])
    print(f"cohesion={cohesion}")
    print(f"  {'правило':<15}{'среднее':>9}{'минимум':>9}{'Джини':>9}")
    for name in rules:
        s = stats[name]
        print(
            f"  {name:<15}{np.mean(s['mean']):>9.3f}{np.mean(s['min']):>9.3f}{np.mean(s['gini']):>9.3f}"
        )
    print()
