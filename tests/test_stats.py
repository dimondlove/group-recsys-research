"""Тесты парной статистики и отчёта по запуску."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from grouprec.analysis import (analyze_run, by_cohesion, equivalence_vs_reference, friedman_by_config,
                               paired_vs_reference)
from grouprec.datasets.fake import make_fake_movielens
from grouprec.experiments.config import DEFAULTS, _merge, validate
from grouprec.experiments.runner import run_experiment


def make_df(n=80, seed=0):
    """A - эталон; B заметно лучше; C отличается только шумом; D совпадает с A."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.3, 0.1, n)
    rows = []
    for i in range(n):
        vals = {"A": base[i], "B": base[i] + 0.05 + 0.01 * rng.normal(),
                "C": base[i] + 0.01 * rng.normal(), "D": base[i]}
        for rule, v in vals.items():
            rows.append({"group_id": f"g{i}", "kind": "random", "size": 5, "rule": rule, "m": v})
    return pd.DataFrame(rows)


class StatsTests(unittest.TestCase):
    def test_detects_real_difference_and_ignores_noise(self):
        t = paired_vs_reference(make_df(), "m", "A").set_index("rule")
        self.assertTrue(t.loc["B", "significant"])
        self.assertAlmostEqual(t.loc["B", "mean_diff"], 0.05, delta=0.01)
        self.assertFalse(t.loc["C", "significant"])
        self.assertEqual(t.loc["D", "p"], 1.0)            # полностью совпадает с эталоном
        self.assertTrue((t["p_holm"] >= t["p"] - 1e-12).all())

    def test_friedman(self):
        f = friedman_by_config(make_df(), "m")
        self.assertLess(f["p"].iloc[0], 0.001)
        self.assertGreater(f["kendall_w"].iloc[0], 0)

    def test_friedman_identical_rules_gives_nan(self):
        df = make_df()
        df["m"] = df["group_id"].map(lambda g: hash(g) % 7 / 10)   # одинаково для всех правил
        self.assertTrue(np.isnan(friedman_by_config(df, "m")["p"].iloc[0]))

    def test_equivalence_tost(self):
        t = equivalence_vs_reference(make_df(), "m", "A", margin=0.02).set_index("rule")
        self.assertTrue(t.loc["D", "equivalent"])          # совпадает с эталоном
        self.assertTrue(t.loc["C", "equivalent"])          # шум порядка 0.01 внутри допуска 0.02
        self.assertFalse(t.loc["B", "equivalent"])         # реальный сдвиг +0.05 за пределами допуска
        self.assertLess(t.loc["C", "ci90_lo"], t.loc["C", "ci90_hi"])
        tight = equivalence_vs_reference(make_df(), "m", "A", margin=0.001).set_index("rule")
        self.assertFalse(tight.loc["C", "equivalent"])     # слишком узкий допуск не доказывается

    def test_unknown_reference(self):
        with self.assertRaises(KeyError):
            paired_vs_reference(make_df(), "m", "Z")


def make_cohesion_df(n=400, seed=1):
    """B лучше эталона A только в слабо согласованных группах; C - шум."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        coh = rng.uniform(0, 1)
        base = rng.normal(0.3, 0.1)
        gain = 0.08 * (1 - coh)
        vals = {"A": base, "B": base + gain + 0.01 * rng.normal(), "C": base + 0.01 * rng.normal()}
        for rule, v in vals.items():
            rows.append({"group_id": f"g{i}", "kind": "random", "size": 5, "rule": rule, "m": v,
                         "cohesion_pred": coh, "overlap_ref": 1.0 if rule == "A" else 0.8})
    return pd.DataFrame(rows)


class StratifyTests(unittest.TestCase):
    def test_effect_concentrated_in_low_cohesion(self):
        strat, corr = by_cohesion(make_cohesion_df(), "m", "A")
        b = strat[strat["rule"] == "B"].set_index("bin")
        self.assertTrue(b.loc["Q1", "significant"])
        self.assertGreater(b.loc["Q1", "mean_diff"], b.loc["Q4", "mean_diff"] + 0.04)
        self.assertEqual(set(strat["bin"]), {"Q1", "Q2", "Q3", "Q4"})
        self.assertAlmostEqual(strat[strat["rule"] == "B"]["n"].sum(), 400, delta=0)
        rho_b = corr[corr["rule"] == "B"]["spearman_rho"].iloc[0]
        self.assertLess(rho_b, -0.8)
        self.assertFalse(strat[strat["rule"] == "C"]["significant"].any())


class ReportTests(unittest.TestCase):
    def test_analyze_run_writes_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_fake_movielens(root / "data" / "f", n_users=250, n_items=300, seed=2)
            cfg = validate(_merge(DEFAULTS, {
                "name": "r", "seed": 3, "data": {"path": "data/f"},
                "model": {"n_factors": 6, "n_iters": 4},
                "groups": {"sizes": [3], "kinds": ["random", "similar"], "n_groups": 12,
                           "pool_size": 250, "min_test_relevant": 1, "neighborhood": 30},
                "aggregation": {"rules": ["average", "least_misery", "borda", "copeland"], "k": 5,
                                "candidates": "unseen_test_union"},
                "output": {"run_id": "x"}}))
            run_dir = run_experiment(cfg, root, progress=False)
            df = pd.read_csv(run_dir / "results.csv")
            self.assertIn("overlap_ref", df.columns)
            self.assertTrue((df.loc[df["rule"] == "average", "overlap_ref"] == 1.0).all())
            res = analyze_run(run_dir)
            self.assertEqual(res["reference"], "average")
            for name in ("diagnostics.csv", "overlap.csv", "stats_paired.csv", "stats_friedman.csv",
                         "stats_by_cohesion.csv", "corr_cohesion.csv", "stats_equivalence.csv"):
                self.assertTrue((run_dir / name).exists(), name)
            self.assertEqual(set(res["paired"]["rule"]), {"least_misery", "borda", "copeland"})

    def test_bad_reference_rule_in_config(self):
        from grouprec.experiments.config import ConfigError
        with self.assertRaises(ConfigError):
            validate(_merge(DEFAULTS, {"aggregation": {"rules": ["borda"], "reference_rule": "average"}}))


if __name__ == "__main__":
    unittest.main()