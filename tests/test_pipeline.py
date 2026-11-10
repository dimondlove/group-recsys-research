"""Тесты конвейера эксперимента на фиктивных данных."""
import tempfile
import unittest
import urllib.error
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd

from grouprec.datasets import download_movielens, load_movielens, split_ratings
from grouprec.datasets.fake import make_fake_movielens
from grouprec.experiments.config import ConfigError, load_config, validate, _merge, DEFAULTS
from grouprec.experiments.evaluation import evaluate_list
from grouprec.experiments.runner import run_experiment
from grouprec.groups import form_groups, similarity_matrix
from grouprec.models import build_model


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        make_fake_movielens(cls.root / "data" / "fake", n_users=250, n_items=300, seed=3)
        cls.ratings = load_movielens(cls.root / "data" / "fake")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # ---- данные и разбиение
    def test_split_is_a_partition_without_leakage(self):
        r = self.ratings
        for method in ("random", "time"):
            tr, te = split_ratings(r, 0.2, method, seed=5, min_train=5)
            self.assertEqual(tr.n_ratings + te.n_ratings, r.n_ratings)
            pairs_tr = set(zip(tr.user.tolist(), tr.item.tolist()))
            pairs_te = set(zip(te.user.tolist(), te.item.tolist()))
            self.assertFalse(pairs_tr & pairs_te)
            self.assertGreaterEqual(np.bincount(tr.user, minlength=r.n_users).min(), 5)

    def test_time_split_puts_latest_in_test(self):
        tr, te = split_ratings(self.ratings, 0.2, "time", seed=0)
        last_train = np.zeros(self.ratings.n_users)
        np.maximum.at(last_train, tr.user, tr.timestamp)
        first_test = np.full(self.ratings.n_users, np.inf)
        np.minimum.at(first_test, te.user, te.timestamp)
        has_test = np.isfinite(first_test)
        self.assertTrue((first_test[has_test] >= last_train[has_test]).all())

    def test_split_reproducible(self):
        a, _ = split_ratings(self.ratings, 0.2, seed=9)
        b, _ = split_ratings(self.ratings, 0.2, seed=9)
        np.testing.assert_array_equal(a.item, b.item)

    def test_load_csv_format(self):
        d = self.root / "csvfmt"
        d.mkdir()
        (d / "ratings.csv").write_text("userId,movieId,rating,timestamp\n10,5,4.0,100\n10,7,3.5,101\n20,5,2.0,102\n")
        r = load_movielens(d)
        self.assertEqual((r.n_users, r.n_items, r.n_ratings), (2, 2, 3))
        self.assertEqual(r.user_ids.tolist(), [10, 20])

    def test_missing_data_path_gives_clear_error(self):
        with self.assertRaises(FileNotFoundError) as cm:
            load_movielens(self.root / "нет_такой_папки")
        self.assertIn("download_movielens", str(cm.exception))

    def test_nested_folder_is_found(self):
        # Windows часто распаковывает архив как ml-1m/ml-1m/ratings.dat
        outer = self.root / "nested" / "ml-1m"
        inner = outer / "ml-1m"
        inner.mkdir(parents=True)
        (inner / "ratings.dat").write_text("1::10::5::100\n1::11::3::101\n2::10::4::102\n", encoding="utf-8")
        r = load_movielens(outer)
        self.assertEqual((r.n_users, r.n_items, r.n_ratings), (2, 2, 3))

    def test_empty_folder_gives_clear_error(self):
        empty = self.root / "empty_dir"
        empty.mkdir()
        with self.assertRaises(FileNotFoundError):
            load_movielens(empty)

    def test_download_failure_explains_manual_install(self):
        err = urllib.error.URLError("CERTIFICATE_VERIFY_FAILED")
        with mock.patch("urllib.request.urlretrieve", side_effect=err):
            with self.assertRaises(RuntimeError) as cm:
                download_movielens("ml-1m", self.root / "dl")
        msg = str(cm.exception)
        self.assertIn("вручную", msg)
        self.assertIn("ratings.dat", msg)
        self.assertFalse((self.root / "dl" / "ml-1m.zip").exists())

    # ---- модель
    def test_als_beats_global_mean(self):
        tr, te = split_ratings(self.ratings, 0.2, seed=1)
        model = build_model("als", n_factors=8, n_iters=8, seed=0).fit(tr)
        baseline = np.sqrt(np.mean((te.rating - tr.rating.mean()) ** 2))
        self.assertLess(model.rmse(te), 0.8 * baseline)
        pred = model.predict(np.array([0, 1]))
        self.assertEqual(pred.shape, (2, self.ratings.n_items))
        self.assertTrue(pred.min() >= 1.0 and pred.max() <= 5.0)

    def test_unknown_model(self):
        with self.assertRaises(KeyError):
            build_model("нет_такой")

    # ---- метрика
    def test_evaluate_list_by_hand(self):
        rel = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0]], dtype=bool)
        res = evaluate_list(rel, np.array([0, 1]))
        self.assertEqual(res["n_valid"], 2)                     # участник без релевантных не оценивается
        second = 1 / np.log2(3)
        expected = [1.0, second / (1 + second)]
        self.assertAlmostEqual(res["ndcg_mean"], np.mean(expected))
        self.assertAlmostEqual(res["ndcg_min"], min(expected))
        self.assertEqual(res["hit_mean"], 1.0)
        self.assertEqual(res["share_zero"], 0.0)

    def test_evaluate_list_no_valid_members(self):
        res = evaluate_list(np.zeros((2, 4), dtype=bool), np.array([0, 1]))
        self.assertEqual(res["n_valid"], 0)
        self.assertTrue(np.isnan(res["ndcg_mean"]))

    # ---- группы
    def test_group_kinds_are_separated_by_similarity(self):
        tr, _ = split_ratings(self.ratings, 0.2, seed=1)
        users = np.arange(self.ratings.n_users)
        S = similarity_matrix(tr, users)
        self.assertTrue(np.allclose(S, S.T))
        rng = np.random.default_rng(0)
        mean = {k: np.mean([g.similarity for g in form_groups(users, S, 5, 60, k, rng, 40)])
                for k in ("random", "similar", "dissimilar")}
        self.assertGreater(mean["similar"], mean["random"])
        self.assertGreater(mean["random"], mean["dissimilar"])

    def test_group_members_unique_and_sized(self):
        tr, _ = split_ratings(self.ratings, 0.2, seed=1)
        users = np.arange(self.ratings.n_users)
        S = similarity_matrix(tr, users)
        for kind in ("random", "similar", "dissimilar"):
            for g in form_groups(users, S, 7, 10, kind, np.random.default_rng(1), 30):
                self.assertEqual(len(set(g.members.tolist())), 7)

    # ---- конфиг и запуск
    def test_config_validation(self):
        with self.assertRaises(ConfigError):
            _merge(DEFAULTS, {"опечатка": 1})
        bad = _merge(DEFAULTS, {"aggregation": {"rules": ["нет_такого"]}})
        with self.assertRaises(ConfigError):
            validate(bad)

    def _cfg(self, run_id):
        cfg = _merge(DEFAULTS, {
            "name": "t", "seed": 7,
            "data": {"path": "data/fake"},
            "model": {"n_factors": 6, "n_iters": 5},
            "groups": {"sizes": [3], "kinds": ["random", "similar"], "n_groups": 8,
                       "pool_size": 250, "min_test_relevant": 1, "neighborhood": 30},
            "aggregation": {"rules": ["average", "least_misery", "borda", "copeland"], "k": 5},
            "output": {"run_id": run_id},
        })
        return validate(cfg)

    def test_activity_band_validation_and_effect(self):
        for bad in ([0.5, 0.2], [0.0], [-0.1, 0.5], [0.2, 1.5]):
            with self.assertRaises(ConfigError):
                validate(_merge(DEFAULTS, {"groups": {"activity_band": bad}}))
        cfg = self._cfg("band")
        cfg["groups"]["activity_band"] = [0.25, 0.75]
        d = run_experiment(cfg, self.root, progress=False)
        df = pd.read_csv(d / "results.csv")
        counts = np.bincount(load_movielens(self.root / "data" / "fake").user)
        self.assertTrue(len(df) > 0 and df["mean_train_ratings"].between(0, counts.max()).all())

    def test_run_is_reproducible_and_complete(self):
        d1 = run_experiment(self._cfg("a"), self.root, progress=False)
        d2 = run_experiment(self._cfg("b"), self.root, progress=False)
        a = pd.read_csv(d1 / "results.csv")
        b = pd.read_csv(d2 / "results.csv")
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(set(a["rule"]), {"average", "least_misery", "borda", "copeland"})
        for name in ("config.yaml", "meta.json", "summary.csv"):
            self.assertTrue((d1 / name).exists())
        log = (self.root / "results" / "experiment_log.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(log[0].split(",")[0], "run_id")
        self.assertGreaterEqual(len(log), 3)


if __name__ == "__main__":
    unittest.main()