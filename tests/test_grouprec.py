import itertools
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from grouprec import (TASK_LIMITS, InputError, Limits, available_rules, compare_rules,  # noqa: E402
                      get_rule, group_cohesion, load_ratings_csv, make_group_ratings,
                      pairwise_preferences, parse_rankings, rank_items, register,
                      top_n_union, top_ties, unregister, validate_ratings)
from grouprec.rules import AggregationRule  # noqa: E402


def copeland_naive(r):
    """Эталон: прямая реализация Копленда тройным циклом."""
    m, n = r.shape
    score = np.zeros(n)
    for a, b in itertools.permutations(range(n), 2):
        x = int((r[:, a] > r[:, b]).sum())
        y = int((r[:, b] > r[:, a]).sum())
        score[a] += (x > y) - (x < y)
    return score


class RuleTests(unittest.TestCase):
    def test_registry(self):
        self.assertEqual(set(available_rules()),
                         {"average", "least_misery", "most_pleasure", "borda", "copeland"})
        with self.assertRaises(KeyError):
            get_rule("нет_такого")

    def test_basic_rules(self):
        r = np.array([[5, 1, 3], [1, 5, 3], [4, 4, 3.0]])
        np.testing.assert_allclose(get_rule("average").scores(r), [10 / 3, 10 / 3, 3])
        np.testing.assert_allclose(get_rule("least_misery").scores(r), [1, 1, 3])
        np.testing.assert_allclose(get_rule("most_pleasure").scores(r), [5, 5, 3])

    def test_copeland_matches_naive_on_random_with_ties(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            r = rng.integers(1, 5, size=(rng.integers(3, 8), rng.integers(3, 12))).astype(float)
            np.testing.assert_allclose(get_rule("copeland").scores(r), copeland_naive(r))

    def test_borda_with_ties(self):
        # одна строка, все оценки равны: каждому по 0,5 * (n - 1)
        np.testing.assert_allclose(get_rule("borda").scores(np.ones((1, 5))), [2.0] * 5)
        # строгий порядок: 0, 1, 2
        np.testing.assert_allclose(get_rule("borda").scores(np.array([[1.0, 2.0, 3.0]])), [0, 1, 2])
        # слабый порядок 3 > 2 = 2: очки 2, 0.5, 0.5
        np.testing.assert_allclose(get_rule("borda").scores(np.array([[2.0, 3.0, 2.0]])), [0.5, 2, 0.5])

    def test_pairwise_matrix(self):
        r = np.array([[3, 2, 1], [1, 2, 3], [2, 2, 2.0]])
        p = pairwise_preferences(r)
        self.assertEqual(p[0, 1], 1)   # первый предпочёл 0 над 1; третий не голосует
        self.assertEqual(p[1, 0], 1)
        self.assertTrue((np.diag(p) == 0).all())

    def test_condorcet_cycle_gives_tie(self):
        r = np.array([[3, 2, 1], [1, 3, 2], [2, 1, 3.0]])
        sc = get_rule("copeland").scores(r)
        np.testing.assert_allclose(sc, [0, 0, 0])
        self.assertEqual(top_ties(sc).tolist(), [0, 1, 2])

    def test_rank_items_tiebreak_and_determinism(self):
        sc = np.array([1.0, 3.0, 3.0, 2.0])
        self.assertEqual(rank_items(sc).tolist(), [1, 2, 3, 0])           # ничья по индексу
        self.assertEqual(rank_items(sc, tiebreak=[0, 0, 5, 0]).tolist(), [2, 1, 3, 0])
        self.assertEqual(rank_items(sc, k=2).tolist(), [1, 2])
        self.assertEqual(top_ties(sc, tiebreak=[0, 0, 5, 0]).tolist(), [2])

    def test_register_custom_rule(self):
        @register
        class Median(AggregationRule):
            name = "median_test"
            title = "Медиана"
            def scores(self, ratings):
                return np.median(ratings, axis=0)
        try:
            np.testing.assert_allclose(get_rule("median_test").scores(np.array([[1, 5], [2, 4], [9, 3.0]])), [2, 4])
            self.assertIn("median_test", available_rules())
        finally:
            unregister("median_test")
        self.assertNotIn("median_test", available_rules())


class DataTests(unittest.TestCase):
    def test_limits(self):
        validate_ratings(np.ones((3, 3)), TASK_LIMITS)
        with self.assertRaises(InputError):
            validate_ratings(np.ones((2, 3)), TASK_LIMITS)      # 2 эксперта
        with self.assertRaises(InputError):
            validate_ratings(np.ones((3, 6)), TASK_LIMITS)      # 6 альтернатив
        with self.assertRaises(InputError):
            validate_ratings(np.ones((21, 3)), TASK_LIMITS)     # 21 эксперт
        validate_ratings(np.ones((50, 1000)), Limits())         # без ограничений сверху

    def test_nan_rejected(self):
        with self.assertRaises(InputError):
            validate_ratings([[1, float("nan")], [1, 2]])

    def test_parse_rankings_and_weak_orders(self):
        members, items, r = parse_rankings("2: A > B > C\nC > A = B")
        self.assertEqual(items, ["A", "B", "C"])
        self.assertEqual(len(members), 3)
        np.testing.assert_allclose(r[0], [2, 1, 0])
        np.testing.assert_allclose(r[2], [0, 0, 2])   # C лучше, A и B равны (по 0 строго ниже)

    def test_parse_rankings_matches_old_tool_example(self):
        _, _, r = parse_rankings("4: A > B > C > D\n3: B > C > D > A\n2: C > B > A > D")
        np.testing.assert_allclose(get_rule("copeland").scores(r), [-1, 3, 1, -3])

    def test_parse_rankings_errors(self):
        with self.assertRaises(InputError):
            parse_rankings("A>B>C\nA>B>D")
        with self.assertRaises(InputError):
            parse_rankings("A>B>B")
        with self.assertRaises(InputError):
            parse_rankings("# пусто")

    def test_csv_semicolon_decimal_comma(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("x;a;b;c\nu1;1,5;2;3\nu2;3;2,5;1\n")
        try:
            members, items, r = load_ratings_csv(f.name)
        finally:
            os.unlink(f.name)
        self.assertEqual(items, ["a", "b", "c"])
        np.testing.assert_allclose(r[0], [1.5, 2, 3])

    def test_top_n_union(self):
        r = np.array([[5, 4, 1, 0], [0, 1, 4, 5.0]])
        self.assertEqual(top_n_union(r, 1).tolist(), [0, 3])
        self.assertEqual(top_n_union(r, 2).tolist(), [0, 1, 2, 3])


class CompareTests(unittest.TestCase):
    def test_compare_smoke_and_prefilter(self):
        r = make_group_ratings(6, 40, 0.5, rng=1)
        res = compare_rules(r, k=5, tiebreak="borda")
        self.assertEqual(set(res["rules"]), set(available_rules()))
        for v in res["rules"].values():
            self.assertEqual(len(v["top"]), 5)
        res2 = compare_rules(r, k=5, prefilter_top_n=5)
        self.assertLessEqual(res2["n_candidates"], 30)
        for v in res2["rules"].values():
            self.assertTrue(all(0 <= i < 40 for i in v["top"]))

    def test_least_misery_protects_weakest(self):
        # элемент 0: двое в восторге, третий ненавидит; элемент 1: всем средне
        r = np.array([[5, 3], [5, 3], [1, 3.0]])
        res = compare_rules(r, rule_names=["average", "least_misery"], k=1)
        self.assertEqual(res["rules"]["average"]["top"], [0])
        self.assertEqual(res["rules"]["least_misery"]["top"], [1])

    def test_synthetic_cohesion_is_monotonic(self):
        means = []
        for c in (0.0, 0.5, 1.0):
            vals = [group_cohesion(make_group_ratings(6, 60, c, rng=s)) for s in range(30)]
            means.append(np.mean(vals))
        self.assertLess(means[0], means[1])
        self.assertLess(means[1], means[2])
        self.assertLess(abs(means[0]), 0.1)

    def test_reproducible(self):
        a = make_group_ratings(5, 20, 0.4, rng=7)
        b = make_group_ratings(5, 20, 0.4, rng=7)
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()