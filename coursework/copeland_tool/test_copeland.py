"""Тесты: python -m unittest -v"""
import unittest

from copeland import ProfileError, copeland, parse_profile


def run(text, **kw):
    return copeland(parse_profile(text), **kw)


class CopelandTests(unittest.TestCase):
    def test_condorcet_winner(self):
        res = run("4: A > B > C > D\n3: B > C > D > A\n2: C > B > A > D")
        self.assertEqual(res.winners, ["B"])
        self.assertEqual(res.condorcet_winner, "B")
        self.assertEqual(res.condorcet_loser, "D")
        self.assertEqual(res.scores, {"A": -1, "B": 3, "C": 1, "D": -3})
        self.assertEqual(res.ranking, [["B"], ["C"], ["A"], ["D"]])

    def test_condorcet_cycle_is_tie(self):
        res = run("A>B>C\nB>C>A\nC>A>B")
        self.assertEqual(sorted(res.winners), ["A", "B", "C"])
        self.assertIsNone(res.condorcet_winner)

    def test_tiebreak_borda(self):
        # Цикл Кондорсе: у всех очки Копленда 0, очки Борда: A=11, B=10, C=9
        text = "4: A>B>C\n3: B>C>A\n3: C>A>B"
        self.assertEqual(sorted(run(text).winners), ["A", "B", "C"])
        res = run(text, tiebreak="borda")
        self.assertEqual(res.ranking, [["A"], ["B"], ["C"]])
        self.assertEqual(res.borda, {"A": 11, "B": 10, "C": 9})

    def test_pairwise_tie(self):
        res = run("A > B > C\nA > B > C\nB > A > C\nB > A > C")
        self.assertEqual(res.pairwise["A"]["B"], 2)
        self.assertEqual(res.ties["A"], 1)

    def test_pairwise_matrix_sums(self):
        res = run("3: A>B>C\n2: C>B>A\n1: B>A>C")
        for a in res.alternatives:
            for b in res.alternatives:
                if a != b:
                    self.assertEqual(res.pairwise[a][b] + res.pairwise[b][a], 6)

    def test_limits(self):
        with self.assertRaises(ProfileError):
            run("A>B\nB>A\nA>B")                      # 2 альтернативы
        with self.assertRaises(ProfileError):
            run("A>B>C>D>E>F\n" * 3)                  # 6 альтернатив
        with self.assertRaises(ProfileError):
            run("A>B>C\nA>B>C")                       # 2 эксперта
        with self.assertRaises(ProfileError):
            run("21: A>B>C")                          # 21 эксперт
        self.assertEqual(run("20: A>B>C").winners, ["A"])   # граница 20 допустима

    def test_inconsistent_rankings(self):
        with self.assertRaises(ProfileError):
            run("A>B>C\nA>B>D\nA>B>C")                # разные наборы альтернатив
        with self.assertRaises(ProfileError):
            run("A>B>B\nA>B>C\nA>B>C")                # повтор в ранжировании


if __name__ == "__main__":
    unittest.main()