import unittest

from analyze_budget_archive_20261026 import choose


class BudgetChoiceTests(unittest.TestCase):
    def trials(self):
        return [{"lambda": strength, "C": c, "geometry": g, "tune_cost": 10,
                 "unavailable_test_cost": -1000 if strength != .5 else 1000}
                for strength in (.0, .5, 1.) for c in (.01, 1., 100.) for g in range(3)]

    def test_nine_candidate_restriction_and_ties(self):
        selected = choose(self.trials(), .5)
        self.assertEqual((selected["lambda"], selected["C"], selected["geometry"]), (.5, .01, 0))

    def test_validation_cost_drives_selection(self):
        rows = self.trials()
        wanted = next(r for r in rows if r["lambda"] == .5 and r["C"] == 100 and r["geometry"] == 2)
        wanted["tune_cost"] = 9
        self.assertIs(choose(rows, .5), wanted)

    def test_missing_candidate_rejected(self):
        rows = self.trials()
        rows.remove(next(r for r in rows if r["lambda"] == .5))
        with self.assertRaises(AssertionError):
            choose(rows, .5)

    def test_baseline_same_grid(self):
        self.assertEqual(choose(self.trials(), .0)["lambda"], .0)


if __name__ == "__main__":
    unittest.main()
