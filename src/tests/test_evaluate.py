import unittest

from trainers.evaluate import wilson_interval


class TestEvaluate(unittest.TestCase):
    def test_wilson_interval_contains_half_for_balanced_data(self):
        lo, hi = wilson_interval(5, 10)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)

    def test_invalid_wilson_counts_rejected(self):
        with self.assertRaises(ValueError):
            wilson_interval(2, 0)

