import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from paired_bootstrap import oriented_difference, paired_bootstrap  # noqa: E402


class PairedBootstrapTest(unittest.TestCase):
    def test_oriented_difference_is_positive_when_reference_is_better(self):
        self.assertAlmostEqual(oriented_difference("pearson", 0.8, 0.6), 0.2)
        self.assertAlmostEqual(oriented_difference("mae", 0.1, 0.2), 0.1)

    def test_fixed_seed_reproduces_intervals(self):
        labels = np.asarray([0.0, 0.2, 0.4, 0.8, 1.0])
        predictions = {
            "comparator": np.asarray([0.1, 0.3, 0.5, 0.6, 0.7]),
            "siRBench": np.asarray([0.0, 0.2, 0.5, 0.8, 0.9]),
        }

        first = paired_bootstrap(labels, predictions, samples=100, seed=7)
        second = paired_bootstrap(labels, predictions, samples=100, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)


if __name__ == "__main__":
    unittest.main()
