import unittest

import numpy as np

from lab1_kernel_history import causal_kernel


class KernelHistoryTests(unittest.TestCase):
    def test_online_mean_matches_direct_solution(self):
        random = np.random.default_rng(7)
        vectors = random.normal(size=(9, 12)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        reactions = np.arange(9, dtype=float) / 2
        levels = np.ones(9)
        actual = causal_kernel(vectors, reactions, levels)
        kernel = np.exp(-np.maximum(0, 1 - vectors @ vectors.T) / 0.03).astype(float)
        row = 7
        expected = kernel[row, :row] @ np.linalg.solve(
            kernel[:row, :row] + 0.1 * np.eye(row), reactions[:row] - levels[row]
        )
        self.assertAlmostEqual(actual[row, 0], expected, places=5)

    def test_current_and_future_reactions_are_excluded(self):
        random = np.random.default_rng(8)
        vectors = random.normal(size=(8, 12)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        reactions = np.arange(8, dtype=float)
        changed = reactions.copy()
        changed[5:] = 50
        left = causal_kernel(vectors, reactions, np.ones(8))
        right = causal_kernel(vectors, changed, np.ones(8))
        np.testing.assert_allclose(left[:6], right[:6], equal_nan=True)
        self.assertFalse(np.array_equal(left[6:], right[6:]))


if __name__ == "__main__":
    unittest.main()
