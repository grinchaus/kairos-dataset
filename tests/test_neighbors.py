from __future__ import annotations

import unittest

import numpy as np

try:
    from lab1_neighbors import neighbor_features
except ImportError:
    neighbor_features = None


@unittest.skipIf(
    neighbor_features is None, "Optional torch dependency is not installed"
)
class NeighborTests(unittest.TestCase):
    def test_own_author_targets_are_excluded(self):
        vectors = np.eye(3, dtype=np.float32)
        authors = np.array(["A", "A", "B"])
        first = neighbor_features(
            vectors[:2], authors[:2], vectors, authors, np.array([0, 1, 2])
        )
        second = neighbor_features(
            vectors[:2], authors[:2], vectors, authors, np.array([2, 0, 2])
        )
        np.testing.assert_allclose(first, second, equal_nan=True)
        np.testing.assert_allclose(first[:, :3], np.tile([0.0, 0.0, 1.0], (2, 1)))


if __name__ == "__main__":
    unittest.main()
