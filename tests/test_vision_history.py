import unittest

import numpy as np

from lab1_vision import causal_visual


class VisualHistoryTests(unittest.TestCase):
    def values(self):
        random = np.random.default_rng(52)
        vectors = random.normal(size=(8, 16)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors, np.arange(8, dtype=float), np.ones(8)

    def test_current_and_future_reactions_do_not_change_features(self):
        vectors, reactions, levels = self.values()
        changed = reactions.copy()
        changed[4:] = 50
        left = causal_visual(vectors, reactions, levels)
        right = causal_visual(vectors, changed, levels)
        np.testing.assert_allclose(left[:5], right[:5], equal_nan=True)
        self.assertFalse(np.array_equal(left[5:], right[5:]))

    def test_future_images_do_not_change_past_features(self):
        vectors, reactions, levels = self.values()
        changed = vectors.copy()
        changed[5:] = -changed[5:]
        np.testing.assert_allclose(
            causal_visual(vectors, reactions, levels)[:5],
            causal_visual(changed, reactions, levels)[:5],
            equal_nan=True,
        )


if __name__ == "__main__":
    unittest.main()
