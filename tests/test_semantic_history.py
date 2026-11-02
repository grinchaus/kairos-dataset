from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from lab1_semantic_history import causal_features, semantic_history


class SemanticHistoryTests(unittest.TestCase):
    def test_current_and_future_outcomes_do_not_affect_current_features(self):
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(12, 8))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        reactions = rng.normal(size=12)
        levels = np.zeros(12)
        original = causal_features(vectors, reactions, levels)
        reactions[7:] += 100
        vectors[8:] *= -1
        changed = causal_features(vectors, reactions, levels)
        np.testing.assert_allclose(original[:8], changed[:8], equal_nan=True)
        self.assertFalse(np.allclose(original[8:], changed[8:], equal_nan=True))

    def test_nearest_post_uses_earlier_reactions(self):
        vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        result = causal_features(
            vectors, np.array([2.0, 8.0, 100.0]), np.array([0.0, 0.0, 1.0])
        )
        self.assertEqual(result[2, 0], 1)
        self.assertEqual(result[2, 2], 1)

    def test_initial_posts_and_output_key_order(self):
        raw = pd.DataFrame(
            {
                "author": ["A"] * 7,
                "post": [f"P{i}" for i in range(1, 8)],
                "posted_at": pd.date_range("2025-01-01", periods=7, tz="UTC"),
                "reactions": [1, 2, 3, 4, 5, 999, 99999],
            }
        )
        table = raw.iloc[[6, 5]][["author", "post", "posted_at"]].reset_index(drop=True)
        table["level"] = [2.0, 1.0]
        table["hist_n"] = [6, 5]
        history = raw.iloc[:5][["author", "post"]].copy()
        history["history_order"] = range(5)
        store = SimpleNamespace(
            post_index=pd.MultiIndex.from_frame(table[["author", "post"]]),
            posts=np.tile([1.0, 0.0], (2, 1)),
            history=np.tile([1.0, 0.0], (5, 1)),
            history_keys=history,
        )
        with patch("lab1_semantic_history.load_posts", return_value=raw):
            result = semantic_history(table, store, "unused")
        self.assertEqual(result.post.tolist(), ["P7", "P6"])
        self.assertAlmostEqual(
            result.loc[1, "semantic_nearest10_level_delta"],
            np.log1p([1, 2, 3, 4, 5]).mean() - 1,
            places=6,
        )
        self.assertAlmostEqual(
            result.loc[0, "semantic_nearest10_level_delta"],
            np.log1p([1, 2, 3, 4, 5, 999]).mean() - 2,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
