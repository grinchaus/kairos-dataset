from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from lab1_neural import DenseClassifier, context_vectors
from lab1_streaming import context_memmap, fit_streamed, probability


class Store:
    def __init__(self):
        random = np.random.default_rng(12)
        self.posts = random.normal(size=(4, 4)).astype(np.float32)
        self.profiles = random.normal(size=(2, 4)).astype(np.float32)
        self.history = random.normal(size=(4, 4)).astype(np.float32)
        self.history_keys = pd.DataFrame(
            {
                "author": ["a", "a", "b", "b"],
                "posted_at": pd.to_datetime(["2020-01-01", "2020-01-02"] * 2, utc=True),
                "history_order": [0, 1, 0, 1],
            }
        )

    def arrays(self, table):
        return (
            self.posts[table.post.to_numpy(dtype=int)],
            self.profiles[(table.author == "b").to_numpy(dtype=int)],
        )


class StreamingTests(unittest.TestCase):
    def test_streamed_context_matches_existing_causal_features_exactly(self):
        store = Store()
        table = pd.DataFrame(
            {
                "author": ["a", "a", "b", "b"],
                "post": range(4),
                "posted_at": pd.to_datetime(["2020-01-03", "2020-01-04"] * 2, utc=True),
                "hist_n": [2, 3, 2, 3],
            }
        )
        expected = context_vectors(table, store)
        with tempfile.TemporaryDirectory() as directory:
            actual = context_memmap(table, store, Path(directory))
            np.testing.assert_array_equal(actual, expected)

    def test_minibatch_training_learns_and_reload_preserves_predictions(self):
        labels = np.arange(240) % 3
        inputs = np.zeros((240, 12), dtype=np.float32)
        inputs[np.arange(240), labels] = 1
        model, predicted, _ = fit_streamed(
            inputs[:180],
            labels[:180],
            "cpu",
            7,
            40,
            validation=(inputs[180:], labels[180:]),
        )
        self.assertGreater((predicted.argmax(axis=1) == labels[180:]).mean(), 0.9)
        restored = DenseClassifier(12)
        restored.load_state_dict(model.state_dict())
        np.testing.assert_array_equal(predicted, probability(restored, inputs[180:]))


if __name__ == "__main__":
    unittest.main()
