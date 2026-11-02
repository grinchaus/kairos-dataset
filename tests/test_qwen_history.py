from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiment_qwen import qwen_table
from lab1_embeddings import EmbeddingStore
import lab1_semantic_history as semantic


class QwenHistoryTests(unittest.TestCase):
    def test_same_texts_from_different_encoders_do_not_share_semantic_cache(self):
        raw = pd.DataFrame(
            {
                "author": ["A"] * 4,
                "post": ["P1", "P2", "P3", "P4"],
                "posted_at": pd.date_range("2025-01-01", periods=4, tz="UTC"),
                "reactions": [1, 100, 5, 20],
            }
        )
        table = raw.iloc[2:][["author", "post", "posted_at"]].reset_index(drop=True)
        table["hist_n"] = [2, 3]
        table["level"] = 1.0
        table["log_reactions"] = np.log1p(raw.reactions.to_numpy()[2:])
        stores = []
        for revision, history in [
            ("first", [[1.0, 0.0], [0.0, 1.0]]),
            ("second", [[0.0, 1.0], [1.0, 0.0]]),
        ]:
            store = object.__new__(EmbeddingStore)
            store.info = {
                "model": "encoder",
                "revision": revision,
                "input_sha256": "identical-texts",
                "history_input_sha256": "identical-history-texts",
            }
            store.post_index = pd.MultiIndex.from_frame(table[["author", "post"]])
            store.profile_index = pd.Index(["A"])
            store.posts = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            store.profiles = np.array([[1.0, 0.0]], dtype=np.float32)
            store.history = np.asarray(history, dtype=np.float32)
            store.history_keys = raw.iloc[:2][["author", "post", "posted_at"]].copy()
            store.history_keys["history_order"] = [0, 1]
            stores.append(store)
        original_info = dict(stores[1].info)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "semantic.py"
            source.write_bytes(Path(semantic.__file__).read_bytes())
            with patch.object(semantic, "__file__", str(source)), patch.object(
                semantic, "load_posts", return_value=raw
            ):
                previous = semantic.enrich_semantic_table(table, stores[0], folder)
                actual, _ = qwen_table(table, stores[1], folder)
                cached, _ = qwen_table(table, stores[1], folder)
                expected = semantic.semantic_history(table, stores[1], folder)
        name = "semantic_nearest1_level_delta"
        np.testing.assert_array_equal(actual["qwen_" + name], expected[name])
        np.testing.assert_array_equal(cached["qwen_" + name], expected[name])
        self.assertNotEqual(previous.loc[0, name], expected.loc[0, name])
        self.assertEqual(stores[1].info, original_info)


if __name__ == "__main__":
    unittest.main()
