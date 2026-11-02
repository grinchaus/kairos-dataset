import unittest

import numpy as np
import pandas as pd

from lab1_graph import GraphEncoder


class GraphEncoderTests(unittest.TestCase):
    def table(self):
        return pd.DataFrame(
            {
                "author": ["a", "a", "b", "c"],
                "graph_text": ["ORG_1"] * 4,
                "profile_company": ["unknown"] * 4,
                "label": [0, 0, 2, 1],
                "resid": [-1.0, -1.0, 2.0, 0.0],
            }
        )

    def test_own_author_targets_are_excluded(self):
        table = self.table()
        changed = table.copy()
        changed.loc[changed.author == "a", ["label", "resid"]] = [2, 100.0]
        left = GraphEncoder().fit(table).transform(table.iloc[:2])
        right = GraphEncoder().fit(changed).transform(table.iloc[:2])
        np.testing.assert_allclose(left, right)

    def test_validation_does_not_read_target_columns(self):
        table = self.table()
        encoder = GraphEncoder().fit(table.iloc[:3])
        keys = table.iloc[3:][["author", "graph_text", "profile_company"]]
        values = encoder.transform(keys)
        self.assertGreater(values.iloc[0].network_other_authors_max, 0)
        self.assertTrue(np.isfinite(values.to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
