from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def entities(row):
    mentions = {
        "mention:" + code
        for code in str(row.graph_text).split()
        if code != "NO_MENTIONS"
    }
    if row.profile_company != "unknown":
        mentions.add("employer:" + str(row.profile_company))
    return mentions


class GraphEncoder:
    def fit(self, table):
        raw = defaultdict(lambda: np.zeros(5))
        for row in table.itertuples(index=False):
            for entity in entities(row):
                values = raw[(row.author, entity)]
                values[int(row.label)] += 1
                values[3] += row.resid
                values[4] += 1
        self.author = {
            key: np.r_[value[:4] / value[4], 1.0] for key, value in raw.items()
        }
        totals = defaultdict(lambda: np.zeros(5))
        for (_, entity), values in self.author.items():
            totals[entity] += values
        self.totals = dict(totals)
        return self

    def transform(self, table):
        rows = []
        for row in table.itertuples(index=False):
            values = []
            for entity in entities(row):
                total = self.totals.get(entity, np.zeros(5)) - self.author.get(
                    (row.author, entity), np.zeros(5)
                )
                values.append(total)
            if not values:
                values = [np.zeros(5)]
            values = np.asarray(values)
            count = values[:, 4]
            result = {
                "network_other_authors_mean": float(np.log1p(count).mean()),
                "network_other_authors_max": float(np.log1p(count).max()),
                "network_shared_fraction": float(np.mean(count > 0)),
            }
            for strength in [3, 10]:
                posterior = (
                    values[:, :4] + strength * np.array([0.2, 0.6, 0.2, 0])
                ) / (count[:, None] + strength)
                for index, name in enumerate(["below", "typical", "above", "residual"]):
                    for aggregation, function in [
                        ("mean", np.mean),
                        ("min", np.min),
                        ("max", np.max),
                    ]:
                        result[f"network_s{strength}_{name}_{aggregation}"] = float(
                            function(posterior[:, index])
                        )
            rows.append(result)
        return pd.DataFrame(rows, index=table.index)
