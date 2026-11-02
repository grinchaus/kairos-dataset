from __future__ import annotations

import numpy as np
import pandas as pd
import hashlib
import os
from pathlib import Path

import joblib

from kairos.data import load_posts


def history_names():
    names = []
    for count in [1, 3, 5, 10]:
        names.extend(
            f"semantic_nearest{count}_{name}"
            for name in ["level_delta", "spread", "similarity"]
        )
    for temperature in [3, 8, 20]:
        names.extend(
            f"semantic_soft{temperature}_{name}"
            for name in ["level_delta", "spread", "below", "above", "effective_posts"]
        )
    return names


def causal_features(vectors, log_reactions, levels):
    output = np.full((len(vectors), len(history_names())), np.nan, dtype=np.float32)
    similarity = vectors @ vectors.T
    for row in range(1, len(vectors)):
        values = log_reactions[:row]
        sims = similarity[row, :row]
        level = levels[row]
        order = np.argsort(-sims, kind="stable")
        features = []
        for count in [1, 3, 5, 10]:
            positions = order[:count]
            features.extend(
                [
                    values[positions].mean() - level,
                    values[positions].std(),
                    sims[positions].mean(),
                ]
            )
        for temperature in [0.03, 0.08, 0.2]:
            weights = np.exp((sims - sims.max()) / temperature)
            weights /= weights.sum()
            mean = weights @ values
            features.extend(
                [
                    mean - level,
                    np.sqrt(weights @ ((values - mean) ** 2)),
                    weights @ (values - level < -0.5493061443340549),
                    weights @ (values - level > 1.157452788691043),
                    1 / (weights @ weights),
                ]
            )
        output[row] = features
    return output


def semantic_history(table, store, data):
    raw = load_posts(data)
    keys = pd.concat(
        [
            store.post_index.to_frame(index=False),
            store.history_keys[["author", "post"]],
        ],
        ignore_index=True,
    )
    timeline = keys.merge(
        raw[["author", "post", "posted_at", "reactions"]],
        on=["author", "post"],
        validate="one_to_one",
    )
    timeline = timeline.merge(
        table[["author", "post", "level", "hist_n"]],
        on=["author", "post"],
        how="left",
        validate="one_to_one",
    )
    timeline["order"] = timeline.hist_n
    timeline.loc[len(store.posts) :, "order"] = (
        store.history_keys.history_order.to_numpy()
    )
    vectors = np.vstack([store.posts, store.history])
    values = np.log1p(timeline.reactions.to_numpy(dtype=float))
    result = np.full((len(timeline), len(history_names())), np.nan, dtype=np.float32)
    ordered = timeline.sort_values(["author", "posted_at", "order"], kind="mergesort")
    for _, positions in ordered.groupby("author", sort=False).groups.items():
        positions = np.asarray(positions, dtype=int)
        result[positions] = causal_features(
            vectors[positions], values[positions], timeline.level.to_numpy()[positions]
        )
    feature_frame = pd.concat(
        [keys, pd.DataFrame(result, columns=history_names())], axis=1
    )
    return table.merge(
        feature_frame, on=["author", "post"], how="left", validate="one_to_one"
    )


def enrich_semantic_table(table, store, data):
    digest = hashlib.sha256(Path(__file__).read_bytes())
    digest.update(store.info["input_sha256"].encode())
    digest.update(store.info["history_input_sha256"].encode())
    digest.update(
        pd.util.hash_pandas_object(
            table[["author", "post", "log_reactions", "level", "hist_n"]], index=False
        )
        .to_numpy()
        .tobytes()
    )
    for path in sorted((Path(data) / "posts").glob("*.json")):
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    cache = (
        Path(__file__).parent / ".cache" / f"semantic-{digest.hexdigest()[:20]}.joblib"
    )
    if cache.is_file():
        extra = joblib.load(cache)
    else:
        extra = semantic_history(table, store, data)[
            ["author", "post"] + history_names()
        ]
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_name(cache.name + f".{os.getpid()}.tmp")
        joblib.dump(extra, temporary, compress=3)
        temporary.replace(cache)
    return table.merge(extra, on=["author", "post"], how="left", validate="one_to_one")
