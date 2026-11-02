from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import ndtr

from kairos.data import load_posts


def kernel_names():
    return [
        f"kernel_t{temperature}_n{noise}_{feature}"
        for temperature in [3, 10, 30]
        for noise in [10, 50, 200]
        for feature in ["delta", "uncertainty", "below", "above"]
    ]


def causal_kernel(vectors, reactions, levels):
    result = np.full((len(vectors), len(kernel_names())), np.nan, dtype=np.float32)
    distance = np.maximum(0, 1 - vectors @ vectors.T)
    column = 0
    for temperature in [0.03, 0.1, 0.3]:
        kernel = np.exp(-distance / temperature).astype(np.float64)
        for noise in [0.1, 0.5, 2.0]:
            inverse = np.empty((0, 0), dtype=np.float64)
            for row in range(len(vectors)):
                cross = kernel[row, :row]
                if row:
                    projection = inverse @ cross
                    mean = projection @ (reactions[:row] - levels[row])
                    variance = max(1e-8, 1 - cross @ projection)
                    scale = max(0.3, float(np.std(reactions[:row]))) * np.sqrt(
                        variance + noise
                    )
                    result[row, column : column + 4] = [
                        mean,
                        scale,
                        ndtr((-0.5493061443340549 - mean) / scale),
                        1 - ndtr((1.157452788691043 - mean) / scale),
                    ]
                else:
                    projection = np.empty(0)
                denominator = max(1e-8, 1 + noise - cross @ projection)
                expanded = np.empty((row + 1, row + 1))
                expanded[:row, :row] = (
                    inverse + np.outer(projection, projection) / denominator
                )
                expanded[:row, row] = -projection / denominator
                expanded[row, :row] = -projection / denominator
                expanded[row, row] = 1 / denominator
                inverse = expanded
            column += 4
    return result


def enrich_kernel(table, store, data):
    digest = hashlib.sha256(Path(__file__).read_bytes())
    digest.update(store.info["revision"].encode())
    digest.update(store.info["input_sha256"].encode())
    digest.update(store.info["history_input_sha256"].encode())
    digest.update(
        pd.util.hash_pandas_object(
            table[["author", "post", "level", "hist_n"]], index=False
        )
        .to_numpy()
        .tobytes()
    )
    for path in sorted((Path(data) / "posts").glob("*.json")):
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    cache = (
        Path(__file__).parent / ".cache" / f"kernel-{digest.hexdigest()[:20]}.joblib"
    )
    if cache.exists():
        extra = joblib.load(cache)
    else:
        keys = pd.concat(
            [
                store.post_index.to_frame(index=False),
                store.history_keys[["author", "post"]],
            ],
            ignore_index=True,
        )
        raw = load_posts(data)
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
        reactions = np.log1p(timeline.reactions.to_numpy(dtype=float))
        result = np.full((len(timeline), len(kernel_names())), np.nan, dtype=np.float32)
        ordered = timeline.sort_values(
            ["author", "posted_at", "order"], kind="mergesort"
        )
        for _, positions in ordered.groupby("author", sort=False).groups.items():
            positions = np.asarray(positions, dtype=int)
            result[positions] = causal_kernel(
                vectors[positions],
                reactions[positions],
                timeline.level.to_numpy()[positions],
            )
        extra = pd.concat([keys, pd.DataFrame(result, columns=kernel_names())], axis=1)
        joblib.dump(extra, cache, compress=3)
    return table.merge(extra, on=["author", "post"], how="left", validate="one_to_one")
