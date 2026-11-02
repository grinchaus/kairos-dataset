from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from kairos.data import load_posts
from lab1_embeddings import cosine
from lab1_neural import context_vectors
from lab1_semantic_history import causal_features, history_names


def visual_history_names():
    return ["vision_" + name for name in history_names()] + [
        "vision_previous_similarity",
        "vision_recent5_similarity",
        "vision_history_similarity",
        "vision_history_images",
    ]


def causal_visual(vectors, reactions, levels):
    semantic = causal_features(vectors, reactions, levels)
    context = np.full((len(vectors), 4), np.nan, dtype=np.float32)
    context[:, 3] = np.arange(len(vectors))
    for row in range(1, len(vectors)):
        current = vectors[row : row + 1]
        context[row] = [
            cosine(current, vectors[row - 1 : row])[0],
            cosine(current, vectors[max(0, row - 5) : row].mean(axis=0, keepdims=True))[
                0
            ],
            cosine(current, vectors[:row].mean(axis=0, keepdims=True))[0],
            row,
        ]
    return np.hstack([semantic, context])


def multimodal_vectors(table, qwen, vision):
    images, captions, _ = vision.arrays(table)
    return np.hstack(
        [context_vectors(table, qwen), images, captions, images * captions]
    ).astype(np.float32)


class VisionStore:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.info = json.loads((self.folder / "encoder.json").read_text())
        self.keys = pd.read_csv(self.folder / "keys.csv")
        self.keys["posted_at"] = pd.to_datetime(
            self.keys.posted_at, utc=True, format="mixed"
        )
        self.index = pd.MultiIndex.from_frame(self.keys[["author", "post"]])
        files = pd.read_csv(self.folder / "files.csv")
        self.file_index = pd.Index(files.file)
        self.rows = self.file_index.get_indexer(self.keys.file)
        self.images = np.load(self.folder / "images.npy", mmap_mode="r")
        self.captions = np.load(self.folder / "captions.npy", mmap_mode="r")
        self.genres = np.load(self.folder / "genres.npy")
        if (
            self.index.has_duplicates
            or self.file_index.has_duplicates
            or (self.rows < 0).any()
        ):
            raise ValueError("Invalid image embedding keys")
        if self.images.shape != (len(files), 512) or self.captions.shape != (
            len(self.keys),
            512,
        ):
            raise ValueError("Image vectors do not match their keys")

    def arrays(self, table):
        positions = self.index.get_indexer(
            pd.MultiIndex.from_frame(table[["author", "post"]])
        )
        present = positions >= 0
        images = np.zeros((len(table), 512), dtype=np.float32)
        captions = np.zeros_like(images)
        images[present] = self.images[self.rows[positions[present]]]
        captions[present] = self.captions[positions[present]]
        return images, captions, present


class VisionProjection:
    def fit(self, table, store):
        images, _, present = store.arrays(table)
        self.pca = PCA(n_components=32, random_state=42).fit(images[present])
        return self

    def transform(self, table, store):
        images, _, present = store.arrays(table)
        values = self.pca.transform(images)
        values[~present] = 0
        names = [f"vision_pc{index:03d}" for index in range(values.shape[1])]
        return (
            pd.concat(
                [table.reset_index(drop=True), pd.DataFrame(values, columns=names)],
                axis=1,
            ),
            names,
        )


def enrich_vision(table, store, data):
    digest = hashlib.sha256(Path(__file__).read_bytes())
    digest.update(Path(__file__).with_name("lab1_semantic_history.py").read_bytes())
    digest.update(store.info["revision"].encode())
    digest.update(store.info["input_sha256"].encode())
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
        Path(__file__).parent
        / ".cache"
        / f"vision-features-{digest.hexdigest()[:20]}.joblib"
    )
    if cache.exists():
        extra = joblib.load(cache)
    else:
        images, captions, present = store.arrays(table)
        values = {
            "vision_present": present.astype(float),
            "vision_caption_alignment": np.where(
                present, np.einsum("ij,ij->i", images, captions), np.nan
            ),
        }
        genre_values = images @ store.genres.T
        for index in range(genre_values.shape[1]):
            values[f"vision_genre{index:02d}"] = np.where(
                present, genre_values[:, index], np.nan
            )
        extra = pd.concat(
            [table[["author", "post"]].reset_index(drop=True), pd.DataFrame(values)],
            axis=1,
        )
        keys = store.keys[store.keys.author.isin(table.author)].reset_index(drop=True)
        raw = load_posts(data)
        timeline = keys.merge(
            raw[["author", "post", "reactions"]],
            on=["author", "post"],
            validate="one_to_one",
        )
        timeline = timeline.merge(
            table[["author", "post", "level"]],
            on=["author", "post"],
            how="left",
            validate="one_to_one",
        )
        vectors = store.images[store.file_index.get_indexer(timeline.file)]
        reactions = np.log1p(timeline.reactions.to_numpy(dtype=float))
        history = np.full(
            (len(timeline), len(visual_history_names())), np.nan, dtype=np.float32
        )
        ordered = timeline.sort_values(
            ["author", "posted_at", "history_order"], kind="mergesort"
        )
        for _, positions in ordered.groupby("author", sort=False).groups.items():
            positions = np.asarray(positions, dtype=int)
            history[positions] = causal_visual(
                vectors[positions],
                reactions[positions],
                timeline.level.to_numpy()[positions],
            )
        historical = pd.concat(
            [
                timeline[["author", "post"]],
                pd.DataFrame(history, columns=visual_history_names()),
            ],
            axis=1,
        )
        extra = extra.merge(
            historical, on=["author", "post"], how="left", validate="one_to_one"
        )
        joblib.dump(extra, cache, compress=3)
    names = [name for name in extra if name not in ["author", "post"]]
    return (
        table.merge(extra, on=["author", "post"], how="left", validate="one_to_one"),
        names,
    )
