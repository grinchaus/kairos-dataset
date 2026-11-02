from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiment_lab1 import experiment_table, reference_probabilities
from experiment_target70 import previous_probability, rich_inputs
from kairos.metrics import score
from lab1_embeddings import CONTEXT_NUM, EmbeddingProjection, EmbeddingStore
from lab1_features import CATEGORICAL
from lab1_rich import RICH_CAT, rich_table
from lab1_semantic_history import enrich_semantic_table, history_names
from train_lab1 import blend, select_blend


def qwen_table(table, store, data):
    keys = table[["author", "post", "posted_at", "level", "hist_n", "log_reactions"]]
    semantic_store = copy.copy(store)
    configuration = {
        key: store.info.get(key)
        for key in [
            "model",
            "revision",
            "prefix",
            "max_length",
            "normalized",
            "input_sha256",
        ]
    }
    semantic_store.info = {
        **store.info,
        "input_sha256": hashlib.sha256(
            json.dumps(configuration, sort_keys=True).encode()
        ).hexdigest(),
    }
    semantic = enrich_semantic_table(keys, semantic_store, data)
    semantic = store.context(semantic)
    names = history_names() + CONTEXT_NUM
    extra = semantic[names].rename(columns={name: "qwen_" + name for name in names})
    return pd.concat(
        [table.reset_index(drop=True), extra.reset_index(drop=True)], axis=1
    ), list(extra)


def qwen_project(projection, table, store):
    projected, names = projection.transform(table[["author", "post"]], store)
    extra = projected[names].rename(
        columns={name: name.replace("e5_", "qwen_") for name in names}
    )
    return pd.concat(
        [table.reset_index(drop=True), extra.reset_index(drop=True)], axis=1
    ), list(extra)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 137])
    parser.add_argument("--depth", type=int, default=6)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        train, numeric, e5 = experiment_table(Path("data"), Path(".cache/embeddings"))
        train, rich_numeric = rich_table(train, Path("data"))
        qwen = EmbeddingStore(Path(".cache/embeddings-qwen"))
        train, qwen_numeric = qwen_table(train, qwen, Path("data"))
        for seed in args.seeds:
            folder = Path("results/target70") / str(seed)
            folder.mkdir(parents=True, exist_ok=True)
            _, _, fit, val, tree_numeric, _, text_probability = reference_probabilities(
                train, numeric, e5, seed, Path("results/improvement") / str(seed)
            )
            projection = EmbeddingProjection(96, 24).fit(fit, qwen)
            fit, projected = qwen_project(projection, fit, qwen)
            val, _ = qwen_project(projection, val, qwen)
            features = (
                tree_numeric + history_names() + rich_numeric + qwen_numeric + projected
            )
            model = CatBoostClassifier(
                iterations=4000,
                depth=args.depth,
                learning_rate=0.04,
                l2_leaf_reg=8,
                loss_function="MultiClass",
                auto_class_weights="Balanced",
                random_seed=42,
                thread_count=2,
                allow_writing_files=False,
            )
            model.fit(
                rich_inputs(fit, features),
                fit.label,
                cat_features=CATEGORICAL + RICH_CAT,
                eval_set=(rich_inputs(val, features), val.label),
                early_stopping_rounds=180,
                use_best_model=True,
                verbose=500,
            )
            probability = model.predict_proba(
                rich_inputs(val, features), thread_count=2
            )
            name = f"qwen-depth{args.depth}"
            model.save_model(str(folder / f"{name}.cbm"))
            np.savez_compressed(
                folder / f"{name}.npz",
                probability=probability,
                labels=val.label.to_numpy(),
                posts=val.post.to_numpy(dtype=str),
                authors=val.author.to_numpy(dtype=str),
            )
            selection = select_blend(val.label, probability, text_probability)
            v2 = previous_probability(seed)
            result = {
                "iterations": model.tree_count_,
                "features": features,
                "standalone": score(val.label, probability.argmax(axis=1)),
                "text_blend": score(
                    val.label, blend(probability, text_probability, selection)
                ),
                "text_selection": selection,
                "v2_blends": {
                    str(weight): score(
                        val.label,
                        np.argmax(
                            ((1 - weight) * v2 + weight * probability) * [0.75, 1, 0.6],
                            axis=1,
                        ),
                    )
                    for weight in [0.25, 0.5, 0.75, 1]
                },
            }
            (folder / f"{name}.json").write_text(json.dumps(result, indent=2))
            print(
                json.dumps(
                    {
                        "seed": seed,
                        **{
                            key: value
                            for key, value in result.items()
                            if key != "features"
                        },
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    from lab1_resources import ensure_managed

    ensure_managed(__file__)
    main()
