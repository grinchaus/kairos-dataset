from __future__ import annotations

import argparse
import json
from pathlib import Path

from catboost import CatBoostClassifier
import numpy as np
from threadpoolctl import threadpool_limits

from experiment_lab1 import experiment_table, reference_probabilities
from experiment_qwen import qwen_project, qwen_table
from experiment_target70 import previous_probability, rich_inputs
from kairos.metrics import score
from lab1_embeddings import EmbeddingProjection, EmbeddingStore
from lab1_features import CATEGORICAL
from lab1_rich import RICH_CAT, rich_table
from lab1_semantic_history import history_names
from lab1_vision import VisionProjection, VisionStore, enrich_vision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        train, numeric, e5 = experiment_table(Path("data"), Path(".cache/embeddings"))
        train, rich_numeric = rich_table(train, Path("data"))
        vision = VisionStore(Path(".cache/embeddings-vision"))
        train, vision_numeric = enrich_vision(train, vision, Path("data"))
        qwen_numeric = []
        qwen = None
        if args.qwen:
            qwen = EmbeddingStore(Path(".cache/embeddings-qwen"))
            train, qwen_numeric = qwen_table(train, qwen, Path("data"))
        for seed in [42, 137]:
            _, _, fit, val, tree_numeric, _, _ = reference_probabilities(
                train, numeric, e5, seed, Path("results/improvement") / str(seed)
            )
            projection = VisionProjection().fit(fit, vision)
            fit, projected = projection.transform(fit, vision)
            val, _ = projection.transform(val, vision)
            features = (
                tree_numeric
                + history_names()
                + rich_numeric
                + vision_numeric
                + projected
            )
            if qwen is not None:
                projection = EmbeddingProjection(96, 24).fit(fit, qwen)
                fit, projected = qwen_project(projection, fit, qwen)
                val, _ = qwen_project(projection, val, qwen)
                features += qwen_numeric + projected
            model = CatBoostClassifier(
                iterations=4000,
                depth=6,
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
            name = "qwen-vision" if args.qwen else "vision"
            folder = Path("results/target70") / str(seed)
            model.save_model(str(folder / f"{name}.cbm"))
            np.savez_compressed(
                folder / f"{name}.npz",
                probability=probability,
                labels=val.label.to_numpy(),
                authors=val.author.to_numpy(dtype=str),
                posts=val.post.to_numpy(dtype=str),
            )
            previous = previous_probability(seed)
            result = {
                "iterations": model.tree_count_,
                "features": features,
                "standalone": score(val.label, probability.argmax(axis=1)),
                "v2_blends": {
                    str(weight): score(
                        val.label,
                        np.argmax(
                            ((1 - weight) * previous + weight * probability)
                            * [0.75, 1, 0.6],
                            axis=1,
                        ),
                    )
                    for weight in [0.1, 0.25, 0.5, 0.75]
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
