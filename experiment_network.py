from __future__ import annotations

import json
from pathlib import Path

from catboost import CatBoostClassifier
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiment_lab1 import experiment_table, reference_probabilities
from experiment_target70 import previous_probability, rich_inputs
from kairos.metrics import score
from lab1_features import CATEGORICAL
from lab1_graph import GraphEncoder
from lab1_rich import RICH_CAT, rich_table
from lab1_semantic_history import history_names


def main():
    with threadpool_limits(limits=2):
        train, numeric, store = experiment_table(
            Path("data"), Path(".cache/embeddings")
        )
        train, rich_numeric = rich_table(train, Path("data"))
        for seed in [42, 137]:
            _, _, fit, val, tree_numeric, _, text = reference_probabilities(
                train, numeric, store, seed, Path("results/improvement") / str(seed)
            )
            graph = GraphEncoder().fit(fit)
            extra_fit, extra_val = graph.transform(fit), graph.transform(val)
            fit = pd.concat(
                [fit.reset_index(drop=True), extra_fit.reset_index(drop=True)], axis=1
            )
            val = pd.concat(
                [val.reset_index(drop=True), extra_val.reset_index(drop=True)], axis=1
            )
            features = tree_numeric + history_names() + rich_numeric + list(extra_fit)
            model = CatBoostClassifier(
                iterations=3000,
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
                early_stopping_rounds=150,
                use_best_model=True,
                verbose=500,
            )
            probability = model.predict_proba(
                rich_inputs(val, features), thread_count=2
            )
            folder = Path("results/target70") / str(seed)
            model.save_model(str(folder / "network.cbm"))
            joblib.dump(graph, folder / "network-encoder.joblib", compress=3)
            np.savez_compressed(
                folder / "network.npz",
                probability=probability,
                labels=val.label.to_numpy(),
                authors=val.author.to_numpy(dtype=str),
                posts=val.post.to_numpy(dtype=str),
            )
            v2 = previous_probability(seed)
            result = {
                "iterations": model.tree_count_,
                "features": features,
                "standalone": score(val.label, probability.argmax(axis=1)),
                "v2_blends": {
                    str(weight): score(
                        val.label,
                        np.argmax(
                            ((1 - weight) * v2 + weight * probability) * [0.75, 1, 0.6],
                            axis=1,
                        ),
                    )
                    for weight in [0.1, 0.25, 0.5]
                },
            }
            (folder / "network.json").write_text(json.dumps(result, indent=2))
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
