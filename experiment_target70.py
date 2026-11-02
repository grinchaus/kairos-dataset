from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np
from threadpoolctl import threadpool_limits

from experiment_lab1 import experiment_table, reference_probabilities
from kairos.metrics import score
from lab1_features import CATEGORICAL
from lab1_rich import RICH_CAT, rich_table
from lab1_kernel_history import enrich_kernel, kernel_names
from lab1_semantic_history import history_names
from train_lab1 import blend, select_blend


def rich_inputs(table, numeric):
    result = table[numeric + CATEGORICAL + RICH_CAT].copy()
    result[numeric] = result[numeric].fillna(-999.0)
    return result


def previous_probability(seed):
    folder = Path("results/improvement") / str(seed)
    semantic = np.load(folder / "semantic.npz")
    dense = np.load(folder / "dense_ensemble.npz")
    selection = json.loads(Path("submission-v2/selection.json").read_text())
    w = np.asarray(selection["blend"]["tree_weight"])
    base = w * semantic["tree"] + (1 - w) * semantic["text"]
    return (1 - selection["neural_weight"]) * base + selection["neural_weight"] * dense[
        "probability"
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["multiclass", "ordinal", "regression"], default="multiclass"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 137])
    parser.add_argument("--kernel", action="store_true")
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("results/target70"))
    args = parser.parse_args()
    train, numeric, store = experiment_table(Path("data"), Path(".cache/embeddings"))
    train, rich_numeric = rich_table(train, Path("data"))
    with threadpool_limits(limits=2):
        if args.kernel:
            train = enrich_kernel(train, store, Path("data"))
            rich_numeric += kernel_names()
        for seed in args.seeds:
            start = time.time()
            folder = args.output / str(seed)
            folder.mkdir(parents=True, exist_ok=True)
            _, _, fit, val, tree_numeric, _, text_probability = reference_probabilities(
                train, numeric, store, seed, Path("results/improvement") / str(seed)
            )
            features = tree_numeric + history_names() + rich_numeric
            x_fit, x_val = rich_inputs(fit, features), rich_inputs(val, features)
            options = dict(
                iterations=3000,
                depth=args.depth,
                learning_rate=0.04,
                l2_leaf_reg=8,
                random_seed=42,
                thread_count=2,
                allow_writing_files=False,
            )
            iterations = []
            name = f"{'kernel' if args.kernel else 'rich'}-{args.mode}"
            if args.depth != 6:
                name += f"-depth{args.depth}"
            if args.mode == "multiclass":
                model = CatBoostClassifier(
                    **options, loss_function="MultiClass", auto_class_weights="Balanced"
                )
                model.fit(
                    x_fit,
                    fit.label,
                    cat_features=CATEGORICAL + RICH_CAT,
                    eval_set=(x_val, val.label),
                    early_stopping_rounds=150,
                    use_best_model=True,
                    verbose=500,
                )
                probability = model.predict_proba(x_val, thread_count=2)
                model.save_model(str(folder / f"{name}.cbm"))
                iterations.append(model.tree_count_)
            elif args.mode == "ordinal":
                outputs = []
                for boundary in [0, 1]:
                    model = CatBoostClassifier(**options, loss_function="Logloss")
                    model.fit(
                        x_fit,
                        (fit.label > boundary).astype(int),
                        cat_features=CATEGORICAL + RICH_CAT,
                        eval_set=(x_val, (val.label > boundary).astype(int)),
                        early_stopping_rounds=150,
                        use_best_model=True,
                        verbose=500,
                    )
                    outputs.append(model.predict_proba(x_val, thread_count=2)[:, 1])
                    iterations.append(model.tree_count_)
                    model.save_model(str(folder / f"{name}-{boundary}.cbm"))
                low, high = np.maximum(outputs[0], outputs[1]), np.minimum(
                    outputs[0], outputs[1]
                )
                probability = np.column_stack([1 - low, low - high, high])
            else:
                model = CatBoostRegressor(**options, loss_function="RMSE")
                model.fit(
                    x_fit,
                    fit.resid,
                    cat_features=CATEGORICAL + RICH_CAT,
                    eval_set=(x_val, val.resid),
                    early_stopping_rounds=150,
                    use_best_model=True,
                    verbose=500,
                )
                iterations.append(model.tree_count_)
                mean = model.predict(x_val, thread_count=2)
                from scipy.special import ndtr

                low = ndtr((-0.5493061443340549 - mean) / 0.7)
                high = ndtr((1.157452788691043 - mean) / 0.7)
                probability = np.column_stack([low, high - low, 1 - high])
                model.save_model(str(folder / f"{name}.cbm"))
            np.savez_compressed(
                folder / f"{name}.npz",
                probability=probability,
                labels=val.label.to_numpy(),
                authors=val.author.to_numpy(dtype=str),
                posts=val.post.to_numpy(dtype=str),
            )
            selection = select_blend(val.label, probability, text_probability)
            v2 = previous_probability(seed)
            experiment = {
                "seed": seed,
                "mode": args.mode,
                "iterations": iterations,
                "features": features,
                "categorical": CATEGORICAL + RICH_CAT,
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
                "elapsed_seconds": round(time.time() - start, 1),
            }
            (folder / f"{name}.json").write_text(json.dumps(experiment, indent=2))
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in experiment.items()
                        if key not in ["features", "categorical"]
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    from lab1_resources import ensure_managed

    ensure_managed(__file__)
    main()
