from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

from experiment_lab1 import experiment_table
from experiment_target70 import previous_probability
from kairos.metrics import score
from lab1_rich import rich_table
from train_lab1 import text_model


def graph_text_model(numeric):
    model = text_model(numeric, c=0.5)
    model.named_steps["features"].transformers.append(
        (
            "graph",
            TfidfVectorizer(
                token_pattern=r"(?u)\b\w+\b",
                min_df=2,
                sublinear_tf=True,
                dtype=np.float32,
            ),
            "graph_text",
        )
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 137])
    args = parser.parse_args()
    train, numeric, _ = experiment_table(Path("data"), Path(".cache/embeddings"))
    train, rich_numeric = rich_table(train, Path("data"))
    with threadpool_limits(limits=2):
        for seed in args.seeds:
            fit_idx, val_idx = next(
                GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
                    train, groups=train.author
                )
            )
            fit, val = train.iloc[fit_idx], train.iloc[val_idx]
            model = graph_text_model(numeric + rich_numeric)
            model.fit(fit, fit.label)
            probability = model.predict_proba(val)
            folder = Path("results/target70") / str(seed)
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                folder / "graph-text.npz",
                probability=probability,
                labels=val.label.to_numpy(),
                authors=val.author.to_numpy(dtype=str),
                posts=val.post.to_numpy(dtype=str),
            )
            v2 = previous_probability(seed)
            result = {
                "standalone": score(val.label, probability.argmax(axis=1)),
                "v2_blends": {
                    str(weight): score(
                        val.label,
                        np.argmax(
                            ((1 - weight) * v2 + weight * probability) * [0.75, 1, 0.6],
                            axis=1,
                        ),
                    )
                    for weight in [0.1, 0.3, 0.5]
                },
            }
            (folder / "graph-text.json").write_text(json.dumps(result, indent=2))
            print(json.dumps({"seed": seed, **result}), flush=True)


if __name__ == "__main__":
    from lab1_resources import ensure_managed

    ensure_managed(__file__)
    main()
