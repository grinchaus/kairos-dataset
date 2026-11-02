from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits
import torch

from experiment_lab1 import experiment_table
from experiment_qwen import qwen_table
from experiment_target70 import previous_probability
from kairos.metrics import score
from lab1_embeddings import CONTEXT_NUM, EmbeddingStore
from lab1_streaming import dense_inputs, fit_streamed
from lab1_rich import rich_table
from lab1_semantic_history import history_names
from lab1_vision import VisionStore, enrich_vision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 137])
    parser.add_argument("--network-offset", type=int, default=0)
    parser.add_argument("--vision", action="store_true")
    args = parser.parse_args()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    with threadpool_limits(limits=2):
        train, numeric, _ = experiment_table(Path("data"), Path(".cache/embeddings"))
        train, rich_numeric = rich_table(train, Path("data"))
        qwen = EmbeddingStore(Path(".cache/embeddings-qwen"))
        train, qwen_numeric = qwen_table(train, qwen, Path("data"))
        columns = numeric + rich_numeric + qwen_numeric + CONTEXT_NUM + history_names()
        vision = None
        if args.vision:
            vision = VisionStore(Path(".cache/embeddings-vision"))
            train, visual_numeric = enrich_vision(train, vision, Path("data"))
            columns += visual_numeric
        for seed in args.seeds:
            fit_idx, val_idx = next(
                GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
                    train, groups=train.author
                )
            )
            fit, val, processor, scaler = dense_inputs(
                train, columns, qwen, fit_idx, val_idx, vision
            )
            y_fit, y_val = (
                train.label.to_numpy()[fit_idx],
                train.label.to_numpy()[val_idx],
            )

            def callback(entry, probability):
                print(
                    json.dumps(
                        {
                            "seed": seed,
                            **entry,
                            **score(y_val, probability.argmax(axis=1)),
                        }
                    ),
                    flush=True,
                )

            model, probability, metadata = fit_streamed(
                fit,
                y_fit,
                device,
                seed + args.network_offset,
                40,
                validation=(val, y_val),
                callback=callback,
            )
            name = ("multimodal-dense" if args.vision else "qwen-dense") + (
                f"-{args.network_offset}" if args.network_offset else ""
            )
            folder = Path("results/target70") / str(seed)
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                folder / f"{name}.npz",
                probability=probability,
                labels=y_val,
                authors=train.iloc[val_idx].author.to_numpy(dtype=str),
                posts=train.iloc[val_idx].post.to_numpy(dtype=str),
            )
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "features": fit.shape[1],
                    "metadata": metadata,
                },
                folder / f"{name}.pt",
            )
            joblib.dump(
                {"processor": processor, "vector_scaler": scaler},
                folder / f"{name}-preprocess.joblib",
            )
            v2 = previous_probability(seed)
            result = {
                "metadata": metadata,
                "numeric_features": columns,
                "standalone": score(y_val, probability.argmax(axis=1)),
                "v2_blends": {
                    str(weight): score(
                        y_val,
                        np.argmax(
                            ((1 - weight) * v2 + weight * probability) * [0.75, 1, 0.6],
                            axis=1,
                        ),
                    )
                    for weight in [0.1, 0.2, 0.3, 0.4, 0.5]
                },
            }
            (folder / f"{name}.json").write_text(json.dumps(result, indent=2))
            print(json.dumps({"seed": seed, "result": result}), flush=True)


if __name__ == "__main__":
    from lab1_resources import ensure_managed

    ensure_managed(__file__)
    main()
