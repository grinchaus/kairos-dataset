from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from experiment_lab1 import experiment_table
from kairos.metrics import score
from lab1_embeddings import CONTEXT_NUM
from lab1_neural import context_vectors, preprocessing, train_dense
from lab1_semantic_history import history_names
from train_lab1 import select_blend, blend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--embeddings", type=Path, default=Path(".cache/embeddings"))
    ap.add_argument("--output", type=Path, default=Path("results/improvement"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 137])
    ap.add_argument("--network-offset", type=int, default=0)
    args = ap.parse_args()
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Neural device:", device, flush=True)
    train, numeric, store = experiment_table(args.data, args.embeddings)
    vectors = context_vectors(train, store)
    for seed in args.seeds:
        folder = args.output / str(seed)
        folder.mkdir(parents=True, exist_ok=True)
        fit_idx, val_idx = next(
            GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
                train, groups=train.author
            )
        )
        processor = preprocessing(numeric + CONTEXT_NUM + history_names())
        feature_scaler = StandardScaler()
        x_fit = np.hstack(
            [
                feature_scaler.fit_transform(vectors[fit_idx]),
                processor.fit_transform(train.iloc[fit_idx]),
            ]
        )
        x_val = np.hstack(
            [
                feature_scaler.transform(vectors[val_idx]),
                processor.transform(train.iloc[val_idx]),
            ]
        )
        x_fit = np.clip(x_fit, -8, 8).astype(np.float32)
        x_val = np.clip(x_val, -8, 8).astype(np.float32)
        y_fit, y_val = train.label.to_numpy()[fit_idx], train.label.to_numpy()[val_idx]

        def callback(entry, probability):
            print(
                json.dumps(
                    {"seed": seed, **entry, **score(y_val, probability.argmax(axis=1))}
                ),
                flush=True,
            )

        model, probability, metadata = train_dense(
            x_fit,
            y_fit,
            x_val,
            y_val,
            device,
            seed + args.network_offset,
            callback=callback,
        )
        metadata["network_seed"] = seed + args.network_offset
        torch.save(
            {
                "state_dict": model.state_dict(),
                "features": x_fit.shape[1],
                "metadata": metadata,
            },
            folder / "dense.pt",
        )
        joblib.dump(
            {"processor": processor, "vector_scaler": feature_scaler},
            folder / "dense-preprocess.joblib",
        )
        np.savez_compressed(
            folder / "dense.npz",
            probability=probability,
            labels=y_val,
            authors=train.iloc[val_idx].author.to_numpy(dtype=str),
            posts=train.iloc[val_idx].post.to_numpy(dtype=str),
        )
        result = {
            "standalone": score(y_val, probability.argmax(axis=1)),
            "metadata": metadata,
        }
        reference = folder / "reference.npz"
        if reference.is_file():
            previous = np.load(reference)
            frozen = json.loads(Path("submission/selection.json").read_text())["blend"]
            weight = np.asarray(frozen["tree_weight"])
            base = weight * previous["tree"] + (1 - weight) * previous["text"]
            config = select_blend(y_val, base, probability)
            result.update(
                selection=config, blended=score(y_val, blend(base, probability, config))
            )
        (folder / "dense.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({"seed": seed, "dense_result": result}), flush=True)


if __name__ == "__main__":
    main()
