from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from experiment_target70 import previous_probability
from kairos.metrics import score
from train_lab1 import checked_table

SEEDS = [42, 137]


def normalize(probability):
    return probability / probability.sum(axis=1, keepdims=True)


def macro(labels, prediction):
    matrix = np.bincount(labels * 3 + prediction, minlength=9).reshape(3, 3)
    return float(
        (2 * matrix.diagonal() / (matrix.sum(axis=0) + matrix.sum(axis=1))).mean()
    )


def calibrate(probabilities, labels, floors=None):
    best = None
    for low in np.arange(0.55, 1.051, 0.05):
        for high in np.arange(0.45, 0.951, 0.05):
            weights = np.array([low, 1.0, high])
            scores = [
                macro(y, (p * weights).argmax(axis=1))
                for p, y in zip(probabilities, labels)
            ]
            if floors is not None and any(
                value < floor for value, floor in zip(scores, floors)
            ):
                continue
            value = float(np.mean(scores))
            if best is None or value > best[0]:
                best = value, weights.tolist(), scores
    return best if best is not None else (-float("inf"), None, None)


def validation_candidates():
    table, _ = checked_table(Path("data"))
    train = table[table.split == "train"].reset_index(drop=True)
    candidates = {"v2": []}
    labels, authors = [], []
    for seed in SEEDS:
        fit_idx, val_idx = next(
            GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
                train, groups=train.author
            )
        )
        val = train.iloc[val_idx]
        y = val.label.to_numpy(dtype=int)
        labels.append(y)
        authors.append(val.author.to_numpy(dtype=str))
        candidates["v2"].append(normalize(previous_probability(seed)))
        folder = Path("results/target70") / str(seed)
        names = [
            "rich-multiclass",
            "rich-ordinal",
            "rich-regression",
            "kernel-multiclass",
            "graph-text",
            "network",
            "vision",
            "qwen-vision",
            "qwen-depth6",
            "qwen-depth7",
            "qwen-dense",
            "qwen-dense-1000",
            "multimodal-dense",
            "multimodal-dense-1000",
        ]
        for name in names:
            path = folder / f"{name}.npz"
            if not path.exists():
                continue
            saved = np.load(path)
            if (
                not np.array_equal(saved["labels"], y)
                or not np.array_equal(saved["posts"], val.post)
                or not np.array_equal(saved["authors"], val.author)
            ):
                raise ValueError(f"Validation keys changed: {name}/{seed}")
            probability = saved["probability"]
            if name in ["rich-ordinal", "rich-regression"]:
                prior = np.bincount(train.iloc[fit_idx].label, minlength=3) / len(
                    fit_idx
                )
                probability = probability / prior
            candidates.setdefault(name, []).append(normalize(probability))
    candidates = {
        name: probabilities
        for name, probabilities in candidates.items()
        if len(probabilities) == len(SEEDS)
    }
    for prefix in ["qwen-dense", "multimodal-dense"]:
        if prefix in candidates and prefix + "-1000" in candidates:
            candidates[prefix + "-ensemble"] = [
                (left + right) / 2
                for left, right in zip(candidates[prefix], candidates[prefix + "-1000"])
            ]
    for name in ["multimodal-dense", "multimodal-dense-1000"]:
        candidates.pop(name, None)
    return candidates, labels, authors


def component_config(name):
    if name == "v2":
        return None
    if name == "graph-text":
        return {"c": 0.5}
    if name.startswith(("qwen-dense", "multimodal-dense")):
        prefix = name.removesuffix("-ensemble")
        sources = [prefix, prefix + "-1000"] if name.endswith("ensemble") else [name]
        records = [
            json.loads(
                (Path("results/target70") / str(seed) / f"{source}.json").read_text()
            )
            for seed in SEEDS
            for source in sources
        ]
        if any(
            record["numeric_features"] != records[0]["numeric_features"]
            for record in records
        ):
            raise ValueError(f"Neural candidate schemas differ: {name}")
        return {
            "epochs": int(
                np.rint(
                    np.median([record["metadata"]["best_epoch"] for record in records])
                )
            ),
            "numeric": records[0]["numeric_features"],
            "network_seeds": (
                [42, 1042]
                if len(sources) == 2
                else [1042 if name.endswith("1000") else 42]
            ),
        }
    records = [
        json.loads((Path("results/target70") / str(seed) / f"{name}.json").read_text())
        for seed in SEEDS
    ]
    if records[0]["features"] != records[1]["features"]:
        raise ValueError(f"Candidate feature schemas differ: {name}")
    iterations = [
        (
            record["iterations"]
            if isinstance(record["iterations"], list)
            else [record["iterations"]]
        )
        for record in records
    ]
    return {
        "features": records[0]["features"],
        "iterations": np.rint(np.median(iterations, axis=0)).astype(int).tolist(),
        "depth": 7 if name == "qwen-depth7" else 6,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    candidates, labels, authors = validation_candidates()
    summary = {}
    for name, probabilities in candidates.items():
        value, calibration, scores = calibrate(probabilities, labels)
        summary[name] = {
            "mean_macro_f1": value,
            "class_weights": calibration,
            "per_split": scores,
        }
    selected = candidates["v2"]
    mixture = {"v2": 1.0}
    floors = [
        macro(y, (probability * [0.75, 1.0, 0.6]).argmax(axis=1))
        for y, probability in zip(labels, selected)
    ]
    best, calibration, scores = calibrate(selected, labels, floors)
    history = []
    for iteration in range(3):
        choice = None
        for name, probabilities in candidates.items():
            for weight in [0.1, 0.2, 0.3, 0.4, 0.5]:
                trial = [
                    (1 - weight) * current + weight * addition
                    for current, addition in zip(selected, probabilities)
                ]
                value, classes, split_scores = calibrate(trial, labels, floors)
                if choice is None or value > choice[0]:
                    choice = value, name, weight, trial, classes, split_scores
        if choice[0] < best + 0.0004:
            break
        best, name, weight, selected, calibration, scores = choice
        mixture = {key: value * (1 - weight) for key, value in mixture.items()}
        mixture[name] = mixture.get(name, 0) + weight
        history.append(
            {
                "round": iteration + 1,
                "added": name,
                "weight": weight,
                "mean_macro_f1": best,
            }
        )
    result = {
        "target_macro_f1": 0.70,
        "components": mixture,
        "class_weights": calibration,
        "validation": {
            str(seed): score(y, (p * calibration).argmax(axis=1))
            for seed, y, p in zip(SEEDS, labels, selected)
        },
        "mean_macro_f1": best,
        "selection_history": history,
        "candidates": summary,
        "test_used_for_selection": False,
        "selection_seeds": SEEDS,
        "selection_policy": "three greedy additions at most; shared calibration; minimum mean gain 0.0004; neither split may fall below frozen v2",
    }
    output = Path("results/target70/candidate-selection.json")
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if args.freeze:
        if Path("submission-v3/selection.json").exists():
            raise SystemExit("Selection is already frozen; refusing to overwrite it")
        result["component_config"] = {
            name: component_config(name) for name in mixture if name != "v2"
        }
        from lab1_target70 import release_fingerprint

        result["release_inputs_sha256"] = release_fingerprint(Path("data"))
        result["encoders"] = {}
        for name, folder in [
            ("e5", ".cache/embeddings"),
            ("qwen", ".cache/embeddings-qwen"),
            ("vision", ".cache/embeddings-vision"),
        ]:
            if name == "qwen" and not any(
                component.startswith(("qwen-", "multimodal-")) for component in mixture
            ):
                continue
            if name == "vision" and not any(
                component in ["vision", "qwen-vision"]
                or component.startswith("multimodal-")
                for component in mixture
            ):
                continue
            info = json.loads((Path(folder) / "encoder.json").read_text())
            keys = (
                [
                    "model",
                    "revision",
                    "normalized",
                    "input_sha256",
                    "mapping_sha256",
                    "image_size",
                    "caption_max_length",
                    "genres",
                ]
                if name == "vision"
                else [
                    "model",
                    "revision",
                    "prefix",
                    "max_length",
                    "normalized",
                    "channel_hashes",
                    "input_sha256",
                    "history_input_sha256",
                ]
            )
            result["encoders"][name] = {key: info[key] for key in keys}
        result["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["labels_sha256"] = hashlib.sha256(
            Path("data/labels.csv").read_bytes()
        ).hexdigest()
        result["previous_commit"] = "b4add45dcbb287733805efa820bc1998598d521e"
        result["feature_source_sha256"] = {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in [
                "lab1_rich.py",
                "lab1_resources.py",
                "lab1_streaming.py",
                "lab1_vision.py",
                "embed_lab1_vision.py",
                "embed_lab1_qwen.py",
                "lab1_target70.py",
                "lab1_graph.py",
                "lab1_kernel_history.py",
                "experiment_qwen.py",
                "lab1_neural.py",
                "lab1_semantic_history.py",
                "lab1_embeddings.py",
                "lab1_improved.py",
                "lab1_features.py",
                "lab1_metadata.py",
                "train_lab1.py",
                "experiment_target70.py",
                "experiment_graph_text.py",
                "experiment_lab1.py",
            ]
        }
        folder = Path("submission-v3")
        folder.mkdir(exist_ok=True)
        (folder / "selection.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
