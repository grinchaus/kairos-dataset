from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kairos.metrics import score
from train_lab1 import blend, select_blend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/improvement"))
    args = ap.parse_args()
    frozen = json.loads(Path("submission/selection.json").read_text())["blend"]
    data = {}
    for seed in [42, 137]:
        folder = args.results / str(seed)
        reference = np.load(folder / "reference.npz")
        semantic = np.load(folder / "semantic.npz")
        values = {
            "labels": reference["labels"],
            "text": reference["text"],
            "reference_tree": reference["tree"],
            "semantic_tree": semantic["tree"],
        }
        neighbors = folder / "neighbors.npz"
        if neighbors.is_file():
            neighbor_data = np.load(neighbors)
            if not np.array_equal(
                reference["posts"], neighbor_data["posts"]
            ) or not np.array_equal(reference["authors"], neighbor_data["authors"]):
                raise ValueError("Neighbor candidate split mismatch")
            values["neighbor_tree"] = neighbor_data["tree"]
        for name in ["dense", "dense_ensemble", "finetune"]:
            path = folder / (name + ".npz")
            if path.is_file():
                candidate = np.load(path)
                if not np.array_equal(
                    reference["posts"], candidate["posts"]
                ) or not np.array_equal(reference["authors"], candidate["authors"]):
                    raise ValueError("Candidate author/post split mismatch")
                values[name] = candidate["probability"]
        data[seed] = values
    candidates = {}
    tree_names = ["reference_tree", "semantic_tree", "average_tree"]
    if all("neighbor_tree" in values for values in data.values()):
        tree_names.append("neighbor_tree")
    for tree_name in tree_names:
        for seed, values in data.items():
            values["average_tree"] = (
                values["reference_tree"] + values["semantic_tree"]
            ) / 2
        primary = data[42]
        selected = select_blend(primary["labels"], primary[tree_name], primary["text"])
        trials = [
            (label, config, neural)
            for label, config in [("frozen", frozen), ("selected", selected)]
            for neural in [
                None,
                "dense",
                "dense_ensemble",
                "finetune",
                "average_neural",
            ]
        ]
        for label, config, neural in trials:
            if (
                neural
                and neural != "average_neural"
                and any(neural not in value for value in data.values())
            ):
                continue
            if neural == "average_neural" and any(
                any(name not in value for name in ["dense", "finetune"])
                for value in data.values()
            ):
                continue
            for weight in [0.0] if neural is None else [0.1, 0.2, 0.3, 0.4]:
                name = f"{tree_name}+text:{label}" + (
                    f"+{neural}:{weight}" if neural else ""
                )
                results = {}
                for seed, values in data.items():
                    tree_weight = np.asarray(config["tree_weight"])
                    base = (
                        tree_weight * values[tree_name]
                        + (1 - tree_weight) * values["text"]
                    )
                    if neural:
                        probabilities = (
                            (values["dense"] + values["finetune"]) / 2
                            if neural == "average_neural"
                            else values[neural]
                        )
                        base = (1 - weight) * base + weight * probabilities
                    prediction = np.argmax(base * config["class_weights"], axis=1)
                    results[str(seed)] = score(values["labels"], prediction)
                candidates[name] = {
                    "tree": tree_name,
                    "neural": neural,
                    "neural_weight": weight,
                    "blend": config,
                    "validation": results,
                    "mean_macro_f1": float(
                        np.mean([r["macro_f1"] for r in results.values()])
                    ),
                }
    reference = {
        str(seed): score(
            values["labels"], blend(values["reference_tree"], values["text"], frozen)
        )
        for seed, values in data.items()
    }
    ranked = sorted(
        candidates.items(), key=lambda entry: entry[1]["mean_macro_f1"], reverse=True
    )
    report = {
        "selection_scope": "official train only; comparison over two author-group holdouts",
        "reference": reference,
        "ranked": [{"name": name, **entry} for name, entry in ranked],
    }
    (args.results / "comparison.json").write_text(json.dumps(report, indent=2))
    print("REFERENCE", json.dumps(reference))
    for name, value in ranked[:12]:
        print(
            name,
            json.dumps(
                {"mean": value["mean_macro_f1"], "validation": value["validation"]}
            ),
        )


if __name__ == "__main__":
    main()
