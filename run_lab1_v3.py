from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits
import torch

import baseline as official_baseline
from kairos.metrics import score
from lab1_target70 import (
    fit_component,
    fit_previous,
    predict_target,
    target_inputs,
    verify_selection,
)
from lab1_uncertainty import author_bootstrap
from train_lab1 import RELEASE_COUNTS, baseline_table


def fit_package(
    fit,
    numeric,
    rich_numeric,
    e5,
    qwen,
    selection,
    previous,
    device,
    output,
    vision=None,
):
    output.mkdir(parents=True, exist_ok=True)
    package = {"selection": selection, "previous": previous, "components": {}}
    keys = hashlib.sha256(
        pd.util.hash_pandas_object(fit[["author", "post"]], index=False)
        .to_numpy()
        .tobytes()
    ).hexdigest()
    for name in selection["components"]:
        if name == "v2":
            continue
        cache = output / f"{name}.joblib"
        if cache.exists():
            saved = joblib.load(cache)
            if saved["selection"] != selection or saved["fit_keys"] != keys:
                raise ValueError(f"Cached component belongs to another fit: {name}")
            component = saved["component"]
        else:
            print("Fitting fixed component:", name, flush=True)
            component = fit_component(
                name,
                fit,
                numeric,
                rich_numeric,
                e5,
                qwen,
                previous,
                selection["component_config"][name],
                device,
                vision=vision,
            )
            joblib.dump(
                {"selection": selection, "fit_keys": keys, "component": component},
                cache,
                compress=3,
            )
        package["components"][name] = component
    return package


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the frozen third Lab 1 experiment"
    )
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--embeddings", type=Path, default=Path(".cache/embeddings"))
    parser.add_argument(
        "--qwen-embeddings", type=Path, default=Path(".cache/embeddings-qwen")
    )
    parser.add_argument(
        "--vision-embeddings", type=Path, default=Path(".cache/embeddings-vision")
    )
    parser.add_argument(
        "--selection", type=Path, default=Path("submission-v3/selection.json")
    )
    parser.add_argument("--output", type=Path, default=Path("results/lab1-v3"))
    parser.add_argument("--audit-seed", type=int, default=2026)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    if args.audit_seed in selection["selection_seeds"]:
        raise ValueError("The additional audit must use a different author split")
    verify_selection(selection)
    if (
        hashlib.sha256((args.data / "labels.csv").read_bytes()).hexdigest()
        != selection["labels_sha256"]
    ):
        raise ValueError("Frozen labels changed")

    def run(*command):
        subprocess.run([sys.executable, *map(str, command)], check=True)

    run("-m", "kairos.label", "--data", args.data)
    if not (args.embeddings / "encoder.json").exists():
        run(
            "embed_lab1.py",
            "--data",
            args.data,
            "--output",
            args.embeddings,
            "--revision",
            selection["encoders"]["e5"]["revision"],
        )
    if (
        "qwen" in selection["encoders"]
        and not (args.qwen_embeddings / "encoder.json").exists()
    ):
        run(
            "embed_lab1_qwen.py",
            "--data",
            args.data,
            "--output",
            args.qwen_embeddings,
            "--revision",
            selection["encoders"]["qwen"]["revision"],
            "--batch-size",
            4,
        )
    if (
        "vision" in selection["encoders"]
        and not (args.vision_embeddings / "encoder.json").exists()
    ):
        run(
            "embed_lab1_vision.py",
            "--data",
            args.data,
            "--output",
            args.vision_embeddings,
            "--revision",
            selection["encoders"]["vision"]["revision"],
        )
    started = time.time()
    torch.set_num_threads(2)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    with threadpool_limits(limits=2):
        table, numeric, rich_numeric, e5, qwen, vision = target_inputs(
            args.data,
            args.embeddings,
            args.qwen_embeddings,
            selection,
            args.vision_embeddings,
        )
        train = table[table.split == "train"].reset_index(drop=True)
        test = table[table.split == "test"].reset_index(drop=True)
        counts = {
            "train_posts": len(train),
            "test_posts": len(test),
            "train_authors": int(train.author.nunique()),
            "test_authors": int(test.author.nunique()),
        }
        if counts != RELEASE_COUNTS:
            raise ValueError("Official release coverage changed")
        args.output.mkdir(parents=True, exist_ok=True)
        fit_idx, val_idx = next(
            GroupShuffleSplit(
                n_splits=1, test_size=0.2, random_state=args.audit_seed
            ).split(train, groups=train.author)
        )
        fit, val = train.iloc[fit_idx].reset_index(drop=True), train.iloc[
            val_idx
        ].reset_index(drop=True)
        audit_folder = Path("results/target70") / str(args.audit_seed)
        audit_folder.mkdir(parents=True, exist_ok=True)
        previous_path = audit_folder / "previous-model.joblib"
        if previous_path.exists():
            previous = joblib.load(previous_path)
        else:
            previous = fit_previous(fit, numeric, e5, device)
            joblib.dump(previous, previous_path, compress=3)
        audit_package = fit_package(
            fit,
            numeric,
            rich_numeric,
            e5,
            qwen,
            selection,
            previous,
            device,
            args.output / "audit-components",
            vision=vision,
        )
        audit_prediction, audit_probabilities = predict_target(
            audit_package, val, e5, qwen, vision
        )
        previous_prediction = np.argmax(
            audit_probabilities["v2"] * [0.75, 1, 0.6], axis=1
        )
        audit = {
            "seed": args.audit_seed,
            "authors": int(val.author.nunique()),
            "posts": len(val),
            "configuration_fixed_before_audit": True,
            "used_for_selection": False,
            "previous_v2": score(val.label, previous_prediction),
            "selected_blend": score(val.label, audit_prediction),
            "comparison": author_bootstrap(
                val.author,
                val.label,
                {
                    "selected_blend": audit_prediction,
                    "official_baseline": previous_prediction,
                },
            ),
        }
        (args.output / "audit.json").write_text(json.dumps(audit, indent=2))
        print("Fixed additional validation:", json.dumps(audit), flush=True)
        del audit_package, previous
        if args.audit_only:
            return
        previous_path = Path("results/lab1-v2/model.joblib")
        local_previous = args.output / "full-previous.joblib"
        if previous_path.exists():
            previous = joblib.load(previous_path)
        elif local_previous.exists():
            previous = joblib.load(local_previous)
        else:
            previous = fit_previous(train, numeric, e5, device)
            joblib.dump(previous, local_previous, compress=3)
        package = fit_package(
            train,
            numeric,
            rich_numeric,
            e5,
            qwen,
            selection,
            previous,
            device,
            args.output / "full-components",
            vision=vision,
        )
        joblib.dump(package, args.output / "model.joblib", compress=3)
        prediction, probabilities = predict_target(package, test, e5, qwen, vision)
        component_previous_prediction = np.argmax(
            probabilities["v2"] * [0.75, 1, 0.6], axis=1
        )
        old = pd.read_csv("submission-v2/selected_blend.csv")
        pd.testing.assert_frame_equal(test[["author", "post"]], old[["author", "post"]])
        previous_prediction = old.label.to_numpy()
        reference = baseline_table(args.data)
        reference_train = reference[reference.split == "train"].reset_index(drop=True)
        reference_test = reference[reference.split == "test"].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            reference_train[["author", "post"]], train[["author", "post"]]
        )
        pd.testing.assert_frame_equal(
            reference_test[["author", "post"]], test[["author", "post"]]
        )
        baseline_model = official_baseline.model().fit(
            reference_train, reference_train.label
        )
        baseline_prediction = baseline_model.predict(reference_test)
        predictions = {
            "selected_blend": prediction,
            "previous_v2": previous_prediction,
            "official_baseline": baseline_prediction,
        }
        results = {}
        for name, values in predictions.items():
            results[name] = {
                **score(test.label, values),
                "confusion_matrix": confusion_matrix(
                    test.label, values, labels=[0, 1, 2]
                ).tolist(),
                "classification_report": classification_report(
                    test.label,
                    values,
                    labels=[0, 1, 2],
                    target_names=["below", "typical", "above"],
                    output_dict=True,
                    zero_division=0,
                ),
            }
            test[["author", "post"]].assign(label=values).to_csv(
                args.output / f"{name}.csv", index=False
            )
        comparison = author_bootstrap(
            test.author,
            test.label,
            {"selected_blend": prediction, "official_baseline": previous_prediction},
        )
        metrics = {
            "author": "Rostislav Manzyuk",
            "github": "grinchaus",
            "data_kind": "official-release",
            **counts,
            "labels_sha256": selection["labels_sha256"],
            "python": platform.python_version(),
            "resources": {
                "cpu_threads": 2,
                "mps_allocator_fraction": 0.12,
                "concurrent_heavy_tasks": 1,
                "dense_input_storage": "float32 memory maps",
                "gpu_neural_input": "minibatches",
            },
            "elapsed_seconds": round(time.time() - started, 1),
            "selection": selection,
            "audit": audit,
            "test": results,
            "confidence_intervals": author_bootstrap(
                test.author, test.label, predictions
            ),
            "previous_comparison": {
                "previous_macro_f1": results["previous_v2"]["macro_f1"],
                "delta_macro_f1": round(
                    results["selected_blend"]["macro_f1"]
                    - results["previous_v2"]["macro_f1"],
                    4,
                ),
                "paired_delta_95ci": comparison["selected_minus_baseline_95ci"],
            },
            "target_reached": f1_score(test.label, prediction, average="macro") >= 0.70,
            "reproducibility": {
                "previous_component_prediction_differences": int(
                    np.count_nonzero(
                        component_previous_prediction != previous_prediction
                    )
                ),
                "comparison_uses_frozen_previous_csv": True,
            },
            "limitations": [
                "Previously evaluated test; this round freezes all settings before scoring.",
                "Selection holdouts overlap and have been used for several experiments.",
                "Reaction and profile histories are crawl-time snapshots.",
                "Graph labels are fitted only on fit authors, excluding the entire current author.",
                "Images are processed locally; no text or media are sent to external inference APIs.",
            ],
        }
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (args.output / "selection.json").write_text(json.dumps(selection, indent=2))
        importances = []
        for name, component in package["components"].items():
            if component["kind"] == "tree":
                for index, model in enumerate(component["models"]):
                    for feature, value in zip(
                        model.feature_names_,
                        model.get_feature_importance(thread_count=2),
                    ):
                        importances.append(
                            {
                                "component": name,
                                "model": index,
                                "feature": feature,
                                "importance": value,
                            }
                        )
        pd.DataFrame(
            importances, columns=["component", "model", "feature", "importance"]
        ).to_csv(args.output / "feature_importance.csv", index=False)
        print(
            json.dumps(
                {
                    name: score(test.label, values)
                    for name, values in predictions.items()
                },
                indent=2,
            ),
            flush=True,
        )
    del (
        table,
        train,
        test,
        fit,
        val,
        e5,
        qwen,
        vision,
        previous,
        package,
        reference,
        reference_train,
        reference_test,
        baseline_model,
    )
    component = model = None
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    run(
        "predict_lab1_v3.py",
        "--data",
        args.data,
        "--embeddings",
        args.embeddings,
        "--qwen-embeddings",
        args.qwen_embeddings,
        "--vision-embeddings",
        args.vision_embeddings,
        "--model",
        args.output / "model.joblib",
        "--output",
        args.output / "reproduced.csv",
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(args.output / "selected_blend.csv"),
        pd.read_csv(args.output / "reproduced.csv"),
    )
    run(
        "evaluate.py",
        "--data",
        args.data,
        "--predictions",
        args.output / "selected_blend.csv",
    )
    run("make_report_v3.py", "--results", args.output)
    run(
        "tools/prepare_submission.py",
        "--data",
        args.data,
        "--results",
        args.output,
        "--output",
        "submission-v3",
    )
    print("Verified local submission-v3; no remote writes", flush=True)


if __name__ == "__main__":
    from lab1_resources import ensure_managed

    ensure_managed(__file__)
    main()
