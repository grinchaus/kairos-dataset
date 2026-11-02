from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import joblib
import pandas as pd
import sklearn
import torch
from sklearn.metrics import classification_report, confusion_matrix
from threadpoolctl import threadpool_limits

import baseline
from kairos.metrics import score
from lab1_features import CATEGORICAL
from lab1_improved import fit_dense_components, improved_inputs, predict_components
from lab1_semantic_history import history_names
from lab1_uncertainty import author_bootstrap
from train_lab1 import (
    RELEASE_COUNTS,
    baseline_table,
    boosted,
    tabular,
    text_fingerprint,
)


def main():
    ap = argparse.ArgumentParser(
        description="Reproduce the frozen local improvement of Lab 1"
    )
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--embeddings", type=Path, default=Path(".cache/embeddings"))
    ap.add_argument(
        "--selection", type=Path, default=Path("submission-v2/selection.json")
    )
    ap.add_argument("--output", type=Path, default=Path("results/lab1-v2"))
    args = ap.parse_args()
    root = Path(__file__).resolve().parent

    def run(*command):
        subprocess.run([sys.executable, *map(str, command)], cwd=root, check=True)

    selection = json.loads(args.selection.read_text())
    run("-m","kairos.label","--data",args.data.resolve())
    if selection["neural"] not in ["dense", "dense_ensemble"]:
        raise SystemExit(
            "This reproduction entry point expects a validated dense classifier"
        )
    if (args.output / "metrics.json").exists():
        saved = args.output / "selection.json"
        if not saved.is_file() or json.loads(saved.read_text()) != selection:
            raise SystemExit(
                "Output already contains another selection; choose another --output"
            )
    if not Path("results/lab1/model.joblib").is_file():
        run("run_lab1.py", "--data", args.data.resolve())
    start = time.time()
    table, numeric, store = improved_inputs(args.data, args.embeddings)
    train = table[table.split == "train"].reset_index(drop=True)
    test = table[table.split == "test"].reset_index(drop=True)
    counts = {
        "train_posts": len(train),
        "test_posts": len(test),
        "train_authors": int(train.author.nunique()),
        "test_authors": int(test.author.nunique()),
    }
    if counts != RELEASE_COUNTS:
        raise SystemExit("Official release coverage mismatch")
    digest = hashlib.sha256((args.data / "labels.csv").read_bytes()).hexdigest()
    if selection["labels_sha256"] != digest or selection[
        "text_input_sha256"
    ] != text_fingerprint(train, numeric):
        raise SystemExit("Frozen selection refers to other release inputs")
    reference = joblib.load("results/lab1/model.joblib")
    projected, _ = reference["embedding_projection"].transform(train, store)
    semantic_numeric = reference["numeric"] + history_names()
    with threadpool_limits(limits=8):
        semantic = boosted(selection["catboost_iterations"], 8, 6)
        semantic.fit(
            tabular(projected, semantic_numeric),
            train.label,
            cat_features=CATEGORICAL,
            verbose=500,
        )
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    dense = fit_dense_components(table, numeric, store, selection, device)
    package = {
        "reference": reference,
        "semantic": semantic,
        "semantic_numeric": semantic_numeric,
        "dense": dense,
        "selection": selection,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, args.output / "model.joblib", compress=3)
    pd.Series(
        semantic.feature_importances_, index=semantic_numeric + CATEGORICAL
    ).sort_values(ascending=False).to_csv(
        args.output / "feature_importance.csv", header=["importance"]
    )
    part, predictions = predict_components(package, table, store)
    previous = pd.read_csv("submission/selected_blend.csv")
    old = part[["author", "post"]].assign(label=predictions["previous_submission"])
    pd.testing.assert_frame_equal(old, previous)
    ref_table = baseline_table(args.data)
    with threadpool_limits(limits=8):
        ref = baseline.model().fit(ref_table[ref_table.split == "train"], train.label)
        predictions["official_baseline"] = ref.predict(
            ref_table[ref_table.split == "test"]
        )
    results = {}
    for name, prediction in predictions.items():
        results[name] = {
            **score(part.label, prediction),
            "confusion_matrix": confusion_matrix(
                part.label, prediction, labels=[0, 1, 2]
            ).tolist(),
            "classification_report": classification_report(
                part.label,
                prediction,
                labels=[0, 1, 2],
                target_names=["below", "typical", "above"],
                output_dict=True,
                zero_division=0,
            ),
        }
        part[["author", "post"]].assign(label=prediction).to_csv(
            args.output / f"{name}.csv", index=False
        )
    previous_ci = author_bootstrap(
        part.author,
        part.label,
        {
            "selected_blend": predictions["selected_blend"],
            "official_baseline": predictions["previous_submission"],
        },
    )
    old_metrics = json.loads(Path("submission/metrics.json").read_text())
    manifest = {
        **old_metrics,
        "test": results,
        "elapsed_seconds": round(time.time() - start, 1),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "selected_config": selection["blend"],
        "catboost_iterations": selection["catboost_iterations"],
        "confidence_intervals": author_bootstrap(part.author, part.label, predictions),
        "improvement": selection,
        "previous_comparison": {
            "previous_macro_f1": results["previous_submission"]["macro_f1"],
            "delta_macro_f1": round(
                results["selected_blend"]["macro_f1"]
                - results["previous_submission"]["macro_f1"],
                4,
            ),
            "paired_delta_95ci": previous_ci["selected_minus_baseline_95ci"],
        },
    }
    manifest["validation"]["selected_blend"] = selection["validation"]["42"]
    manifest["validation"]["history_catboost"] = selection["semantic_validation"]["42"]
    manifest["validation"]["dense_classifier"] = selection["dense_validation"]["42"]
    (args.output / "metrics.json").write_text(json.dumps(manifest, indent=2))
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2))
    print(
        json.dumps(
            {
                name: {key: entry[key] for key in ["macro_f1", "qwk", "accuracy"]}
                for name, entry in results.items()
            },
            indent=2,
        ),
        flush=True,
    )
    run(
        "predict_lab1_v2.py",
        "--data",
        args.data.resolve(),
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
        args.data.resolve(),
        "--predictions",
        args.output / "selected_blend.csv",
    )
    run(
        "make_report.py",
        "--results",
        args.output,
        "--output",
        "submission-v2/Lab1_Rostislav_Manzyuk.pdf",
    )
    run(
        "tools/prepare_submission.py",
        "--data",
        args.data.resolve(),
        "--results",
        args.output,
        "--output",
        "submission-v2",
    )
    print("Local improved submission verified: submission-v2/", flush=True)


if __name__ == "__main__":
    main()
