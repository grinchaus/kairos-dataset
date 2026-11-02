from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

from kairos.metrics import score
from lab1_embeddings import CONTEXT_NUM, EmbeddingProjection, EmbeddingStore
from lab1_features import CATEGORICAL
from lab1_semantic_history import history_names, semantic_history
from train_lab1 import boosted, checked_table, select_blend, blend, tabular, text_model


def experiment_table(data, embeddings):
    table, numeric = checked_table(data)
    store = EmbeddingStore(embeddings)
    digest = hashlib.sha256(Path("lab1_semantic_history.py").read_bytes())
    digest.update(store.info["input_sha256"].encode())
    digest.update(store.info["history_input_sha256"].encode())
    digest.update(np.nan_to_num(table.log_reactions.to_numpy()).tobytes())
    for path in sorted((Path(data) / "posts").glob("*.json")):
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    cache = Path(".cache") / f"semantic-{digest.hexdigest()[:20]}.joblib"
    if cache.is_file():
        extra = joblib.load(cache)
    else:
        extra = semantic_history(table, store, data)[history_names()]
        temporary = cache.with_name(cache.name + f".{os.getpid()}.tmp")
        joblib.dump(extra, temporary, compress=3)
        temporary.replace(cache)
    for name in history_names():
        table[name] = extra[name].to_numpy()
    table = store.context(table)
    return table[table.split == "train"].reset_index(drop=True), numeric, store


def reference_probabilities(train, numeric, store, seed, folder):
    fit_idx, val_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
            train, groups=train.author
        )
    )
    fit, val = train.iloc[fit_idx], train.iloc[val_idx]
    if set(fit.author) & set(val.author):
        raise AssertionError("Author leakage")
    projection = EmbeddingProjection(64, 16).fit(fit, store)
    fit, projected = projection.transform(fit, store)
    val, _ = projection.transform(val, store)
    tree_numeric = numeric + CONTEXT_NUM + projected
    output = folder / "reference.npz"
    if output.is_file():
        saved = np.load(output)
        if not np.array_equal(saved["posts"], val.post) or not np.array_equal(
            saved["authors"], val.author
        ):
            raise ValueError("Reference split mismatch")
        tree_prob, text_prob = saved["tree"], saved["text"]
    elif (
        seed == 42
        and Path("results/validation-timestamps/validation_probabilities.npz").is_file()
    ):
        saved = np.load("results/validation-timestamps/validation_probabilities.npz")
        if not np.array_equal(saved["posts"], val.post) or not np.array_equal(
            saved["authors"], val.author
        ):
            raise ValueError("Original validation split mismatch")
        tree_prob, text_prob = saved["tree"], saved["text"]
        np.savez_compressed(
            output,
            tree=tree_prob,
            text=text_prob,
            posts=val.post.to_numpy(dtype=str),
            authors=val.author.to_numpy(dtype=str),
            labels=val.label.to_numpy(),
        )
    else:
        reference = boosted(2088, 2, 6)
        reference.fit(
            tabular(fit, tree_numeric), fit.label, cat_features=CATEGORICAL, verbose=500
        )
        tree_prob = reference.predict_proba(tabular(val, tree_numeric), thread_count=2)
        linear = text_model(numeric).fit(fit, fit.label)
        text_prob = linear.predict_proba(val)
        np.savez_compressed(
            output,
            tree=tree_prob,
            text=text_prob,
            posts=val.post.to_numpy(dtype=str),
            authors=val.author.to_numpy(dtype=str),
            labels=val.label.to_numpy(),
        )
    return fit_idx, val_idx, fit, val, tree_numeric, tree_prob, text_prob


def main():
    ap = argparse.ArgumentParser(
        description="Train-only candidate experiments, with held-out authors"
    )
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--embeddings", type=Path, default=Path(".cache/embeddings"))
    ap.add_argument("--output", type=Path, default=Path("results/improvement"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 137])
    args = ap.parse_args()
    train, numeric, store = experiment_table(args.data, args.embeddings)
    summary = {}
    with threadpool_limits(limits=2):
        for seed in args.seeds:
            start = time.time()
            folder = args.output / str(seed)
            folder.mkdir(parents=True, exist_ok=True)
            _, _, fit, val, tree_numeric, reference_tree, text = (
                reference_probabilities(train, numeric, store, seed, folder)
            )
            frozen = json.loads(Path("submission/selection.json").read_text())["blend"]
            reference = score(val.label, blend(reference_tree, text, frozen))
            features = tree_numeric + history_names()
            model = boosted(2800, 8, 6)
            model.fit(
                tabular(fit, features),
                fit.label,
                cat_features=CATEGORICAL,
                eval_set=(tabular(val, features), val.label),
                early_stopping_rounds=150,
                use_best_model=True,
                verbose=400,
            )
            probability = model.predict_proba(tabular(val, features), thread_count=2)
            selection = select_blend(val.label, probability, text)
            frozen_result = score(val.label, blend(probability, text, frozen))
            selected_result = score(val.label, blend(probability, text, selection))
            np.savez_compressed(
                folder / "semantic.npz",
                tree=probability,
                text=text,
                labels=val.label.to_numpy(),
                posts=val.post.to_numpy(dtype=str),
                authors=val.author.to_numpy(dtype=str),
            )
            model.save_model(str(folder / "semantic.cbm"))
            summary[str(seed)] = {
                "reference_frozen": reference,
                "semantic_frozen": frozen_result,
                "semantic_selected": selected_result,
                "selection": selection,
                "iterations": model.tree_count_,
                "elapsed_seconds": round(time.time() - start, 1),
            }
            (folder / "semantic.json").write_text(
                json.dumps(summary[str(seed)], indent=2)
            )
            print(json.dumps({"seed": seed, **summary[str(seed)]}), flush=True)
    (args.output / "semantic-summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
