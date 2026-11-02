from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from experiment_lab1 import experiment_table, reference_probabilities
from kairos.metrics import score
from lab1_features import CATEGORICAL
from lab1_neighbors import neighbor_features, neighbor_names
from lab1_semantic_history import history_names
from train_lab1 import boosted, tabular, select_blend, blend


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(3)
    train, numeric, store = experiment_table(Path("data"), Path(".cache/embeddings"))
    for seed in [42, 137]:
        folder = Path("results/improvement") / str(seed)
        with threadpool_limits(limits=6):
            _, _, fit, val, features, _, text = reference_probabilities(
                train, numeric, store, seed, folder
            )
            bank, _ = store.arrays(fit)
            query, _ = store.arrays(val)
            fit_neighbors = neighbor_features(
                bank, fit.author, bank, fit.author, fit.label, device
            )
            val_neighbors = neighbor_features(
                query, val.author, bank, fit.author, fit.label, device
            )
            fit, val = fit.copy(), val.copy()
            for position, name in enumerate(neighbor_names()):
                fit[name] = fit_neighbors[:, position]
                val[name] = val_neighbors[:, position]
            numeric_features = features + history_names() + neighbor_names()
            model = boosted(2800, 6, 6)
            model.fit(
                tabular(fit, numeric_features),
                fit.label,
                cat_features=CATEGORICAL,
                eval_set=(tabular(val, numeric_features), val.label),
                early_stopping_rounds=150,
                use_best_model=True,
                verbose=500,
            )
            probability = model.predict_proba(tabular(val, numeric_features))
            selection = select_blend(val.label, probability, text)
            np.savez_compressed(
                folder / "neighbors.npz",
                tree=probability,
                text=text,
                labels=val.label.to_numpy(),
                posts=val.post.to_numpy(dtype=str),
                authors=val.author.to_numpy(dtype=str),
            )
            result = {
                "standalone": score(val.label, probability.argmax(axis=1)),
                "blend": score(val.label, blend(probability, text, selection)),
                "selection": selection,
                "iterations": model.tree_count_,
            }
            (folder / "neighbors.json").write_text(json.dumps(result, indent=2))
            model.save_model(str(folder / "neighbors.cbm"))
            print(json.dumps({"seed": seed, **result}), flush=True)


if __name__ == "__main__":
    main()
