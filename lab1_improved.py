from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from lab1_embeddings import CONTEXT_NUM, EmbeddingStore
from lab1_neural import (
    DenseClassifier,
    context_vectors,
    dense_probability,
    preprocessing,
    refit_dense,
)
from lab1_semantic_history import enrich_semantic_table, history_names
from train_lab1 import blend, checked_table, tabular


def improved_inputs(data, embeddings):
    table, numeric = checked_table(data)
    store = EmbeddingStore(embeddings)
    table = enrich_semantic_table(table, store, data)
    return store.context(table), numeric, store


def fit_dense_components(table, numeric, store, selection, device):
    train_mask = (table.split == "train").to_numpy()
    train = table.loc[train_mask].reset_index(drop=True)
    vectors = context_vectors(table, store)[train_mask]
    scaler = StandardScaler().fit(vectors)
    processor = preprocessing(numeric + CONTEXT_NUM + history_names())
    inputs = np.hstack([scaler.transform(vectors), processor.fit_transform(train)])
    inputs = np.clip(inputs, -8, 8).astype(np.float32)
    models = []
    for seed in selection["dense_seeds"]:
        model = refit_dense(
            inputs,
            train.label.to_numpy(),
            device,
            seed,
            selection["dense_epochs"],
            selection["dense_schedule_epochs"],
        )
        models.append(
            {
                "features": inputs.shape[1],
                "state": {
                    name: value.numpy().copy()
                    for name, value in model.state_dict().items()
                },
            }
        )
    return {
        "models": models,
        "scaler": scaler,
        "processor": processor,
        "numeric": numeric + CONTEXT_NUM + history_names(),
    }


def predict_components(package, table, store, split="test"):
    for name, digest in package["selection"].get("feature_source_sha256", {}).items():
        path = Path(__file__).parent / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Feature implementation differs from the frozen model")
    reference = package["reference"]
    for key in ["revision", "input_sha256", "history_input_sha256"]:
        if reference["embedding_info"].get(key) != store.info.get(key):
            raise ValueError("Encoder inputs do not match the frozen model")
    mask = (table.split == split).to_numpy()
    part = table.loc[mask].reset_index(drop=True)
    projected, _ = reference["embedding_projection"].transform(part, store)
    ref_tree = reference["tree"].predict_proba(tabular(projected, reference["numeric"]))
    text = reference["text"].predict_proba(part)
    semantic = package["semantic"].predict_proba(
        tabular(projected, package["semantic_numeric"])
    )
    dense = package["dense"]
    vectors = context_vectors(table, store)[mask]
    inputs = np.hstack(
        [dense["scaler"].transform(vectors), dense["processor"].transform(part)]
    )
    inputs = np.clip(inputs, -8, 8).astype(np.float32)
    probabilities = []
    for saved in dense["models"]:
        model = DenseClassifier(saved["features"])
        model.load_state_dict(
            {name: torch.as_tensor(values) for name, values in saved["state"].items()}
        )
        probabilities.append(dense_probability(model, inputs))
    neural = np.mean(probabilities, axis=0)
    selection = package["selection"]
    choices = {
        "reference_tree": ref_tree,
        "semantic_tree": semantic,
        "average_tree": (ref_tree + semantic) / 2,
    }
    selected_tree = choices[selection["tree"]]
    weights = np.asarray(selection["blend"]["tree_weight"])
    base = weights * selected_tree + (1 - weights) * text
    weight = selection["neural_weight"]
    selected = np.argmax(
        ((1 - weight) * base + weight * neural) * selection["blend"]["class_weights"],
        axis=1,
    )
    return part, {
        "selected_blend": selected,
        "previous_submission": blend(ref_tree, text, reference["config"]),
        "history_catboost": semantic.argmax(axis=1),
        "text_and_metadata": text.argmax(axis=1),
        "dense_classifier": neural.argmax(axis=1),
    }
