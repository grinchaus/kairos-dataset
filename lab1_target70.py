from __future__ import annotations

import json
import hashlib
from pathlib import Path

from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np
import pandas as pd
from scipy.special import ndtr
import torch

from experiment_graph_text import graph_text_model
from experiment_qwen import qwen_project, qwen_table
from experiment_target70 import rich_inputs
from lab1_embeddings import CONTEXT_NUM, EmbeddingProjection, EmbeddingStore
from lab1_features import CATEGORICAL
from lab1_graph import GraphEncoder
from lab1_improved import improved_inputs
from lab1_kernel_history import enrich_kernel
from lab1_neural import (
    DenseClassifier,
    context_vectors,
    dense_probability,
)
from lab1_rich import RICH_CAT, rich_table
from lab1_vision import VisionProjection, VisionStore, enrich_vision
from lab1_streaming import (
    dense_inputs,
    fit_streamed,
    inference_inputs,
    probability as streamed_probability,
)
from lab1_semantic_history import history_names
from train_lab1 import boosted, tabular, text_model


def release_fingerprint(data):
    data = Path(data)
    paths = [data / "labels.csv", data / "split.json", data / "media/manifest.jsonl"]
    for directory in ["authors", "posts"]:
        paths.extend(sorted((data / directory).glob("*.json")))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(data).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def fit_previous(fit, numeric, store, device, threads=2):
    selection = json.loads(Path("submission-v2/selection.json").read_text())
    projection = EmbeddingProjection(64, 16).fit(fit, store)
    projected, names = projection.transform(fit, store)
    tree_numeric = numeric + CONTEXT_NUM + names
    semantic_numeric = tree_numeric + history_names()
    semantic = boosted(selection["catboost_iterations"], threads, 6)
    semantic.fit(
        tabular(projected, semantic_numeric),
        fit.label,
        cat_features=CATEGORICAL,
        verbose=500,
    )
    linear = text_model(numeric).fit(fit, fit.label)
    columns = numeric + CONTEXT_NUM + history_names()
    inputs, _, processor, scaler = dense_inputs(
        fit, columns, store, np.arange(len(fit))
    )
    models = []
    for seed in selection["dense_seeds"]:
        model, _, _ = fit_streamed(
            inputs,
            fit.label.to_numpy(),
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
    dense = {
        "models": models,
        "scaler": scaler,
        "processor": processor,
        "numeric": columns,
    }
    reference = {
        "tree": None,
        "text": linear,
        "numeric": tree_numeric,
        "embedding_projection": projection,
        "embedding_info": store.info,
        "config": json.loads(Path("submission/selection.json").read_text())["blend"],
    }
    return {
        "reference": reference,
        "semantic": semantic,
        "semantic_numeric": semantic_numeric,
        "dense": dense,
        "selection": selection,
    }


def previous_probability(package, part, store):
    reference = package["reference"]
    for key in ["revision", "input_sha256", "history_input_sha256"]:
        if reference["embedding_info"].get(key) != store.info.get(key):
            raise ValueError(f"Previous model encoder changed: {key}")
    projected, _ = reference["embedding_projection"].transform(part, store)
    semantic = package["semantic"].predict_proba(
        tabular(projected, package["semantic_numeric"]), thread_count=2
    )
    text = reference["text"].predict_proba(part)
    dense = package["dense"]
    inputs = np.clip(
        np.hstack(
            [
                dense["scaler"].transform(context_vectors(part, store)),
                dense["processor"].transform(part),
            ]
        ),
        -8,
        8,
    ).astype(np.float32)
    probabilities = []
    for saved in dense["models"]:
        model = DenseClassifier(saved["features"])
        model.load_state_dict(
            {name: torch.as_tensor(values) for name, values in saved["state"].items()}
        )
        probabilities.append(dense_probability(model, inputs))
    neural = np.mean(probabilities, axis=0)
    selection = package["selection"]
    weights = np.asarray(selection["blend"]["tree_weight"])
    base = weights * semantic + (1 - weights) * text
    probability = (1 - selection["neural_weight"]) * base + selection[
        "neural_weight"
    ] * neural
    return probability / probability.sum(axis=1, keepdims=True)


def verify_selection(selection):
    for name, digest in selection["feature_source_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen feature implementation changed: {name}")


def target_inputs(
    data,
    embeddings,
    qwen_folder,
    selection,
    vision_folder=Path(".cache/embeddings-vision"),
):
    verify_selection(selection)
    if release_fingerprint(data) != selection["release_inputs_sha256"]:
        raise ValueError("Raw release inputs differ from the frozen experiment")
    table, numeric, e5 = improved_inputs(data, embeddings)
    table, rich_numeric = rich_table(table, data)
    qwen = None
    if any(
        name.startswith(("qwen-", "multimodal-")) for name in selection["components"]
    ):
        qwen = EmbeddingStore(qwen_folder)
        table, _ = qwen_table(table, qwen, data)
    if "kernel-multiclass" in selection["components"]:
        table = enrich_kernel(table, e5, data)
    vision = None
    if any(
        name in ["vision", "qwen-vision"] or name.startswith("multimodal-")
        for name in selection["components"]
    ):
        vision = VisionStore(vision_folder)
        table, _ = enrich_vision(table, vision, data)
    for name, store in [("e5", e5), ("qwen", qwen), ("vision", vision)]:
        if store is None:
            continue
        for key, value in selection["encoders"][name].items():
            if store.info.get(key) != value:
                raise ValueError(f"Frozen encoder changed: {name}/{key}")
    return table, numeric, rich_numeric, e5, qwen, vision


def fit_component(
    name,
    fit,
    numeric,
    rich_numeric,
    e5,
    qwen,
    previous,
    config,
    device,
    threads=2,
    vision=None,
):
    if name == "graph-text":
        return {
            "kind": "graph-text",
            "model": graph_text_model(numeric + rich_numeric).fit(fit, fit.label),
        }
    if name.startswith(("qwen-dense", "multimodal-dense")):
        columns = config["numeric"]
        inputs, _, processor, scaler = dense_inputs(
            fit,
            columns,
            qwen,
            np.arange(len(fit)),
            vision=vision if name.startswith("multimodal-") else None,
        )
        models = []
        for seed in config["network_seeds"]:
            model, _, _ = fit_streamed(
                inputs, fit.label.to_numpy(), device, seed, config["epochs"], 40
            )
            models.append(
                {
                    "features": inputs.shape[1],
                    "state": {
                        key: value.numpy().copy()
                        for key, value in model.state_dict().items()
                    },
                }
            )
        return {
            "kind": "qwen-dense",
            "multimodal": name.startswith("multimodal-"),
            "models": models,
            "scaler": scaler,
            "processor": processor,
            "numeric": columns,
        }
    projection = previous["reference"]["embedding_projection"]
    frame, _ = projection.transform(fit, e5)
    saved = {
        "kind": "tree",
        "config": config,
        "prior": np.bincount(fit.label, minlength=3) / len(fit),
    }
    if name.startswith("qwen-depth") or name == "qwen-vision":
        qwen_projection = EmbeddingProjection(96, 24).fit(fit, qwen)
        frame, _ = qwen_project(qwen_projection, frame, qwen)
        saved["qwen_projection"] = qwen_projection
    if name in ["vision", "qwen-vision"]:
        projection = VisionProjection().fit(fit, vision)
        frame, _ = projection.transform(frame, vision)
        saved["vision_projection"] = projection
    if name == "network":
        graph = GraphEncoder().fit(fit)
        extra = graph.transform(fit)
        frame = pd.concat(
            [frame.reset_index(drop=True), extra.reset_index(drop=True)], axis=1
        )
        saved["graph"] = graph
    inputs = rich_inputs(frame, config["features"])
    options = dict(
        depth=config["depth"],
        learning_rate=0.04,
        l2_leaf_reg=8,
        random_seed=42,
        thread_count=threads,
        allow_writing_files=False,
    )
    models = []
    for index, iterations in enumerate(config["iterations"]):
        if name == "rich-ordinal":
            model = CatBoostClassifier(
                **options, iterations=iterations, loss_function="Logloss"
            )
            labels = (fit.label > index).astype(int)
        elif name == "rich-regression":
            model = CatBoostRegressor(
                **options, iterations=iterations, loss_function="RMSE"
            )
            labels = fit.resid
        else:
            model = CatBoostClassifier(
                **options,
                iterations=iterations,
                loss_function="MultiClass",
                auto_class_weights="Balanced",
            )
            labels = fit.label
        model.fit(inputs, labels, cat_features=CATEGORICAL + RICH_CAT, verbose=500)
        models.append(model)
    saved["models"] = models
    saved["name"] = name
    return saved


def component_probability(component, part, e5, qwen, previous, vision=None):
    if component["kind"] == "graph-text":
        probability = component["model"].predict_proba(part)
    elif component["kind"] == "qwen-dense":
        inputs = inference_inputs(
            part,
            component["numeric"],
            qwen,
            component["scaler"],
            component["processor"],
            vision if component.get("multimodal") else None,
        )
        outputs = []
        for saved in component["models"]:
            model = DenseClassifier(saved["features"])
            model.load_state_dict(
                {name: torch.as_tensor(value) for name, value in saved["state"].items()}
            )
            outputs.append(streamed_probability(model, inputs))
        probability = np.mean(outputs, axis=0)
    else:
        frame, _ = previous["reference"]["embedding_projection"].transform(part, e5)
        if "qwen_projection" in component:
            frame, _ = qwen_project(component["qwen_projection"], frame, qwen)
        if "vision_projection" in component:
            frame, _ = component["vision_projection"].transform(frame, vision)
        if "graph" in component:
            extra = component["graph"].transform(part)
            frame = pd.concat(
                [frame.reset_index(drop=True), extra.reset_index(drop=True)], axis=1
            )
        inputs = rich_inputs(frame, component["config"]["features"])
        models = component["models"]
        if component["name"] == "rich-ordinal":
            first, second = [
                model.predict_proba(inputs, thread_count=2)[:, 1] for model in models
            ]
            low, high = np.maximum(first, second), np.minimum(first, second)
            probability = (
                np.column_stack([1 - low, low - high, high]) / component["prior"]
            )
        elif component["name"] == "rich-regression":
            mean = models[0].predict(inputs, thread_count=2)
            low = ndtr((-0.5493061443340549 - mean) / 0.7)
            high = ndtr((1.157452788691043 - mean) / 0.7)
            probability = (
                np.column_stack([low, high - low, 1 - high]) / component["prior"]
            )
        else:
            probability = models[0].predict_proba(inputs, thread_count=2)
    return probability / probability.sum(axis=1, keepdims=True)


def predict_target(package, part, e5, qwen, vision=None):
    torch.set_num_threads(2)
    verify_selection(package["selection"])
    probabilities = {"v2": previous_probability(package["previous"], part, e5)}
    for name, component in package["components"].items():
        probabilities[name] = component_probability(
            component, part, e5, qwen, package["previous"], vision
        )
    probability = sum(
        weight * probabilities[name]
        for name, weight in package["selection"]["components"].items()
    )
    prediction = np.argmax(probability * package["selection"]["class_weights"], axis=1)
    return prediction, probabilities
