from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from lab1_features import CATEGORICAL
from lab1_neural import DenseClassifier, preprocessing, set_seed
from lab1_resources import configure_torch

ROOT = Path(__file__).parent


def input_key(table, numeric, store, vision):
    digest = hashlib.sha256(Path(__file__).read_bytes())
    columns = ["author", "post"] + numeric + CATEGORICAL
    digest.update(
        pd.util.hash_pandas_object(table[columns], index=False).to_numpy().tobytes()
    )
    digest.update(store.info["revision"].encode())
    digest.update(store.info["input_sha256"].encode())
    if vision is not None:
        digest.update(vision.info["input_sha256"].encode())
    return digest.hexdigest()[:24]


def context_memmap(table, store, folder, vision=None):
    path = folder / "vectors.npy"
    ready = folder / "vectors-ready.json"
    width = 6 * store.posts.shape[1] + (1536 if vision is not None else 0)
    if ready.exists():
        values = np.load(path, mmap_mode="r")
        if values.shape != (len(table), width):
            raise ValueError("Cached dense vector shape changed")
        return values
    required = len(table) * width * 4
    if required + 5 * 2**30 > shutil.disk_usage(folder).free:
        raise RuntimeError("Insufficient disk reserve for streamed dense features")
    output = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(len(table), width)
    )
    historical = store.history_keys.groupby("author", sort=False).indices
    working = table.reset_index(drop=True)
    for author, positions in working.groupby("author", sort=False).groups.items():
        positions = np.asarray(positions, dtype=int)
        part = working.iloc[positions]
        posts, profiles = store.arrays(part)
        past = np.asarray(historical.get(author, []), dtype=int)
        timeline = pd.concat(
            [
                part[["posted_at", "hist_n"]].reset_index(drop=True),
                store.history_keys.iloc[past][["posted_at", "history_order"]].rename(
                    columns={"history_order": "hist_n"}
                ),
            ],
            ignore_index=True,
        )
        vectors = np.vstack([posts, store.history[past]])
        order = timeline.sort_values(
            ["posted_at", "hist_n"], kind="mergesort"
        ).index.to_numpy()
        values = vectors[order]
        sums = np.vstack(
            [np.zeros((1, values.shape[1])), values.cumsum(axis=0, dtype=np.float64)]
        )
        rows = np.arange(len(values))
        recent = np.zeros_like(vectors)
        overall = np.zeros_like(vectors)
        recent[order] = (sums[rows] - sums[np.maximum(0, rows - 5)]) / np.maximum(
            1, np.minimum(rows, 5)
        )[:, None]
        overall[order] = sums[rows] / np.maximum(1, rows)[:, None]
        recent, overall = recent[: len(part)], overall[: len(part)]
        blocks = [
            posts,
            profiles,
            posts - recent,
            posts - overall,
            posts * recent,
            posts * profiles,
        ]
        if vision is not None:
            images, captions, _ = vision.arrays(part)
            blocks.extend([images, captions, images * captions])
        output[positions] = np.hstack(blocks).astype(np.float32)
    output.flush()
    ready.write_text(json.dumps({"shape": list(output.shape)}))
    del output
    return np.load(path, mmap_mode="r")


def transform_inputs(table, vectors, indices, scaler, processor, path):
    indices = np.asarray(indices, dtype=int)
    if not len(indices):
        return None
    sample = processor.transform(table.iloc[indices[:1]])
    width = vectors.shape[1] + sample.shape[1]
    output = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(len(indices), width)
    )
    for start in range(0, len(indices), 1024):
        rows = indices[start : start + 1024]
        dense = scaler.transform(np.array(vectors[rows], dtype=np.float32, copy=True))
        metadata = processor.transform(table.iloc[rows]).astype(np.float32)
        values = np.hstack([dense, metadata])
        np.clip(values, -8, 8, out=values)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite dense input")
        output[start : start + len(rows)] = values
    output.flush()
    del output
    return np.load(path, mmap_mode="r")


def dense_inputs(table, numeric, store, fit_indices, val_indices=None, vision=None):
    key = input_key(table, numeric, store, vision)
    folder = ROOT / ".cache" / f"dense-stream-{key}"
    folder.mkdir(parents=True, exist_ok=True)
    vectors = context_memmap(table, store, folder, vision)
    fit_indices = np.asarray(fit_indices, dtype=int)
    val_indices = np.asarray([] if val_indices is None else val_indices, dtype=int)
    digest = hashlib.sha256(fit_indices.tobytes() + val_indices.tobytes()).hexdigest()[
        :20
    ]
    part = folder / digest
    part.mkdir(exist_ok=True)
    if (part / "ready.json").exists():
        saved = joblib.load(part / "preprocess.joblib")
        return (
            np.load(part / "fit.npy", mmap_mode="r"),
            np.load(part / "val.npy", mmap_mode="r") if len(val_indices) else None,
            saved["processor"],
            saved["scaler"],
        )
    scaler = StandardScaler()
    for start in range(0, len(fit_indices), 2048):
        scaler.partial_fit(vectors[fit_indices[start : start + 2048]])
    processor = preprocessing(numeric).fit(table.iloc[fit_indices])
    fit = transform_inputs(
        table, vectors, fit_indices, scaler, processor, part / "fit.npy"
    )
    val = transform_inputs(
        table, vectors, val_indices, scaler, processor, part / "val.npy"
    )
    joblib.dump(
        {"processor": processor, "scaler": scaler},
        part / "preprocess.joblib",
        compress=3,
    )
    (part / "ready.json").write_text(
        json.dumps({"fit": len(fit_indices), "val": len(val_indices)})
    )
    return fit, val, processor, scaler


def inference_inputs(table, numeric, store, scaler, processor, vision=None):
    key = input_key(table, numeric, store, vision)
    folder = ROOT / ".cache" / f"dense-stream-{key}"
    folder.mkdir(parents=True, exist_ok=True)
    vectors = context_memmap(table, store, folder, vision)
    return transform_inputs(
        table,
        vectors,
        np.arange(len(table)),
        scaler,
        processor,
        folder / "inference.npy",
    )


def probability(model, inputs, device="cpu"):
    model.eval()
    result = np.empty((len(inputs), 3), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(inputs), 1024):
            batch = torch.as_tensor(
                np.array(inputs[start : start + 1024], copy=True), device=device
            )
            result[start : start + len(batch)] = (
                model(batch).softmax(dim=1).cpu().numpy()
            )
    return result


def fit_streamed(
    inputs,
    labels,
    device,
    seed,
    epochs,
    schedule_epochs=40,
    validation=None,
    callback=None,
):
    configure_torch(torch)
    set_seed(seed)
    model = DenseClassifier(inputs.shape[1]).to(device)
    weights = len(labels) / (3 * np.bincount(labels, minlength=3))
    criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=schedule_epochs
    )
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(labels), device=device).cpu().numpy()
        total = 0.0
        for start in range(0, len(order), 512):
            rows = order[start : start + 512]
            batch = torch.as_tensor(np.array(inputs[rows], copy=True), device=device)
            target = torch.as_tensor(labels[rows], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(rows)
        scheduler.step()
        entry = {"epoch": epoch, "training_loss": total / len(labels)}
        if validation is not None:
            val_inputs, val_labels = validation
            model.eval()
            numerator = 0.0
            predictions = np.empty((len(val_labels), 3), dtype=np.float32)
            with torch.inference_mode():
                for start in range(0, len(val_labels), 1024):
                    batch = torch.as_tensor(
                        np.array(val_inputs[start : start + 1024], copy=True),
                        device=device,
                    )
                    target = torch.as_tensor(
                        val_labels[start : start + 1024],
                        dtype=torch.long,
                        device=device,
                    )
                    logits = model(batch)
                    numerator += nn.functional.cross_entropy(
                        logits, target, weight=criterion.weight, reduction="sum"
                    ).item()
                    predictions[start : start + len(target)] = (
                        logits.softmax(dim=1).cpu().numpy()
                    )
            validation_loss = float(numerator / weights[val_labels].sum())
            entry["validation_loss"] = validation_loss
            if callback:
                callback(entry, predictions)
            if validation_loss < best_loss - 1e-4:
                best_loss, best_epoch, stale = validation_loss, epoch, 0
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            else:
                stale += 1
        else:
            print(json.dumps(entry), flush=True)
        history.append(entry)
        if validation is not None and stale >= 6:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    predicted = (
        probability(model, validation[0], device) if validation is not None else None
    )
    return (
        model.cpu().eval(),
        predicted,
        {
            "best_epoch": best_epoch or epochs,
            "best_validation_loss": best_loss if validation is not None else None,
            "history": history,
        },
    )
