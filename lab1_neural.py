from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

from lab1_features import CATEGORICAL


def context_vectors(table, store):
    posts, profiles = store.arrays(table)
    timeline = table[["author", "posted_at", "hist_n"]].reset_index(drop=True)
    history = store.history_keys[["author", "posted_at", "history_order"]].rename(
        columns={"history_order": "hist_n"}
    )
    timeline = pd.concat(
        [timeline, history[history.author.isin(table.author)]], ignore_index=True
    )
    mask = store.history_keys.author.isin(table.author).to_numpy()
    vectors = np.vstack([posts, store.history[mask]])
    recent = np.zeros_like(vectors)
    overall = np.zeros_like(vectors)
    ordered = timeline.sort_values(["author", "posted_at", "hist_n"], kind="mergesort")
    for _, positions in ordered.groupby("author", sort=False).groups.items():
        positions = np.asarray(positions, dtype=int)
        values = vectors[positions]
        sums = np.vstack(
            [np.zeros((1, values.shape[1])), values.cumsum(axis=0, dtype=np.float64)]
        )
        row = np.arange(len(positions))
        recent[positions] = (sums[row] - sums[np.maximum(0, row - 5)]) / np.maximum(
            1, np.minimum(row, 5)
        )[:, None]
        overall[positions] = sums[row] / np.maximum(1, row)[:, None]
    recent, overall = recent[: len(table)], overall[: len(table)]
    return np.hstack(
        [
            posts,
            profiles,
            posts - recent,
            posts - overall,
            posts * recent,
            posts * profiles,
        ]
    ).astype(np.float32)


def preprocessing(numeric):
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="median", add_indicator=True),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "category",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
        ],
        sparse_threshold=0,
    )


class DenseClassifier(nn.Module):
    def __init__(self, features, hidden=256, dropout=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def forward(self, inputs):
        return self.layers(inputs)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_dense(
    x_train, y_train, x_val, y_val, device, seed=42, epochs=40, callback=None
):
    set_seed(seed)
    torch.set_num_threads(3)
    model = DenseClassifier(x_train.shape[1]).to(device)
    train_x = torch.as_tensor(x_train, device=device)
    train_y = torch.as_tensor(y_train, dtype=torch.long, device=device)
    val_x = torch.as_tensor(x_val, device=device)
    val_y = torch.as_tensor(y_val, dtype=torch.long, device=device)
    weights = len(y_train) / (3 * np.bincount(y_train, minlength=3))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_y), device=device)
        total = 0.0
        for start in range(0, len(order), 512):
            indices = order[start : start + 512]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(train_x[indices]), train_y[indices])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(indices)
        scheduler.step()
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(batch) for batch in val_x.split(2048)])
            validation_loss = criterion(logits, val_y).item()
            probability = logits.softmax(dim=1).cpu().numpy()
        entry = {
            "epoch": epoch,
            "training_loss": total / len(train_y),
            "validation_loss": validation_loss,
        }
        history.append(entry)
        if callback:
            callback(entry, probability)
        if validation_loss < best_loss - 1e-4:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= 6:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probability = (
            torch.cat([model(batch) for batch in val_x.split(2048)])
            .softmax(dim=1)
            .cpu()
            .numpy()
        )
    return (
        model.cpu(),
        probability,
        {
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "history": history,
        },
    )


def refit_dense(inputs, labels, device, seed=42, epochs=2, schedule_epochs=40):
    set_seed(seed)
    torch.set_num_threads(3)
    model = DenseClassifier(inputs.shape[1]).to(device)
    x = torch.as_tensor(inputs, device=device)
    y = torch.as_tensor(labels, dtype=torch.long, device=device)
    weights = len(labels) / (3 * np.bincount(labels, minlength=3))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=schedule_epochs
    )
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(labels), device=device)
        total = 0.0
        for start in range(0, len(order), 512):
            rows = order[start : start + 512]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x[rows]), y[rows])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(rows)
        scheduler.step()
        print(
            f"Dense refit seed={seed} epoch={epoch} loss={total/len(labels):.5f}",
            flush=True,
        )
    return model.cpu().eval()


def dense_probability(model, inputs):
    model.eval()
    with torch.no_grad():
        values = [
            model(torch.as_tensor(batch, dtype=torch.float32)).softmax(dim=1).numpy()
            for batch in np.array_split(
                inputs, max(1, int(np.ceil(len(inputs) / 2048)))
            )
        ]
    return np.vstack(values)
