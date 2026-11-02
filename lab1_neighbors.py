from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def neighbor_names():
    return [
        f"neighbor{k}_{name}"
        for k in [1, 3, 5, 15, 50]
        for name in [
            "below",
            "typical",
            "above",
            "mean_similarity",
            "spread_similarity",
        ]
    ]


def neighbor_features(
    vectors, authors, bank_vectors, bank_authors, bank_labels, device="cpu"
):
    authors = np.asarray(authors, dtype=str)
    bank_authors = np.asarray(bank_authors, dtype=str)
    bank_labels = np.asarray(bank_labels, dtype=int)
    if len(bank_vectors) != len(bank_authors) or len(bank_labels) != len(bank_authors):
        raise ValueError("Neighbor bank arrays differ in length")
    names = pd.Index(np.unique(np.concatenate([authors, bank_authors])))
    query_codes = torch.tensor(names.get_indexer(authors), device=device)
    bank_codes = torch.tensor(names.get_indexer(bank_authors), device=device)
    bank = torch.as_tensor(np.asarray(bank_vectors), dtype=torch.float32, device=device)
    output = np.full((len(vectors), len(neighbor_names())), np.nan, dtype=np.float32)
    if len(bank) < 1:
        return output
    with torch.no_grad():
        for start in range(0, len(vectors), 256):
            query = torch.as_tensor(
                np.asarray(vectors[start : start + 256]),
                dtype=torch.float32,
                device=device,
            )
            similarities = query @ bank.T
            excluded = (
                query_codes[start : start + len(query), None] == bank_codes[None, :]
            )
            similarities.masked_fill_(excluded, float("-inf"))
            values, indices = similarities.topk(min(50, len(bank)), dim=1, sorted=True)
            values, indices = values.cpu().numpy(), indices.cpu().numpy()
            labels = bank_labels[indices]
            parts = []
            for count in [1, 3, 5, 15, 50]:
                sim = values[:, :count]
                valid = np.isfinite(sim)
                denominator = valid.sum(axis=1).clip(min=1)
                for label in range(3):
                    parts.append(
                        ((labels[:, :count] == label) & valid).sum(axis=1) / denominator
                    )
                mean = np.where(valid, sim, 0).sum(axis=1) / denominator
                spread = np.sqrt(
                    np.where(valid, (sim - mean[:, None]) ** 2, 0).sum(axis=1)
                    / denominator
                )
                mean[~valid.any(axis=1)] = np.nan
                spread[~valid.any(axis=1)] = np.nan
                parts.extend([mean, spread])
            output[start : start + len(query)] = np.column_stack(parts)
    return output
