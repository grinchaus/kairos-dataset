from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


class TextContextClassifier(nn.Module):
    def __init__(
        self, model_name, revision, static_features, mean, scale, top_layers=2
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.top_layers = top_layers
        for layer in self.encoder.encoder.layer[-top_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        self.register_buffer("text_mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("text_scale", torch.as_tensor(scale, dtype=torch.float32))
        self.text_projection = nn.Sequential(
            nn.Linear(len(mean), 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(0.25)
        )
        self.static_projection = nn.Sequential(
            nn.Linear(static_features, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(576, 128), nn.GELU(), nn.Dropout(0.25), nn.Linear(128, 3)
        )

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        if mode:
            for layer in self.encoder.encoder.layer[-self.top_layers :]:
                layer.train()
        return self

    def forward(self, input_ids, attention_mask, static):
        output = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = F.normalize(pooled, p=2, dim=1)
        text = self.text_projection(
            ((pooled - self.text_mean) / self.text_scale).clamp(-8, 8)
        )
        context = self.static_projection(static)
        return self.classifier(torch.cat([text, context, text * context], dim=1))


def length_batches(indices, lengths, batch_size, rng=None):
    indices = np.asarray(indices).copy()
    if rng is not None:
        rng.shuffle(indices)
        blocks = [
            indices[start : start + batch_size * 64]
            for start in range(0, len(indices), batch_size * 64)
        ]
        batches = []
        for block in blocks:
            block = block[np.argsort(lengths[block], kind="stable")]
            batches.extend(
                block[start : start + batch_size]
                for start in range(0, len(block), batch_size)
            )
        rng.shuffle(batches)
        return batches
    indices = indices[np.argsort(lengths[indices], kind="stable")]
    return [
        indices[start : start + batch_size]
        for start in range(0, len(indices), batch_size)
    ]
