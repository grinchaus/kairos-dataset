from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("HF_HOME", str(Path(__file__).parent / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import joblib
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from experiment_lab1 import experiment_table
from kairos.metrics import score
from lab1_embeddings import CONTEXT_NUM
from lab1_finetune import TextContextClassifier, length_batches
from lab1_neural import context_vectors, preprocessing, set_seed
from lab1_semantic_history import history_names
from train_lab1 import select_blend, blend


def token_cache(train, model, revision, tokenizer):
    fingerprint = hashlib.sha256((model + revision + "512").encode())
    fingerprint.update("\0".join(train.clean_text).encode())
    folder = Path(".cache") / f"tokens-{fingerprint.hexdigest()[:16]}"
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / "ids.npy").is_file():
        tokens = tokenizer(
            ["query: " + text for text in train.clean_text],
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )
        np.save(
            folder / "lengths.npy",
            tokens["attention_mask"].sum(axis=1).astype(np.int16),
        )
        np.save(folder / "ids.npy", tokens["input_ids"].astype(np.int32))
    return np.load(folder / "ids.npy", mmap_mode="r"), np.load(folder / "lengths.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--embeddings", type=Path, default=Path(".cache/embeddings"))
    ap.add_argument("--output", type=Path, default=Path("results/improvement"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 137])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(3)
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Fine-tuning device:", device, flush=True)
    train, numeric, store = experiment_table(args.data, args.embeddings)
    model_name, revision = store.info["model"], store.info["revision"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, local_files_only=True, trust_remote_code=False
    )
    tokens, lengths = token_cache(train, model_name, revision, tokenizer)
    posts, profiles = store.arrays(train)
    contextual = context_vectors(train, store)
    dimension = posts.shape[1]
    fixed = np.hstack([profiles, posts - contextual[:, 2 * dimension : 3 * dimension]])
    del contextual
    for seed in args.seeds:
        start_time = time.time()
        folder = args.output / str(seed)
        folder.mkdir(parents=True, exist_ok=True)
        set_seed(seed)
        fit_idx, val_idx = next(
            GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
                train, groups=train.author
            )
        )
        processor = preprocessing(numeric + CONTEXT_NUM + history_names())
        scaler = StandardScaler().fit(fixed[fit_idx])
        text_scaler = StandardScaler().fit(posts[fit_idx])
        static = np.empty((len(train), 0), dtype=np.float32)
        fit_static = np.hstack(
            [
                scaler.transform(fixed[fit_idx]),
                processor.fit_transform(train.iloc[fit_idx]),
            ]
        )
        val_static = np.hstack(
            [scaler.transform(fixed[val_idx]), processor.transform(train.iloc[val_idx])]
        )
        static = np.empty((len(train), fit_static.shape[1]), dtype=np.float32)
        static[fit_idx] = np.clip(fit_static, -8, 8)
        static[val_idx] = np.clip(val_static, -8, 8)
        static = torch.as_tensor(static, device=device)
        model = TextContextClassifier(
            model_name, revision, static.shape[1], text_scaler.mean_, text_scaler.scale_
        ).to(device)
        encoder_parameters, head_parameters = [], []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                (
                    encoder_parameters
                    if name.startswith("encoder.")
                    else head_parameters
                ).append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_parameters, "lr": 4e-6},
                {"params": head_parameters, "lr": 2e-4},
            ],
            weight_decay=0.02,
        )
        steps_per_epoch = math.ceil(len(fit_idx) / args.batch_size)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            int(0.05 * steps_per_epoch * args.epochs),
            steps_per_epoch * args.epochs,
        )
        labels = train.label.to_numpy()
        class_weights = len(fit_idx) / (3 * np.bincount(labels[fit_idx], minlength=3))
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
        )
        targets = torch.as_tensor(labels, dtype=torch.long, device=device)
        best_loss, best_state, best_probability, best_step, stale = (
            float("inf"),
            None,
            None,
            0,
            0,
        )
        history = []
        step = 0

        def tensors(indices):
            length = min(512, int(math.ceil(int(lengths[indices].max()) / 8) * 8))
            ids = torch.tensor(
                np.asarray(tokens[indices, :length]), dtype=torch.long, device=device
            )
            return ids, (ids != tokenizer.pad_token_id).long(), static[indices]

        def evaluate():
            model.eval()
            probability = np.zeros((len(val_idx), 3), dtype=np.float32)
            logits_all = torch.empty((len(val_idx), 3), device=device)
            positions = np.full(len(train), -1, dtype=int)
            positions[val_idx] = np.arange(len(val_idx))
            with torch.no_grad():
                for indices in length_batches(val_idx, lengths, 32):
                    logits_all[positions[indices]] = model(*tensors(indices))
                loss = criterion(logits_all, targets[val_idx]).item()
                probability[:] = logits_all.softmax(dim=1).cpu().numpy()
            return loss, probability

        stop = False
        for epoch in range(1, args.epochs + 1):
            rng = np.random.default_rng(seed + epoch)
            batches = length_batches(fit_idx, lengths, args.batch_size, rng)
            for batch_number, indices in enumerate(batches, 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(*tensors(indices)), targets[indices])
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                step += 1
                if step % 100 == 0:
                    elapsed = time.time() - start_time
                    print(
                        json.dumps(
                            {
                                "seed": seed,
                                "epoch": epoch,
                                "step": step,
                                "loss": round(loss.item(), 4),
                                "elapsed_seconds": round(elapsed),
                                "steps_per_second": round(step / elapsed, 2),
                            }
                        ),
                        flush=True,
                    )
                if batch_number % max(1, len(batches) // 2) == 0 or batch_number == len(
                    batches
                ):
                    validation_loss, probability = evaluate()
                    entry = {
                        "step": step,
                        "epoch": epoch,
                        "validation_loss": validation_loss,
                        **score(labels[val_idx], probability.argmax(axis=1)),
                    }
                    history.append(entry)
                    print(json.dumps({"seed": seed, "validation": entry}), flush=True)
                    if validation_loss < best_loss - 1e-4:
                        best_loss, best_step, best_probability, stale = (
                            validation_loss,
                            step,
                            probability.copy(),
                            0,
                        )
                        trained = {
                            name
                            for name, p in model.named_parameters()
                            if p.requires_grad
                        }
                        best_state = {
                            name: value.detach().cpu().clone()
                            for name, value in model.state_dict().items()
                            if name in trained or not name.startswith("encoder.")
                        }
                        torch.save(best_state, folder / "finetune.pt")
                    else:
                        stale += 1
                        if stale >= 3:
                            stop = True
                            break
            if stop:
                break
        metadata = {
            "model": model_name,
            "revision": revision,
            "top_layers": 2,
            "maximum_length": 512,
            "best_step": best_step,
            "steps_per_epoch": steps_per_epoch,
            "seed": seed,
            "best_validation_loss": best_loss,
            "history": history,
            "elapsed_seconds": round(time.time() - start_time),
        }
        np.savez_compressed(
            folder / "finetune.npz",
            probability=best_probability,
            labels=labels[val_idx],
            authors=train.iloc[val_idx].author.to_numpy(dtype=str),
            posts=train.iloc[val_idx].post.to_numpy(dtype=str),
        )
        joblib.dump(
            {"processor": processor, "scaler": scaler, "text_scaler": text_scaler},
            folder / "finetune-preprocess.joblib",
        )
        result = {
            "metadata": metadata,
            "standalone": score(labels[val_idx], best_probability.argmax(axis=1)),
        }
        reference = np.load(folder / "reference.npz")
        frozen = json.loads(Path("submission/selection.json").read_text())["blend"]
        weight = np.asarray(frozen["tree_weight"])
        base = weight * reference["tree"] + (1 - weight) * reference["text"]
        selection = select_blend(labels[val_idx], base, best_probability)
        result.update(
            selection=selection,
            blended=score(labels[val_idx], blend(base, best_probability, selection)),
        )
        (folder / "finetune.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({"seed": seed, "finetune_result": result}), flush=True)
        del model, static, optimizer
        if device == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
