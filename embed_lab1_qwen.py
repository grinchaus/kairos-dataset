from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from lab1_resources import (
    configure_torch,
    ensure_managed,
    validate_completed,
    gpu_usage,
)

if __name__ == "__main__":
    ensure_managed(__file__)

ROOT = Path(__file__).parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from huggingface_hub import model_info
from transformers import AutoModel, AutoTokenizer

from embed_lab1 import channel_hashes, fingerprint, input_channels
from train_lab1 import checked_table

MODEL = "Qwen/Qwen3-Embedding-0.6B"
PROMPT = (
    "Instruct: Represent a LinkedIn post or author profile for predicting whether "
    "a post receives below-typical, typical, or above-typical engagement relative "
    "to the author's previous posts.\nQuery: "
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path(".cache/embeddings-qwen"))
    parser.add_argument("--revision")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_torch(torch)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    prior = args.output / "pending.json"
    revision = args.revision or (
        json.loads(prior.read_text())["revision"]
        if prior.exists()
        else model_info(MODEL, token=False).sha
    )
    table, _ = checked_table(args.data)
    channels = input_channels(args.data, table)
    hashes = channel_hashes(channels)
    metadata = {
        "model": MODEL,
        "revision": revision,
        "prefix": PROMPT,
        "max_length": args.max_length,
        "normalized": True,
        "attention": "sdpa",
        "padding_side": "right",
        "dtype": str(dtype),
        "input_sha256": fingerprint(
            table, ["author", "post", "clean_text", "profile_text"]
        ),
        "channel_hashes": hashes,
        "history_input_sha256": hashes["history"],
        "device": device,
        "torch": torch.__version__,
    }
    pending = args.output / "pending.json"
    started_file = args.output / "started.json"
    if not started_file.exists():
        started_file.write_text(
            json.dumps(
                {
                    "timestamp": (
                        pending.stat().st_mtime if pending.exists() else time.time()
                    )
                }
            )
        )
    if pending.exists() and json.loads(pending.read_text()) != metadata:
        raise ValueError("Embedding run configuration changed")
    pending.write_text(json.dumps(metadata, indent=2))
    if (args.output / "encoder.json").exists():
        print("Complete matching embeddings already exist", flush=True)
        return
    print(f"Loading {MODEL}@{revision} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=revision, token=False, padding_side="right"
    )
    model = (
        AutoModel.from_pretrained(
            MODEL,
            revision=revision,
            token=False,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )
    for name, (frame, keys, column) in channels.items():
        target = args.output / f"{name}.npy"
        if target.exists() and (args.output / f"{name}.csv").exists():
            continue
        unique, inverse = np.unique(
            frame[column].to_numpy(dtype=object), return_inverse=True
        )
        partial = args.output / f"{name}-partial.npy"
        progress = args.output / f"{name}-progress.json"
        done_file = args.output / f"{name}-done.npy"
        mode = "r+" if partial.exists() else "w+"
        vectors = np.lib.format.open_memmap(
            partial, mode=mode, dtype=np.float32, shape=(len(unique), 1024)
        )
        if done_file.exists():
            done = np.load(done_file)
        else:
            done = np.zeros(len(unique), dtype=bool)
            completed = (
                json.loads(progress.read_text())["completed"]
                if progress.exists()
                else 0
            )
            legacy_order = np.argsort([len(value) for value in unique], kind="stable")
            done[legacy_order[:completed]] = True
            np.save(done_file, done)
        repaired = validate_completed(vectors, done)
        print(
            json.dumps({"channel": name, "repaired_after_interruption": repaired}),
            flush=True,
        )
        remaining = np.flatnonzero(~done)
        lengths = []
        for offset in range(0, len(remaining), 1024):
            rows = remaining[offset : offset + 1024]
            tokenized = tokenizer(
                [PROMPT + unique[row] for row in rows],
                truncation=True,
                max_length=args.max_length,
                return_length=True,
            )
            lengths.extend(tokenized["length"])
        order = remaining[np.argsort(lengths, kind="stable")]
        token_lengths = np.zeros(len(unique), dtype=np.int32)
        token_lengths[remaining] = lengths
        completed = int(done.sum())
        channel_start = time.time()
        initial = completed
        print(
            json.dumps(
                {
                    "channel": name,
                    "already_encoded": completed,
                    "remaining": len(order),
                    "batching": "token length",
                }
            ),
            flush=True,
        )
        for offset in range(0, len(order), args.batch_size):
            rows = order[offset : offset + args.batch_size]
            inputs = tokenizer(
                [PROMPT + unique[row] for row in rows],
                padding="max_length",
                truncation=True,
                max_length=min(
                    args.max_length,
                    max(64, 1 << (int(token_lengths[rows].max()) - 1).bit_length()),
                ),
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                sequence = model(**inputs).last_hidden_state
                positions = inputs["attention_mask"].sum(dim=1) - 1
                hidden = sequence[torch.arange(len(rows), device=device), positions]
                encoded = torch.nn.functional.normalize(hidden.float(), p=2, dim=1)
                batch = encoded.cpu().numpy()
            if not np.isfinite(batch).all():
                raise ValueError("Non-finite embedding")
            vectors[rows] = batch
            if device == "mps":
                torch.mps.synchronize()
            time.sleep(0.05)
            done[rows] = True
            completed = int(done.sum())
            if completed % (args.batch_size * 25) == 0 or completed == len(unique):
                vectors.flush()
                temporary = done_file.with_suffix(".tmp.npy")
                np.save(temporary, done)
                temporary.replace(done_file)
                progress.write_text(json.dumps({"completed": completed}))
                print(
                    json.dumps(
                        {
                            "channel": name,
                            "completed": completed,
                            "total": len(unique),
                            **gpu_usage(torch, device),
                            "texts_per_second": round(
                                (completed - initial)
                                / max(1, time.time() - channel_start),
                                2,
                            ),
                        }
                    ),
                    flush=True,
                )
        np.save(target, vectors[inverse])
        frame[keys].to_csv(args.output / f"{name}.csv", index=False)
        del vectors
        partial.unlink()
        progress.unlink()
        done_file.unlink()
    metadata["elapsed_seconds"] = round(
        time.time() - json.loads(started_file.read_text())["timestamp"], 1
    )
    for name, (frame, _, _) in channels.items():
        vectors = np.load(args.output / f"{name}.npy", mmap_mode="r")
        if vectors.shape != (len(frame), 1024) or not np.isfinite(vectors).all():
            raise ValueError(f"Incomplete encoder output: {name}")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
            raise ValueError(f"Unnormalized encoder output: {name}")
    (args.output / "encoder.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
