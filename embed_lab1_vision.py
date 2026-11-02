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
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizerFast,
    CLIPVisionModelWithProjection,
)

from embed_lab1 import fingerprint, history_texts
from tools.fetch_core_images import image_candidates, valid_file
from train_lab1 import checked_table

MODEL = "openai/clip-vit-base-patch32"
REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
GENRES = [
    "a business presentation slide",
    "a certificate or diploma",
    "a group photo at a business conference",
    "a screenshot of a website",
    "an infographic",
    "an image containing motivational text",
    "a personal portrait photograph",
    "a product advertisement",
    "an event invitation poster",
    "a landscape photograph",
    "a celebration at work",
    "a chart or graph",
    "a photograph of a product",
    "a logo of a company",
]


def vision_mapping(data, wait):
    cached = Path(".cache/vision-input.csv")
    if wait:
        while not cached.exists():
            time.sleep(10)
        mapping = pd.read_csv(cached)
    else:
        names = {
            path.relative_to(data).as_posix()
            for path in (data / "media").glob("*/*")
            if path.is_file()
        }
        if not names:
            raise ValueError(
                "Core media are required for image encoding; see tools/fetch_core_images.py"
            )
        mapping = image_candidates(data, names)
    table, _ = checked_table(data)
    history = history_texts(data, table)
    keys = pd.concat(
        [
            table[["author", "post", "clean_text", "posted_at", "hist_n"]].rename(
                columns={"hist_n": "history_order"}
            ),
            history[["author", "post", "clean_text", "posted_at", "history_order"]],
        ],
        ignore_index=True,
    )
    result = mapping.merge(keys, on=["author", "post"], validate="one_to_one")
    if len(result) != len(mapping):
        raise ValueError("Some selected images have no own-post text")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path(".cache/embeddings-vision"))
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--wait-for-download", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_torch(torch)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    dtype = torch.float32 if device == "cpu" else torch.float16
    mapping = vision_mapping(args.data, args.wait_for_download)
    files = sorted(mapping.file.unique())
    metadata = {
        "model": MODEL,
        "revision": args.revision,
        "dimension": 512,
        "image_size": 224,
        "caption_max_length": 77,
        "normalized": True,
        "input_sha256": fingerprint(
            mapping, ["author", "post", "file", "kind", "clean_text"]
        ),
        "mapping_sha256": fingerprint(mapping, ["author", "post", "file", "kind"]),
        "genres": GENRES,
        "posts": len(mapping),
        "images": len(files),
        "device": device,
        "dtype": str(dtype),
        "torch": torch.__version__,
    }
    pending = args.output / "pending.json"
    if pending.exists() and json.loads(pending.read_text()) != metadata:
        raise ValueError("Vision encoder inputs changed")
    pending.write_text(json.dumps(metadata, indent=2))
    if (args.output / "encoder.json").exists():
        print("Matching complete image embeddings already exist", flush=True)
        return
    mapping[["author", "post", "file", "kind", "posted_at", "history_order"]].to_csv(
        args.output / "keys.csv", index=False
    )
    pd.DataFrame({"file": files}).to_csv(args.output / "files.csv", index=False)
    started = time.time()
    print(f"Loading local CLIP encoders on {device}", flush=True)
    if (
        not (args.output / "captions.npy").exists()
        or not (args.output / "genres.npy").exists()
    ):
        tokenizer = CLIPTokenizerFast.from_pretrained(
            MODEL, revision=args.revision, token=False
        )
        text_model = (
            CLIPTextModelWithProjection.from_pretrained(
                MODEL,
                revision=args.revision,
                token=False,
                use_safetensors=False,
                weights_only=True,
                torch_dtype=dtype,
                trust_remote_code=False,
            )
            .to(device)
            .eval()
        )

        def encode_texts(texts, batch_size=64):
            vectors = []
            for offset in range(0, len(texts), batch_size):
                inputs = tokenizer(
                    texts[offset : offset + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                ).to(device)
                with torch.inference_mode():
                    values = text_model(**inputs).text_embeds.float()
                    vectors.append(
                        torch.nn.functional.normalize(values, dim=1).cpu().numpy()
                    )
            return np.concatenate(vectors)

        np.save(args.output / "genres.npy", encode_texts(GENRES))
        if not (args.output / "captions.npy").exists():
            texts, inverse = np.unique(
                mapping.clean_text.to_numpy(dtype=object), return_inverse=True
            )
            print(f"Encoding captions: {len(texts)} distinct texts", flush=True)
            np.save(args.output / "captions.npy", encode_texts(texts.tolist())[inverse])
        del text_model
    if device == "mps":
        torch.mps.empty_cache()
    processor = CLIPImageProcessor.from_pretrained(
        MODEL, revision=args.revision, token=False
    )
    model = (
        CLIPVisionModelWithProjection.from_pretrained(
            MODEL,
            revision=args.revision,
            token=False,
            use_safetensors=False,
            weights_only=True,
            torch_dtype=dtype,
            trust_remote_code=False,
        )
        .to(device)
        .eval()
    )
    target = args.output / "images.npy"
    output = np.lib.format.open_memmap(
        target,
        mode="r+" if target.exists() else "w+",
        dtype=np.float32,
        shape=(len(files), 512),
    )
    progress = args.output / "completed.npy"
    done = np.load(progress) if progress.exists() else np.zeros(len(files), dtype=bool)
    repaired = validate_completed(output, done)
    print(
        json.dumps(
            {
                "images_restored": int(done.sum()),
                "repaired_after_interruption": repaired,
            }
        ),
        flush=True,
    )
    while not done.all():
        available = [
            index
            for index in np.flatnonzero(~done)
            if valid_file(args.data / files[index])
        ]
        if not available:
            if not args.wait_for_download:
                raise ValueError(
                    "Selected image content is missing or has an invalid SHA256"
                )
            time.sleep(15)
            continue
        for offset in range(0, len(available), args.batch_size):
            rows = available[offset : offset + args.batch_size]
            tensors = []
            for row in rows:
                with Image.open(args.data / files[row]) as image:
                    tensors.append(
                        processor(images=image.convert("RGB"), return_tensors="pt")[
                            "pixel_values"
                        ]
                    )
            pixels = torch.cat(tensors).to(device=device, dtype=dtype)
            with torch.inference_mode():
                values = model(pixel_values=pixels).image_embeds.float()
                vectors = torch.nn.functional.normalize(values, dim=1).cpu().numpy()
            if not np.isfinite(vectors).all():
                raise ValueError("Non-finite image embedding")
            output[rows] = vectors
            done[rows] = True
            if device == "mps":
                torch.mps.synchronize()
            time.sleep(0.05)
        output.flush()
        temporary = progress.with_suffix(".tmp.npy")
        np.save(temporary, done)
        temporary.replace(progress)
        print(
            json.dumps(
                {
                    "images_encoded": int(done.sum()),
                    "total": len(done),
                    **gpu_usage(torch, device),
                }
            ),
            flush=True,
        )
        if not done.all():
            time.sleep(5)
    if not np.isfinite(output).all() or not np.allclose(
        np.linalg.norm(output, axis=1), 1, atol=1e-4
    ):
        raise ValueError("Image embeddings are incomplete")
    metadata["elapsed_seconds"] = round(time.time() - started, 1)
    (args.output / "encoder.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
