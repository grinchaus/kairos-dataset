from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from threadpoolctl import threadpool_limits
import torch

from lab1_target70 import predict_target, target_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--embeddings", type=Path, default=Path(".cache/embeddings"))
    parser.add_argument(
        "--qwen-embeddings", type=Path, default=Path(".cache/embeddings-qwen")
    )
    parser.add_argument(
        "--vision-embeddings", type=Path, default=Path(".cache/embeddings-vision")
    )
    parser.add_argument(
        "--model", type=Path, default=Path("results/lab1-v3/model.joblib")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/lab1-v3/reproduced.csv")
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        package = joblib.load(args.model)
        table, _, _, e5, qwen, vision = target_inputs(
            args.data,
            args.embeddings,
            args.qwen_embeddings,
            package["selection"],
            args.vision_embeddings,
        )
        test = table[table.split == "test"].reset_index(drop=True)
        prediction, _ = predict_target(package, test, e5, qwen, vision)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        test[["author", "post"]].assign(label=prediction).to_csv(
            args.output, index=False
        )
    print(args.output, flush=True)


if __name__ == "__main__":
    from lab1_resources import ensure_managed

    ensure_managed(__file__)
    main()
