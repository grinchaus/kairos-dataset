from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import torch

from lab1_improved import improved_inputs, predict_components


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--embeddings", type=Path, default=Path(".cache/embeddings"))
    ap.add_argument("--model", type=Path, default=Path("results/lab1-v2/model.joblib"))
    ap.add_argument(
        "--output", type=Path, default=Path("results/lab1-v2/reproduced.csv")
    )
    args = ap.parse_args()
    torch.set_num_threads(4)
    model = joblib.load(args.model)
    table, _, store = improved_inputs(args.data, args.embeddings)
    part, predictions = predict_components(model, table, store)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    part[["author", "post"]].assign(label=predictions["selected_blend"]).to_csv(
        args.output, index=False
    )
    try:
        display_output = args.output.resolve().relative_to(Path.cwd())
    except ValueError:
        display_output = args.output.name
    print(display_output)


if __name__ == "__main__":
    main()
