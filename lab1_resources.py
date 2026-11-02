from __future__ import annotations

import os
from pathlib import Path
import sys


def ensure_managed(script):
    if os.environ.get("LAB1_RESOURCE_MANAGED") == "1" or "--help" in sys.argv:
        return
    root = Path(script).resolve().parent
    launcher = root / "tools/run_limited.py"
    os.execv(
        sys.executable,
        [
            sys.executable,
            str(launcher),
            "--log",
            str(root / ".cache" / f"supervised-{Path(script).stem}.log"),
            "--",
            sys.executable,
            str(Path(script).resolve()),
            *sys.argv[1:],
        ],
    )


def resource_environment():
    environment = dict(os.environ)
    for name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        environment[name] = "2"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.12"
    environment["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.08"
    environment["LAB1_RESOURCE_MANAGED"] = "1"
    return environment


def configure_torch(torch):
    torch.set_num_threads(2)
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(0.12)
    elif torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.5)


def validate_completed(vectors, done):
    import numpy as np

    positions = np.flatnonzero(done)
    repaired = 0
    for start in range(0, len(positions), 1024):
        rows = positions[start : start + 1024]
        values = vectors[rows]
        valid = np.isfinite(values).all(axis=1) & np.isclose(
            np.linalg.norm(values, axis=1), 1, atol=1e-4
        )
        done[rows[~valid]] = False
        repaired += int((~valid).sum())
    return repaired


def gpu_usage(torch, device):
    if device != "mps":
        return {}
    driver = torch.mps.driver_allocated_memory()
    if driver > torch.mps.recommended_max_memory() * 0.2:
        raise RuntimeError("GPU driver memory reserve exceeded; stopping this process")
    return {
        "gpu_driver_gb": round(driver / 2**30, 2),
        "gpu_tensor_gb": round(torch.mps.current_allocated_memory() / 2**30, 2),
    }


def resource_violation(snapshot, baseline, max_rss_gb):
    if snapshot["rss_gb"] > max_rss_gb:
        return "process_memory_limit"
    if snapshot["available_gb"] < max(4.0, snapshot["total_gb"] * 0.15):
        return "system_memory_reserve"
    if snapshot["pressure"] >= 2:
        return "system_memory_pressure"
    if snapshot["swap_gb"] - baseline["swap_gb"] > 1:
        return "swap_growth"
    if snapshot["disk_free_gb"] < 3:
        return "disk_reserve"
    return None
