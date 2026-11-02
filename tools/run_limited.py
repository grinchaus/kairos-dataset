from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from filelock import FileLock, Timeout
import psutil

from lab1_resources import resource_environment, resource_violation


def snapshot(process=None):
    memory = psutil.virtual_memory()
    pressure = 1
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            pressure = int(result.stdout.strip())
    rss = 0
    if process is not None:
        try:
            members = [process] + process.children(recursive=True)
        except psutil.Error:
            members = []
        for member in members:
            try:
                rss += member.memory_info().rss
            except psutil.Error:
                pass
    return {
        "total_gb": memory.total / 2**30,
        "available_gb": memory.available / 2**30,
        "rss_gb": rss / 2**30,
        "swap_gb": psutil.swap_memory().used / 2**30,
        "pressure": pressure,
        "disk_free_gb": shutil.disk_usage(ROOT).free / 2**30,
    }


def terminate(process):
    try:
        members = process.children(recursive=True) + [process]
    except psutil.Error:
        members = []
    for member in reversed(members):
        try:
            member.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(members, timeout=5)
    for member in alive:
        try:
            member.kill()
        except psutil.Error:
            pass


def other_compute():
    prefixes = ("embed_lab1", "experiment_", "run_lab1", "predict_lab1")
    for process in psutil.process_iter(["pid", "cwd"]):
        try:
            if process.pid == os.getpid() or process.info["cwd"] != str(ROOT):
                continue
            if any(
                Path(value).name.startswith(prefixes) and value.endswith(".py")
                for value in process.cmdline() or []
            ):
                return process.pid
        except (psutil.Error, OSError):
            continue
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run one lab task with memory monitoring and low CPU priority"
    )
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--rss-gb", type=float, default=10)
    parser.add_argument("--min-available-gb", type=float)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--lock", type=Path, default=ROOT / ".cache/heavy-compute.lock")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A command is required after --")
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(args.lock)
    try:
        context = lock.acquire(timeout=0)
    except Timeout:
        raise SystemExit("Another supervised lab task is already running")
    with context:
        existing = other_compute()
        if existing is not None:
            raise SystemExit(
                f"Another lab computation is still running: pid {existing}"
            )
        baseline = snapshot()
        reserve = (
            args.min_available_gb
            if args.min_available_gb is not None
            else max(8.0, baseline["total_gb"] / 3)
        )
        if baseline["available_gb"] < reserve or baseline["pressure"] != 1:
            raise SystemExit("Host memory is not ready for another computation")
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("a") as handle:
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                env=resource_environment(),
                stdout=handle,
                stderr=subprocess.STDOUT,
                preexec_fn=(lambda: os.nice(15)) if os.name == "posix" else None,
            )
            process = psutil.Process(child.pid)

            def interrupted(signum, frame):
                terminate(process)
                raise SystemExit(128 + signum)

            signal.signal(signal.SIGTERM, interrupted)
            signal.signal(signal.SIGINT, interrupted)
            telemetry = args.log.with_name(args.log.stem + ".resources.jsonl")

            def emit(event):
                line = json.dumps(event)
                with telemetry.open("a") as stream:
                    stream.write(line + "\n")
                print(line, flush=True)

            started = time.time()
            next_report = 0
            warnings = 0
            peak_rss = 0.0
            minimum_available = baseline["available_gb"]
            while child.poll() is None:
                current = snapshot(process)
                peak_rss = max(peak_rss, current["rss_gb"])
                minimum_available = min(minimum_available, current["available_gb"])
                reason = resource_violation(current, baseline, args.rss_gb)
                warnings = warnings + 1 if reason else 0
                if reason and (warnings >= 2 or current["pressure"] >= 4):
                    terminate(process)
                    emit(
                        {"status": "stopped_for_resources", "reason": reason, **current}
                    )
                    raise SystemExit(75)
                if time.time() >= next_report:
                    emit(
                        {
                            "status": "running",
                            "seconds": round(time.time() - started),
                            **{
                                key: (
                                    round(value, 2)
                                    if isinstance(value, float)
                                    else value
                                )
                                for key, value in current.items()
                            },
                        }
                    )
                    next_report = time.time() + 30
                time.sleep(args.interval)
            emit(
                {
                    "status": "finished",
                    "exit_code": child.returncode,
                    "seconds": round(time.time() - started),
                    "sampled_peak_rss_gb": round(peak_rss, 2),
                    "minimum_available_gb": round(minimum_available, 2),
                }
            )
            raise SystemExit(child.returncode)


if __name__ == "__main__":
    main()
