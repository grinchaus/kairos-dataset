import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from lab1_resources import resource_environment, resource_violation


class ResourceLimitTests(unittest.TestCase):
    def snapshot(self):
        return {
            "rss_gb": 1.0,
            "available_gb": 24.0,
            "total_gb": 48.0,
            "swap_gb": 0.0,
            "pressure": 1,
            "disk_free_gb": 30.0,
        }

    def test_healthy_host_can_continue(self):
        baseline = self.snapshot()
        self.assertIsNone(resource_violation(baseline, baseline, 10))

    def test_pressure_swap_and_rss_each_stop_computation(self):
        baseline = self.snapshot()
        for key, value in [
            ("rss_gb", 11.0),
            ("pressure", 2),
            ("swap_gb", 1.1),
            ("available_gb", 6.0),
        ]:
            current = {**baseline, key: value}
            self.assertIsNotNone(resource_violation(current, baseline, 10))

    def test_gpu_guard_cannot_be_disabled_by_inherited_environment(self):
        with patch.dict(
            "os.environ",
            {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0", "OMP_NUM_THREADS": "64"},
        ):
            environment = resource_environment()
        self.assertEqual(environment["PYTORCH_MPS_HIGH_WATERMARK_RATIO"], "0.12")
        self.assertEqual(environment["OMP_NUM_THREADS"], "2")

    def test_watchdog_terminates_child_above_rss_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            result = subprocess.run(
                [
                    sys.executable,
                    "tools/run_limited.py",
                    "--log",
                    str(folder / "child.log"),
                    "--lock",
                    str(folder / "lock"),
                    "--rss-gb",
                    "0.02",
                    "--min-available-gb",
                    "0",
                    "--interval",
                    "0.1",
                    "--",
                    sys.executable,
                    "-c",
                    "import time; data=bytearray(40*1024*1024); time.sleep(10)",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 75, result.stdout + result.stderr)
            events = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(events[-1]["reason"], "process_memory_limit")


if __name__ == "__main__":
    unittest.main()
