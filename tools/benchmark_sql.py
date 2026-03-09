from __future__ import annotations

"""
Repeatable SQL benchmark runner for nn-dataset + sibling nn-gpt.

Sample output:

# SQL Benchmark

- Dataset root: C:\...\nn-dataset
- NNGPT root: C:\...\nn-gpt
- Dataset commit: c03c2be5b61d57b5cbb0a0adb98b4a3011a54a8d
- NNGPT commit: <hash>
- Python: 3.11.9 (main, ...)
- Python executable: C:\...\nn-gpt\.venv\Scripts\python.exe
- Platform: Windows-11-10.0.22631-SP0
- CPU: Intel64 Family 6 Model ...
- Logical CPUs: 16
- Command: C:\...\python.exe test.py

## Runs

| Run | Status | Wall (s) | correctness suite | test_anchor_band_correctness_all_bands |
| --- | --- | ---: | ---: | ---: |
| 1 | PASS | 28.4142 | 26.8600 | 3.9958 |
| 2 | PASS | 27.9911 | 26.4210 | 3.8124 |

## Summary

| Metric | Min (s) | Avg (s) | Max (s) | StdDev (s) |
| --- | ---: | ---: | ---: | ---: |
| wall_time | 27.9911 | 28.2027 | 28.4142 | 0.2116 |
| correctness_suite | 26.4210 | 26.6405 | 26.8600 | 0.2195 |
"""

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NNGPT_ROOT = (REPO_ROOT.parent / "nn-gpt").resolve()
TIMING_RE = r"^(?P<name>[A-Za-z0-9_]+) took (?P<secs>\d+(?:\.\d+)?) seconds$"
CORRECTNESS_RE = r"^\[INFO\] correctness suite took (?P<secs>\d+(?:\.\d+)?)s$"


@dataclass
class RunResult:
    index: int
    returncode: int
    wall_time: float
    metrics: dict[str, float]
    passed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--nn-gpt-root", type=Path, default=DEFAULT_NNGPT_ROOT)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reset-db", action="store_true")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "benchmark_results.json")
    return parser.parse_args()


def run_checked(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    return proc


def run_git(repo: Path, *args: str) -> str:
    return run_checked(["git", *args], cwd=repo).stdout.strip()


def install_editable(dataset_root: Path) -> None:
    run_checked(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            str(dataset_root),
            "--no-deps",
            "--no-build-isolation",
        ],
        cwd=dataset_root,
    )


def remove_db(db_path: Path) -> None:
    if not db_path.exists():
        return
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "rmdir", "/s", "/q", str(db_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return
    shutil.rmtree(db_path)


def cpu_info() -> str:
    value = platform.processor().strip()
    if value:
        return value
    return os.environ.get("PROCESSOR_IDENTIFIER", "unknown")


def parse_metrics(output: str) -> dict[str, float]:
    import re

    metrics: dict[str, float] = {}
    timing_re = re.compile(TIMING_RE)
    correctness_re = re.compile(CORRECTNESS_RE)

    for line in output.splitlines():
        match = timing_re.match(line.strip())
        if match:
            metrics[match.group("name")] = float(match.group("secs"))
            continue
        match = correctness_re.match(line.strip())
        if match:
            metrics["correctness_suite"] = float(match.group("secs"))
    return metrics


def benchmark_once(index: int, nn_gpt_root: Path) -> tuple[RunResult, str]:
    cmd = [sys.executable, "test.py"]
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=nn_gpt_root, text=True, capture_output=True)
    wall_time = time.perf_counter() - start
    combined = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    metrics = parse_metrics(combined)
    passed = proc.returncode == 0 and "ALL TESTS PASSED" in combined
    return (
        RunResult(
            index=index,
            returncode=proc.returncode,
            wall_time=wall_time,
            metrics=metrics,
            passed=passed,
        ),
        combined,
    )


def stats(values: Iterable[float]) -> dict[str, float]:
    seq = list(values)
    if not seq:
        return {"min": 0.0, "avg": 0.0, "max": 0.0, "stddev": 0.0}
    return {
        "min": min(seq),
        "avg": statistics.mean(seq),
        "max": max(seq),
        "stddev": statistics.pstdev(seq),
    }


def write_results(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_markdown(metadata: dict[str, str], runs: list[RunResult]) -> None:
    metric_names: list[str] = ["wall_time"]
    for run in runs:
        for name in run.metrics:
            if name not in metric_names:
                metric_names.append(name)

    print("# SQL Benchmark")
    print()
    for key, value in metadata.items():
        print(f"- {key}: {value}")
    print()

    print("## Runs")
    print()
    headers = ["Run", "Status", "Wall (s)"] + metric_names[1:]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---", "---", "---:"] + ["---:"] * (len(headers) - 3)) + " |")
    for run in runs:
        row = [
            str(run.index),
            "PASS" if run.passed else f"FAIL ({run.returncode})",
            f"{run.wall_time:.4f}",
        ]
        for name in metric_names[1:]:
            value = run.metrics.get(name)
            row.append("" if value is None else f"{value:.4f}")
        print("| " + " | ".join(row) + " |")
    print()

    print("## Summary")
    print()
    print("| Metric | Min (s) | Avg (s) | Max (s) | StdDev (s) |")
    print("| --- | ---: | ---: | ---: | ---: |")
    all_values: dict[str, list[float]] = {"wall_time": [run.wall_time for run in runs]}
    for name in metric_names[1:]:
        all_values[name] = [run.metrics[name] for run in runs if name in run.metrics]
    for name, values in all_values.items():
        summary = stats(values)
        print(
            f"| {name} | {summary['min']:.4f} | {summary['avg']:.4f} | "
            f"{summary['max']:.4f} | {summary['stddev']:.4f} |"
        )


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    nn_gpt_root = args.nn_gpt_root.resolve()

    install_editable(dataset_root)
    if args.reset_db:
        remove_db(nn_gpt_root / "db")

    warmups: list[dict[str, object]] = []
    for index in range(1, args.warmup + 1):
        result, output = benchmark_once(index=index, nn_gpt_root=nn_gpt_root)
        warmups.append(
            {
                "index": index,
                "returncode": result.returncode,
                "wall_time": result.wall_time,
                "metrics": result.metrics,
                "passed": result.passed,
            }
        )
        if not result.passed:
            raise RuntimeError(f"Warmup run {index} failed.\n{output}")

    runs: list[RunResult] = []
    captured: list[dict[str, object]] = []
    for offset in range(1, args.runs + 1):
        result, output = benchmark_once(index=offset, nn_gpt_root=nn_gpt_root)
        runs.append(result)
        captured.append(
            {
                "index": offset,
                "returncode": result.returncode,
                "wall_time": result.wall_time,
                "metrics": result.metrics,
                "passed": result.passed,
                "stdout": output,
            }
        )
        if not result.passed:
            raise RuntimeError(f"Benchmark run {offset} failed.\n{output}")

    metadata = {
        "Dataset root": str(dataset_root),
        "NNGPT root": str(nn_gpt_root),
        "Dataset commit": run_git(dataset_root, "rev-parse", "HEAD"),
        "NNGPT commit": run_git(nn_gpt_root, "rev-parse", "HEAD"),
        "Python": sys.version.replace("\n", " "),
        "Python executable": sys.executable,
        "Platform": platform.platform(),
        "CPU": cpu_info(),
        "Logical CPUs": str(os.cpu_count() or 0),
        "Command": f"{sys.executable} test.py",
    }

    payload = {
        "metadata": metadata,
        "warmup": warmups,
        "runs": captured,
    }
    write_results(args.output.resolve(), payload)
    print_markdown(metadata, runs)
    print()
    print(f"Results written to: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
