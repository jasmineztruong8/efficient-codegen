# execution/benchmark.py
'''
Executes code in isolated subprocesses with timing.
Measures:
	runtime (median, multiple runs)
	pass/fail
	Purpose: core benchmarking script
'''

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def build_program(code: str, tests: str) -> str:
    return (
        code.rstrip()
        + "\n\n"
        + "# ---- tests ----\n"
        + tests.rstrip()
        + "\nprint('__PASS__')\n"
    )


def run_once_subprocess(program: str, timeout_sec: float = 3.0) -> tuple[bool, str, float]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        script_path = f.name

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        elapsed = time.perf_counter() - start

        if proc.returncode != 0:
            return False, proc.stderr, elapsed
        if "__PASS__" not in proc.stdout:
            return False, "Did not observe pass marker.", elapsed

        return True, "", elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        return False, "TimeoutExpired", elapsed


def benchmark_record(record: dict, repeats: int = 7) -> dict:
    program = build_program(record["output"], record["tests"])
    times = []

    for _ in range(repeats):
        ok, err, elapsed = run_once_subprocess(program)
        if not ok:
            return {
                "passed": False,
                "median_runtime_sec": None,
                "all_runtimes_sec": times,
                "error": err,
            }
        times.append(elapsed)

    return {
        "passed": True,
        "median_runtime_sec": statistics.median(times),
        "all_runtimes_sec": times,
        "error": None,
    }
    
def parse_args():
    import argparse
    _root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to candidates json")
    parser.add_argument("--output", type=str, required=True, help="Path to write benchmark results json")
    parser.add_argument("--repeats", type=int, default=7, help="Number of timed runs per candidate")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    total = len(data)
    for i, rec in enumerate(data, 1):
        result = benchmark_record(rec, repeats=args.repeats)
        merged = {
            "dataset_index": rec["dataset_index"],
            "instruction": rec["instruction"],
            **result,
        }
        results.append(merged)

        print(
            f"[{i}/{total}] idx={rec['dataset_index']} "
            f"passed={result['passed']} "
            f"median={result['median_runtime_sec']}"
        )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()