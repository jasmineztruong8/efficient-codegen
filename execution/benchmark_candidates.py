"""
benchmark_candidates.py

Short description:
Benchmark passing code candidates by repeatedly executing their tests and
recording runtime statistics such as median, mean, min, and max execution time.

This script expects as input a JSONL file of passing candidates, where each
record includes:
- dataset_index
- candidate_id
- extracted_code
- tests
- task / function_name (optional metadata)

It re-runs each passing candidate multiple times in a fresh namespace and
times only the test execution stage, after first loading the candidate code.

Output:
One JSONL record per benchmarked candidate.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark passing code candidates.")
    parser.add_argument(
        "--input_path",
        type=str,
        default="outputs/passing_candidates_20.jsonl",
        help="Path to passing candidates JSONL",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/benchmarked_candidates_20.jsonl",
        help="Path to save benchmarked candidate results",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=7,
        help="Number of repeated timing runs per candidate",
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=1,
        help="Number of warmup runs before timed runs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: only benchmark the first N candidates",
    )
    return parser.parse_args()


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def run_code_and_tests_once(code: str, tests: str) -> Dict[str, Any]:
    """
    Execute candidate code, then execute tests, and time the tests stage only.

    Returns:
        {
            "passed": bool,
            "elapsed": float | None,
            "error_type": str | None,
            "error_message": str | None,
            "traceback": str | None,
            "stdout": str
        }
    """
    namespace: Dict[str, Any] = {}
    fake_out = io.StringIO()

    try:
        with contextlib.redirect_stdout(fake_out), contextlib.redirect_stderr(fake_out):
            exec(code, namespace)
    except Exception as e:
        return {
            "passed": False,
            "elapsed": None,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "stdout": fake_out.getvalue(),
        }

    try:
        with contextlib.redirect_stdout(fake_out), contextlib.redirect_stderr(fake_out):
            start = time.perf_counter()
            exec(tests, namespace)
            end = time.perf_counter()
    except Exception as e:
        return {
            "passed": False,
            "elapsed": None,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "stdout": fake_out.getvalue(),
        }

    return {
        "passed": True,
        "elapsed": end - start,
        "error_type": None,
        "error_message": None,
        "traceback": None,
        "stdout": fake_out.getvalue(),
    }


def benchmark_one_candidate(
    code: str,
    tests: str,
    warmup_runs: int,
    num_runs: int,
) -> Dict[str, Any]:
    """
    Benchmark a passing candidate with warmup + repeated timed runs.
    """
    # Warmup
    for _ in range(warmup_runs):
        warmup_result = run_code_and_tests_once(code, tests)
        if not warmup_result["passed"]:
            return {
                "benchmark_passed": False,
                "times": [],
                "median_time": None,
                "mean_time": None,
                "min_time": None,
                "max_time": None,
                "std_time": None,
                "error_type": warmup_result["error_type"],
                "error_message": warmup_result["error_message"],
                "traceback": warmup_result["traceback"],
                "stdout": warmup_result["stdout"],
            }

    times: List[float] = []
    last_stdout = ""

    for _ in range(num_runs):
        result = run_code_and_tests_once(code, tests)
        if not result["passed"]:
            return {
                "benchmark_passed": False,
                "times": times,
                "median_time": None,
                "mean_time": None,
                "min_time": None,
                "max_time": None,
                "std_time": None,
                "error_type": result["error_type"],
                "error_message": result["error_message"],
                "traceback": result["traceback"],
                "stdout": result["stdout"],
            }

        times.append(result["elapsed"])
        last_stdout = result["stdout"]

    std_time = statistics.pstdev(times) if len(times) > 1 else 0.0

    return {
        "benchmark_passed": True,
        "times": times,
        "median_time": statistics.median(times),
        "mean_time": statistics.mean(times),
        "min_time": min(times),
        "max_time": max(times),
        "std_time": std_time,
        "error_type": None,
        "error_message": None,
        "traceback": None,
        "stdout": last_stdout,
    }


def main() -> None:
    args = parse_args()
    ensure_parent_dir(args.output_path)

    total = 0
    benchmarked_ok = 0

    with open(args.input_path, "r", encoding="utf-8") as in_f, open(
        args.output_path, "w", encoding="utf-8"
    ) as out_f:
        for idx, line in enumerate(in_f):
            if args.limit is not None and idx >= args.limit:
                break

            record = json.loads(line)
            total += 1

            code = record.get("extracted_code", "")
            tests = record.get("tests", "")

            bench = benchmark_one_candidate(
                code=code,
                tests=tests,
                warmup_runs=args.warmup_runs,
                num_runs=args.num_runs,
            )

            out_record = {
                "dataset_index": record.get("dataset_index"),
                "task": record.get("task"),
                "function_name": record.get("function_name"),
                "candidate_id": record.get("candidate_id"),
                "passed_eval": record.get("passed"),
                "benchmark_passed": bench["benchmark_passed"],
                "num_runs": args.num_runs,
                "warmup_runs": args.warmup_runs,
                "times": bench["times"],
                "median_time": bench["median_time"],
                "mean_time": bench["mean_time"],
                "min_time": bench["min_time"],
                "max_time": bench["max_time"],
                "std_time": bench["std_time"],
                "error_type": bench["error_type"],
                "error_message": bench["error_message"],
                "traceback": bench["traceback"],
                "stdout": bench["stdout"],
                "extracted_code": code,
                "tests": tests,
            }

            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")

            if bench["benchmark_passed"]:
                benchmarked_ok += 1
                print(
                    f"[{total}] dataset_index={record.get('dataset_index')} "
                    f"candidate_id={record.get('candidate_id')} "
                    f"median={bench['median_time']:.6f}"
                )
            else:
                print(
                    f"[{total}] dataset_index={record.get('dataset_index')} "
                    f"candidate_id={record.get('candidate_id')} "
                    f"benchmark failed"
                )

    print("\nDone.")
    print(f"Total passing candidates read: {total}")
    print(f"Successfully benchmarked:      {benchmarked_ok}")
    print(f"Saved to: {args.output_path}")


if __name__ == "__main__":
    main()