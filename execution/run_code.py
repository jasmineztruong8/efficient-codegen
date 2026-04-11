# execution/run_code.py
'''
Executes generated code once per sample.
Used for:
	correctness checking
	Simpler, single-run execution
'''

from __future__ import annotations

import json
import statistics
import textwrap
import time
import traceback
from pathlib import Path


def build_program(code: str, tests: str) -> str:
    return (
        code.rstrip()
        + "\n\n"
        + "# ---- tests ----\n"
        + tests.rstrip()
        + "\n"
    )


def run_once(program: str) -> tuple[bool, str, float]:
    namespace = {}
    start = time.perf_counter()
    try:
        exec(program, namespace, namespace)
        elapsed = time.perf_counter() - start
        return True, "", elapsed
    except Exception:
        elapsed = time.perf_counter() - start
        return False, traceback.format_exc(), elapsed


def benchmark_record(record: dict, repeats: int = 7) -> dict:
    code = record["output"]
    tests = record["tests"]
    program = build_program(code, tests)

    times = []
    errors = []

    for _ in range(repeats):
        ok, err, elapsed = run_once(program)
        if not ok:
            errors.append(err)
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


def main() -> None:
    _root = Path(__file__).parent.parent
    input_path = _root / "data/curated/prototyping/prototype_final_20.json"
    output_path = _root / "data/curated/prototyping/prototype_benchmark_results.json"

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for rec in data:
        result = benchmark_record(rec, repeats=7)
        merged = {
            "dataset_index": rec["dataset_index"],
            "function_name": rec.get("function_name"),
            "score": rec.get("score"),
            "instruction": rec["instruction"],
            **result,
        }
        results.append(merged)
        print(
            f"idx={merged['dataset_index']} "
            f"passed={merged['passed']} "
            f"median={merged['median_runtime_sec']}"
        )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()