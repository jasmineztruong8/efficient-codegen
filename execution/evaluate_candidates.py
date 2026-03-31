"""
evaluate_candidates.py

Short description:
Evaluate generated code candidates against the provided tests.
This script reads one JSONL record per problem, where each record contains
a list of candidate generations under the "candidates" field.

For each candidate, it:
- extracts Python code from the model output
- executes the candidate in an isolated namespace
- runs the associated tests
- records pass/fail and error information

Output:
One JSONL record per candidate evaluation.
"""

from __future__ import annotations

import argparse
import io
import contextlib
import json
import re
import traceback
from math import comb
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated code candidates.")
    parser.add_argument(
        "--input_path",
        type=str,
        default="outputs/generated_candidates_20.jsonl",
        help="Path to generated candidates JSONL",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/evaluated_candidates_20.jsonl",
        help="Path to save evaluated candidate results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: only evaluate the first N problem records",
    )
    return parser.parse_args()


def extract_code(text: str) -> str:
    """
    Extract Python code from a generated candidate.

    Handles:
    - fenced code blocks: ```python ... ```
    - plain text generations
    """
    text = text.strip()

    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    return text


def safe_snippet(text: str, max_len: int = 500) -> str:
    text = text.strip()
    return text if len(text) <= max_len else text[:max_len] + " ...[truncated]"


def evaluate_one_candidate(code: str, tests: str):
    namespace = {}
    fake_out = io.StringIO()

    try:
        with contextlib.redirect_stdout(fake_out), contextlib.redirect_stderr(fake_out):
            exec(code, namespace)
    except Exception as e:
        return {
            "passed": False,
            "stage": "code_exec",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "stdout": fake_out.getvalue(),
        }

    try:
        with contextlib.redirect_stdout(fake_out), contextlib.redirect_stderr(fake_out):
            exec(tests, namespace)
    except Exception as e:
        return {
            "passed": False,
            "stage": "tests_exec",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "stdout": fake_out.getvalue(),
        }

    return {
        "passed": True,
        "stage": "tests_exec",
        "error_type": None,
        "error_message": None,
        "traceback": None,
        "stdout": fake_out.getvalue(),
    }

def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    ensure_parent_dir(args.output_path)

    num_problem_records = 0
    num_candidates = 0
    num_passed = 0
    # track per-problem pass counts for Pass@k estimation
    problem_pass_counts: dict[int, dict] = {}  # idx -> {n, c}

    with open(args.input_path, "r", encoding="utf-8") as in_f, open(
        args.output_path, "w", encoding="utf-8"
    ) as out_f:
        for problem_idx, line in enumerate(in_f):
            if args.limit is not None and problem_idx >= args.limit:
                break

            record = json.loads(line)
            num_problem_records += 1

            dataset_index = record.get("dataset_index")
            task = record.get("task")
            function_name = record.get("function_name")
            tests = record.get("tests", "")
            candidates = record.get("candidates", [])

            if not isinstance(candidates, list):
                print(f"Warning: dataset_index={dataset_index} has non-list candidates field")
                continue

            problem_pass_counts[dataset_index] = {"n": len(candidates), "c": 0}

            for candidate_id, raw_candidate in enumerate(candidates):
                num_candidates += 1

                raw_text = raw_candidate if isinstance(raw_candidate, str) else str(raw_candidate)
                code = extract_code(raw_text)

                result = evaluate_one_candidate(code=code, tests=tests)

                if result["passed"]:
                    num_passed += 1
                    problem_pass_counts[dataset_index]["c"] += 1

                out_record = {
                    "dataset_index": dataset_index,
                    "task": task,
                    "function_name": function_name,
                    "candidate_id": candidate_id,
                    "passed": result["passed"],
                    "stage": result["stage"],
                    "error_type": result["error_type"],
                    "error_message": result["error_message"],
                    "traceback": result["traceback"],
                    "raw_candidate_preview": safe_snippet(raw_text, max_len=400),
                    "extracted_code": code,
                    "tests": tests,
                }

                out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")

            print(
                f"[{num_problem_records}] dataset_index={dataset_index} "
                f"candidates={len(candidates)}"
            )

    # Compute Pass@1 using the unbiased estimator: 1 - C(n-c, k) / C(n, k)
    k = 1
    pass_at_1_scores = []
    for stats in problem_pass_counts.values():
        n, c = stats["n"], stats["c"]
        score = 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)
        pass_at_1_scores.append(score)
    pass_at_1 = sum(pass_at_1_scores) / len(pass_at_1_scores) if pass_at_1_scores else 0.0

    print("\nDone.")
    print(f"Problem records processed: {num_problem_records}")
    print(f"Candidates evaluated:     {num_candidates}")
    print(f"Candidates passed:        {num_passed}")
    print(f"Pass@1 (unbiased):        {pass_at_1:.3f}")


if __name__ == "__main__":
    main()