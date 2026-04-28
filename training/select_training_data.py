"""
select_training_data.py

Build fine-tuning datasets from benchmarked candidates.

Reads benchmarked_candidates.jsonl (output of execution/benchmark_candidates.py)
and dataset_clean.json (for instruction/input lookup), and writes two files:

  runtime_aware.jsonl  — fastest-correct candidate per problem (main training set)
  control.jsonl        — first-correct candidate per problem (ablation baseline)

Each output line is a JSON record with a "messages" field ready for SFTTrainer.

Usage:
  python training/select_training_data.py \
    --candidates_path outputs/benchmarked_candidates.jsonl \
    --dataset_path data/curated/train/dataset_clean.json \
    --output_dir training/data
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

# Must match SYSTEM_PROMPT in generation/generate_candidates.py and training/train.py
# so that the format seen during training matches what is used at inference time.
SYSTEM_PROMPT = (
    "Write a correct Python solution optimized for fast execution time. "
    "Use efficient algorithms and data structures to minimize runtime. "
    "Return only the code with no explanation."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build runtime-aware and control SFT datasets.")
    parser.add_argument(
        "--candidates_path",
        type=str,
        required=True,
        help="Path to benchmarked_candidates.jsonl",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset_clean.json (for instruction/input lookup by dataset_index)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training/data",
        help="Directory to write runtime_aware.jsonl and control.jsonl",
    )
    return parser.parse_args()


def build_messages(instruction: str, input_content: str, code: str) -> list:
    content = instruction
    if input_content:
        content += f"\n\nInput:\n{input_content}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
        {"role": "assistant", "content": code},
    ]


def main() -> None:
    args = parse_args()

    print(f"Loading dataset from {args.dataset_path}")
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    dataset_by_index = {p["dataset_index"]: p for p in dataset}
    print(f"Loaded {len(dataset_by_index)} problems from dataset")

    print(f"Loading benchmarked candidates from {args.candidates_path}")
    by_problem: dict[int, list] = defaultdict(list)
    total_read = 0
    with open(args.candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            total_read += 1
            if record.get("benchmark_passed"):
                by_problem[record["dataset_index"]].append(record)

    print(f"Read {total_read} candidate records")
    print(f"Problems with ≥1 passing benchmarked candidate: {len(by_problem)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_aware_path = output_dir / "runtime_aware.jsonl"
    control_path = output_dir / "control.jsonl"

    skipped = 0
    written = 0
    speedup_ratios = []

    with open(runtime_aware_path, "w", encoding="utf-8") as ra_f, \
         open(control_path, "w", encoding="utf-8") as ctrl_f:

        for dataset_index, candidates in by_problem.items():
            problem = dataset_by_index.get(dataset_index)
            if problem is None:
                skipped += 1
                continue

            instruction = problem["instruction"]
            input_content = problem.get("input", "")

            # Runtime-aware: select the candidate with the lowest median execution time
            fastest = min(candidates, key=lambda c: c["median_time"])
            ra_f.write(json.dumps({
                "dataset_index": dataset_index,
                "candidate_id": fastest["candidate_id"],
                "median_time": fastest["median_time"],
                "messages": build_messages(instruction, input_content, fastest["extracted_code"]),
            }, ensure_ascii=False) + "\n")

            # Control: select the first passing candidate (no speed filter)
            first = min(candidates, key=lambda c: c["candidate_id"])
            ctrl_f.write(json.dumps({
                "dataset_index": dataset_index,
                "candidate_id": first["candidate_id"],
                "median_time": first["median_time"],
                "messages": build_messages(instruction, input_content, first["extracted_code"]),
            }, ensure_ascii=False) + "\n")

            # Track speedup of fastest vs first for stats
            if first["median_time"] and first["median_time"] > 0:
                speedup_ratios.append(first["median_time"] / fastest["median_time"])

            written += 1

    print(f"\nDone.")
    print(f"Problems written:              {written}")
    print(f"Problems skipped (not in dataset): {skipped}")
    print(f"Runtime-aware dataset:         {runtime_aware_path}")
    print(f"Control dataset:               {control_path}")

    if speedup_ratios:
        print(f"\nSpeedup of fastest vs. first-correct candidate:")
        print(f"  Median speedup:  {statistics.median(speedup_ratios):.2f}x")
        print(f"  Mean speedup:    {statistics.mean(speedup_ratios):.2f}x")
        print(f"  Max speedup:     {max(speedup_ratios):.2f}x")
        print(f"  Problems with >2x speedup: {sum(1 for r in speedup_ratios if r > 2)}")


if __name__ == "__main__":
    main()
