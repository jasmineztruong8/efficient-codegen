"""
select_training_data.py

Build fine-tuning datasets from benchmarked candidates.

Reads benchmarked_candidates.jsonl (output of execution/benchmark_candidates.py)
and dataset_clean.json (for instruction/input lookup), and writes two files:

  runtime_aware.jsonl  — fastest-correct candidate per problem (main training set)
  control.jsonl        — first-correct candidate per problem (ablation baseline)

Each output line is a JSON record with a "messages" field ready for SFTTrainer.

Use --min_speedup_ratio to restrict runtime_aware.jsonl to problems where the
fastest candidate is meaningfully faster than the first-correct one (e.g. 1.5
means the fastest must be at least 1.5x faster). This sharpens the training
signal at the cost of a smaller dataset. control.jsonl is always written for
all problems regardless of this filter.

Usage:
  python training/select_training_data.py \
    --candidates_path outputs/benchmarked_candidates.jsonl \
    --dataset_path data/curated/train/dataset_clean.json \
    --output_dir training/data

  # Stronger signal: only problems with >=1.5x speedup
  python training/select_training_data.py \
    --candidates_path outputs/benchmarked_candidates.jsonl \
    --dataset_path data/curated/train/dataset_clean.json \
    --output_dir training/data \
    --min_speedup_ratio 1.5
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
    parser.add_argument(
        "--min_speedup_ratio",
        type=float,
        default=1.0,
        help=(
            "Minimum speedup ratio (fastest / first-correct) required to include a problem "
            "in runtime_aware.jsonl. Default 1.0 keeps all problems. Use e.g. 1.5 to keep "
            "only problems where the fastest candidate is at least 1.5x faster than the "
            "first-correct candidate, sharpening the training signal."
        ),
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
    written_ra = 0
    written_ctrl = 0
    filtered_by_speedup = 0
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

            fastest = min(candidates, key=lambda c: c["median_time"])
            first = min(candidates, key=lambda c: c["candidate_id"])

            # Control: always written, no speed filter
            ctrl_f.write(json.dumps({
                "dataset_index": dataset_index,
                "candidate_id": first["candidate_id"],
                "median_time": first["median_time"],
                "messages": build_messages(instruction, input_content, first["extracted_code"]),
            }, ensure_ascii=False) + "\n")
            written_ctrl += 1

            # Compute speedup and apply threshold filter for runtime-aware set
            speedup = (
                first["median_time"] / fastest["median_time"]
                if first["median_time"] and fastest["median_time"] and fastest["median_time"] > 0
                else 1.0
            )
            speedup_ratios.append(speedup)

            if speedup < args.min_speedup_ratio:
                filtered_by_speedup += 1
                continue

            ra_f.write(json.dumps({
                "dataset_index": dataset_index,
                "candidate_id": fastest["candidate_id"],
                "median_time": fastest["median_time"],
                "speedup_vs_first": round(speedup, 4),
                "messages": build_messages(instruction, input_content, fastest["extracted_code"]),
            }, ensure_ascii=False) + "\n")
            written_ra += 1

    print(f"\nDone.")
    print(f"Control dataset:               {written_ctrl} problems → {control_path}")
    print(f"Runtime-aware dataset:         {written_ra} problems → {runtime_aware_path}")
    if args.min_speedup_ratio > 1.0:
        print(f"  (filtered out {filtered_by_speedup} problems below {args.min_speedup_ratio}x speedup threshold)")
    print(f"Problems skipped (not in dataset): {skipped}")

    if speedup_ratios:
        print(f"\nSpeedup of fastest vs. first-correct candidate (all problems):")
        print(f"  Median speedup:  {statistics.median(speedup_ratios):.2f}x")
        print(f"  Mean speedup:    {statistics.mean(speedup_ratios):.2f}x")
        print(f"  Max speedup:     {max(speedup_ratios):.2f}x")
        print(f"  Problems with >1.5x speedup: {sum(1 for r in speedup_ratios if r >= 1.5)}")
        print(f"  Problems with >2x speedup:   {sum(1 for r in speedup_ratios if r >= 2.0)}")


if __name__ == "__main__":
    main()
