"""
Create deterministic train/validation/test splits from the clean EffiCoder subset.

This replaces prefix slices such as scale1k with disjoint, shuffled splits so
fine-tuning and final evaluation do not share problem IDs.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Iterable

_ROOT = Path(__file__).parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create disjoint dataset splits.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(_ROOT / "data/curated/scale_full/dataset_clean.json"),
        help="Clean dataset JSON to split.",
    )
    parser.add_argument(
        "--benchmark_input",
        type=str,
        default=str(_ROOT / "data/curated/scale_full/benchmark_results.json"),
        help="Optional reference benchmark results to split by dataset_index.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(_ROOT / "data/curated"),
        help="Directory where train/validation/test folders will be written.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--validation_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--remove_legacy_scale1k",
        action="store_true",
        help="Delete data/curated/scale1k after writing the new splits.",
    )
    return parser.parse_args()


def write_json(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(records), f, indent=2, ensure_ascii=False)


def validate_ratios(train_ratio: float, validation_ratio: float, test_ratio: float) -> None:
    total = train_ratio + validation_ratio + test_ratio
    if not 0.999999 <= total <= 1.000001:
        raise ValueError(f"Ratios must sum to 1.0, got {total:.6f}")
    if min(train_ratio, validation_ratio, test_ratio) <= 0:
        raise ValueError("All split ratios must be positive.")


def split_records(records: list[dict], train_ratio: float, validation_ratio: float) -> dict[str, list[dict]]:
    n = len(records)
    train_end = int(n * train_ratio)
    validation_end = train_end + int(n * validation_ratio)
    return {
        "train": records[:train_end],
        "validation": records[train_end:validation_end],
        "test": records[validation_end:],
    }


def split_benchmark_results(benchmark_path: Path, output_root: Path, split_ids: dict[str, set[int]]) -> None:
    if not benchmark_path.exists():
        print(f"Benchmark results not found, skipping: {benchmark_path}")
        return

    with benchmark_path.open("r", encoding="utf-8") as f:
        benchmark_records = json.load(f)

    for split_name, ids in split_ids.items():
        records = [record for record in benchmark_records if record.get("dataset_index") in ids]
        write_json(output_root / split_name / "benchmark_results.json", records)
        print(f"  {split_name:10s} benchmark rows: {len(records)}")


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.validation_ratio, args.test_ratio)

    input_path = Path(args.input)
    output_root = Path(args.output_root)

    with input_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    shuffled = list(records)
    random.Random(args.seed).shuffle(shuffled)
    splits = split_records(shuffled, args.train_ratio, args.validation_ratio)

    split_ids = {}
    print(f"Loaded {len(records)} records from {input_path}")
    for split_name, split_records_list in splits.items():
        split_ids[split_name] = {record["dataset_index"] for record in split_records_list}
        write_json(output_root / split_name / "dataset_clean.json", split_records_list)
        print(f"  {split_name:10s} dataset rows:   {len(split_records_list)}")

    split_benchmark_results(Path(args.benchmark_input), output_root, split_ids)

    legacy_scale1k = output_root / "scale1k"
    if args.remove_legacy_scale1k and legacy_scale1k.exists():
        shutil.rmtree(legacy_scale1k)
        print(f"Removed legacy prefix-slice split: {legacy_scale1k}")

    print("\nDone. Use train for SFT, validation for tuning/checkpoint selection, and test only for final reporting.")


if __name__ == "__main__":
    main()
