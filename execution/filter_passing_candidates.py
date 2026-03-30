"""
filter_passing_candidates.py

Short description:
Read evaluated candidate results and keep only the passing candidates.
This creates a smaller artifact for downstream runtime benchmarking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter passing evaluated candidates.")
    parser.add_argument(
        "--input_path",
        type=str,
        default="outputs/evaluated_candidates_20.jsonl",
        help="Path to evaluated candidates JSONL",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/passing_candidates_20.jsonl",
        help="Path to save passing candidates JSONL",
    )
    return parser.parse_args()


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    ensure_parent_dir(args.output_path)

    total = 0
    passed = 0

    with open(args.input_path, "r", encoding="utf-8") as in_f, open(
        args.output_path, "w", encoding="utf-8"
    ) as out_f:
        for line in in_f:
            total += 1
            record: Dict[str, Any] = json.loads(line)

            if record.get("passed") is True:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                passed += 1

    print("Done.")
    print(f"Total evaluated candidates: {total}")
    print(f"Passing candidates kept:   {passed}")
    print(f"Saved to: {args.output_path}")


if __name__ == "__main__":
    main()