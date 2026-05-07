"""
grid_search.py

Lightweight hyperparameter grid search harness for training/train.py.

Example:
  python training/grid_search.py \
    --mode runtime_aware \
    --data_path training/data/runtime_aware.jsonl \
    --search_output_dir training/hp_search/runtime_aware \
    --num_train_epochs_values 1,2 \
    --learning_rate_values 1e-4,2e-4 \
    --max_trials 1
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a hyperparameter grid search over training/train.py."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["runtime_aware", "control"],
        required=True,
        help="Training mode passed through to train.py.",
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to training JSONL.")
    parser.add_argument(
        "--search_output_dir",
        type=str,
        required=True,
        help="Directory where trial outputs and summary files are written.",
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None, help="Forwarded to train.py for debug runs.")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="hpml-efficient-codegen")
    parser.add_argument("--max_trials", type=int, default=None, help="Run only the first N trials.")

    # Grid dimensions
    parser.add_argument("--num_train_epochs_values", type=_parse_int_list, default=[3])
    parser.add_argument("--per_device_train_batch_size_values", type=_parse_int_list, default=[4])
    parser.add_argument("--gradient_accumulation_steps_values", type=_parse_int_list, default=[4])
    parser.add_argument("--learning_rate_values", type=_parse_float_list, default=[2e-4])
    parser.add_argument("--lora_r_values", type=_parse_int_list, default=[16])
    parser.add_argument("--lora_alpha_values", type=_parse_int_list, default=[32])
    parser.add_argument("--lora_dropout_values", type=_parse_float_list, default=[0.05])

    return parser.parse_args()


def build_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    grid = []
    keys = [
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
    ]
    values = [
        args.num_train_epochs_values,
        args.per_device_train_batch_size_values,
        args.gradient_accumulation_steps_values,
        args.learning_rate_values,
        args.lora_r_values,
        args.lora_alpha_values,
        args.lora_dropout_values,
    ]
    for combo in itertools.product(*values):
        grid.append(dict(zip(keys, combo)))
    return grid


def _extract_final_train_loss(trainer_state_path: Path) -> float | None:
    if not trainer_state_path.exists():
        return None
    with trainer_state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    log_history = state.get("log_history", [])
    for entry in reversed(log_history):
        if "train_loss" in entry:
            return float(entry["train_loss"])
    return None


def run_trial(
    args: argparse.Namespace,
    hp: dict[str, Any],
    trial_idx: int,
    total_trials: int,
    search_output_dir: Path,
) -> dict[str, Any]:
    trial_name = f"trial_{trial_idx:03d}"
    trial_dir = search_output_dir / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "training/train.py",
        "--mode",
        args.mode,
        "--data_path",
        args.data_path,
        "--output_dir",
        str(trial_dir),
        "--model_name",
        args.model_name,
        "--max_seq_length",
        str(args.max_seq_length),
        "--num_train_epochs",
        str(hp["num_train_epochs"]),
        "--per_device_train_batch_size",
        str(hp["per_device_train_batch_size"]),
        "--gradient_accumulation_steps",
        str(hp["gradient_accumulation_steps"]),
        "--learning_rate",
        str(hp["learning_rate"]),
        "--lora_r",
        str(hp["lora_r"]),
        "--lora_alpha",
        str(hp["lora_alpha"]),
        "--lora_dropout",
        str(hp["lora_dropout"]),
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.use_wandb:
        cmd.extend(["--use_wandb", "--wandb_project", args.wandb_project])

    print(f"[{trial_idx}/{total_trials}] Running {trial_name} with hp={hp}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed_sec = time.time() - start

    (trial_dir / "stdout.log").write_text(result.stdout, encoding="utf-8")
    (trial_dir / "stderr.log").write_text(result.stderr, encoding="utf-8")

    trainer_state_path = trial_dir / "trainer_state.json"
    final_train_loss = _extract_final_train_loss(trainer_state_path)

    trial_result = {
        "trial_name": trial_name,
        "output_dir": str(trial_dir),
        "status": "ok" if result.returncode == 0 else "failed",
        "return_code": result.returncode,
        "elapsed_sec": round(elapsed_sec, 2),
        "final_train_loss": final_train_loss,
        **hp,
    }

    if result.returncode != 0:
        print(f"  -> FAILED (return_code={result.returncode})")
    else:
        print(f"  -> OK (final_train_loss={final_train_loss}, elapsed_sec={elapsed_sec:.1f})")
    return trial_result


def main() -> None:
    args = parse_args()
    search_output_dir = Path(args.search_output_dir)
    search_output_dir.mkdir(parents=True, exist_ok=True)

    grid = build_grid(args)
    if args.max_trials is not None:
        grid = grid[: args.max_trials]
    if not grid:
        raise ValueError("Grid is empty. Check *_values arguments.")

    print(f"Total trials to run: {len(grid)}")
    results: list[dict[str, Any]] = []

    for i, hp in enumerate(grid, start=1):
        trial_result = run_trial(args, hp, i, len(grid), search_output_dir)
        results.append(trial_result)

    summary_json = search_output_dir / "results.json"
    summary_csv = search_output_dir / "results.csv"
    summary_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    fieldnames = [
        "trial_name",
        "status",
        "return_code",
        "elapsed_sec",
        "final_train_loss",
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "output_dir",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    ok_results = [r for r in results if r["status"] == "ok" and r["final_train_loss"] is not None]
    if ok_results:
        best = min(ok_results, key=lambda r: r["final_train_loss"])
        print("\nBest trial by final_train_loss:")
        print(json.dumps(best, indent=2))
    else:
        print("\nNo successful trials with final_train_loss recorded.")

    print(f"\nSaved summary files:\n- {summary_json}\n- {summary_csv}")


if __name__ == "__main__":
    main()
