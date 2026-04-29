"""
evaluate_model.py

Evaluate any model (base Qwen or a fine-tuned checkpoint) on Pass@1 and
median execution time. Used to compare the three conditions in the ablation:

  Base SLM        — no fine-tuning
  Control SFT     — fine-tuned on first-correct candidates
  Runtime-aware SFT — fine-tuned on fastest-correct candidates

Wraps generate → evaluate correctness → benchmark runtime into one script
and logs results to WandB for easy comparison across runs.

Detects PEFT checkpoints automatically (looks for adapter_config.json).

For profiling inference: pass --profile. Traces are saved to --trace_dir
and follow the same pattern as profiling/profile_model.py.

Usage:
  # Base model
  python training/evaluate_model.py \
    --model_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --run_name base_slm \
    --data_path data/curated/test/dataset_clean.json \
    --use_wandb

  # Fine-tuned checkpoint
  python training/evaluate_model.py \
    --model_path training/checkpoints/runtime_aware \
    --base_model_name Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --run_name runtime_aware_sft \
    --data_path data/curated/test/dataset_clean.json \
    --use_wandb

  # With profiling
  python training/evaluate_model.py \
    --model_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --run_name base_slm \
    --data_path data/curated/test/dataset_clean.json \
    --profile --trace_dir outputs/tb_profiler/eval
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import signal
import statistics
import time
import traceback
from math import comb
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Must match SYSTEM_PROMPT in generation/generate_candidates.py and training/train.py
SYSTEM_PROMPT = (
    "Write a correct Python solution optimized for fast execution time. "
    "Use efficient algorithms and data structures to minimize runtime. "
    "Return only the code with no explanation."
)

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
EVAL_TIMEOUT_SEC = 5


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_path: str,
    base_model_name: str,
    device: str,
    dtype,
):
    """
    Load a model and tokenizer.

    If model_path contains adapter_config.json it is treated as a PEFT
    checkpoint: the base model is loaded first, then the adapter is merged.
    Otherwise model_path is used directly (HF hub name or local base model).
    """
    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.exists():
        from peft import PeftModel
        print(f"Detected PEFT checkpoint at {model_path}")
        print(f"Loading base model: {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()  # merge LoRA weights for faster inference
    else:
        print(f"Loading model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left padding for batched generation
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def build_messages(instruction: str, input_content: str) -> list:
    content = instruction
    if input_content:
        content += f"\n\nInput:\n{input_content}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def generate_candidates(
    model,
    tokenizer,
    problems: list,
    num_candidates: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
    profiler=None,
) -> list[dict]:
    """
    Generate num_candidates solutions per problem.

    Returns a list of dicts, one per problem, with keys:
      dataset_index, instruction, input, tests, candidates (list of str)

    profiler: optional torch.profiler.profile context — if provided,
    profiler.step() is called after each batch for trace data.
    """
    results = []
    with torch.inference_mode():
        for batch_start in tqdm(range(0, len(problems), batch_size), desc="Generating"):
            batch = problems[batch_start: batch_start + batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    build_messages(p["instruction"], p.get("input", "")),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in batch
            ]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(device)

            # Use greedy decoding when temperature=0; sampling otherwise.
            greedy = temperature == 0.0
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=not greedy,
                temperature=None if greedy else temperature,
                top_p=None if greedy else top_p,
                num_return_sequences=num_candidates,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            input_len = inputs["input_ids"].shape[1]
            decoded = tokenizer.batch_decode(
                output_ids[:, input_len:],
                skip_special_tokens=True,
            )

            for i, problem in enumerate(batch):
                results.append({
                    "dataset_index": problem["dataset_index"],
                    "instruction": problem["instruction"],
                    "input": problem.get("input", ""),
                    "tests": problem["tests"],
                    "candidates": decoded[i * num_candidates: (i + 1) * num_candidates],
                })

            # --- Profiling hook ---
            # If a profiler is passed, step it here. To extend profiling
            # (e.g. log per-batch GPU memory), add logic below.
            if profiler is not None:
                profiler.step()

    return results


# ---------------------------------------------------------------------------
# Correctness evaluation
# ---------------------------------------------------------------------------

def _timeout_handler(signum, frame):
    raise TimeoutError("Candidate timed out")


def extract_code(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text


def evaluate_one_candidate(code: str, tests: str) -> dict[str, Any]:
    namespace: dict = {}
    fake_out = io.StringIO()
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(EVAL_TIMEOUT_SEC)
    try:
        try:
            with contextlib.redirect_stdout(fake_out), contextlib.redirect_stderr(fake_out):
                exec(code, namespace)
        except TimeoutError:
            return {"passed": False, "error_type": "TimeoutError"}
        except Exception as e:
            return {"passed": False, "error_type": type(e).__name__}

        try:
            with contextlib.redirect_stdout(fake_out), contextlib.redirect_stderr(fake_out):
                exec(tests, namespace)
        except TimeoutError:
            return {"passed": False, "error_type": "TimeoutError"}
        except Exception as e:
            return {"passed": False, "error_type": type(e).__name__}

        return {"passed": True, "error_type": None}
    finally:
        signal.alarm(0)


# ---------------------------------------------------------------------------
# Runtime benchmarking
# ---------------------------------------------------------------------------

def benchmark_candidate(code: str, tests: str, num_runs: int, warmup_runs: int) -> dict[str, Any]:
    """Run code+tests repeatedly and return median execution time."""
    def run_once():
        ns: dict = {}
        fake_out = io.StringIO()
        try:
            with contextlib.redirect_stdout(fake_out), contextlib.redirect_stderr(fake_out):
                exec(code, ns)
            with contextlib.redirect_stdout(fake_out), contextlib.redirect_stderr(fake_out):
                t0 = time.perf_counter()
                exec(tests, ns)
                return time.perf_counter() - t0, True
        except Exception:
            return None, False

    for _ in range(warmup_runs):
        _, ok = run_once()
        if not ok:
            return {"benchmark_passed": False, "median_time": None}

    times = []
    for _ in range(num_runs):
        elapsed, ok = run_once()
        if not ok:
            return {"benchmark_passed": False, "median_time": None}
        times.append(elapsed)

    return {
        "benchmark_passed": True,
        "median_time": statistics.median(times),
        "mean_time": statistics.mean(times),
        "std_time": statistics.pstdev(times) if len(times) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model for the ablation study.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="HF model name (base) or path to PEFT checkpoint directory",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default=DEFAULT_MODEL,
        help="Base model name — only needed when loading a PEFT checkpoint",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name for this evaluation run (e.g. base_slm, control_sft, runtime_aware_sft)",
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset_clean.json")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Optional path to save per-problem results JSONL")
    parser.add_argument("--num_candidates", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--num_runs", type=int, default=7, help="Benchmark timing runs per candidate")
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems evaluated")
    # WandB
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="hpml-efficient-codegen")
    # Profiling — same pattern as profiling/profile_model.py
    parser.add_argument("--profile", action="store_true", help="Enable PyTorch Profiler during generation")
    parser.add_argument("--trace_dir", type=str, default="outputs/tb_profiler/eval",
                        help="Directory for TensorBoard profiler traces (used with --profile)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print(f"Loading dataset from {args.data_path}")
    with open(args.data_path, "r", encoding="utf-8") as f:
        problems = json.load(f)
    if args.limit:
        problems = problems[:args.limit]
    print(f"Evaluating on {len(problems)} problems")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16 if torch.cuda.is_available()
        else torch.float32
    )

    model, tokenizer = load_model_and_tokenizer(
        args.model_path, args.base_model_name, device, dtype
    )

    # --- Profiling setup ---
    # Profiler is passed into generate_candidates() which calls profiler.step()
    # after each batch. To extend: add memory/timing logging per batch there.
    profiler_ctx = None
    profiler = None

    if args.profile:
        from torch.profiler import (
            ProfilerActivity,
            profile as torch_profile,
            schedule,
            tensorboard_trace_handler,
        )
        Path(args.trace_dir).mkdir(parents=True, exist_ok=True)
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        profiler_ctx = torch_profile(
            activities=activities,
            schedule=schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=tensorboard_trace_handler(args.trace_dir, worker_name="eval"),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        profiler = profiler_ctx.__enter__()
        print(f"Profiling enabled — traces will be saved to {args.trace_dir}")

    # Generation
    t_gen_start = time.perf_counter()
    generated = generate_candidates(
        model=model,
        tokenizer=tokenizer,
        problems=problems,
        num_candidates=args.num_candidates,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
        profiler=profiler,
    )
    gen_time = time.perf_counter() - t_gen_start

    if profiler_ctx is not None:
        profiler_ctx.__exit__(None, None, None)
        print(f"Profiler traces saved to {args.trace_dir}")

    # Evaluate correctness + benchmark runtime
    print("Evaluating correctness and benchmarking...")
    pass_counts: dict[int, dict] = {}
    median_times: list[float] = []
    per_problem_results = []

    for record in tqdm(generated, desc="Evaluating"):
        idx = record["dataset_index"]
        tests = record["tests"]
        candidates = record["candidates"]

        n = len(candidates)
        c = 0
        best_time = None

        for raw in candidates:
            code = extract_code(raw)
            eval_result = evaluate_one_candidate(code, tests)
            if not eval_result["passed"]:
                continue
            c += 1

            bench = benchmark_candidate(code, tests, args.num_runs, args.warmup_runs)
            if bench["benchmark_passed"] and bench["median_time"] is not None:
                if best_time is None or bench["median_time"] < best_time:
                    best_time = bench["median_time"]

        pass_counts[idx] = {"n": n, "c": c}
        if best_time is not None:
            median_times.append(best_time)

        per_problem_results.append({
            "dataset_index": idx,
            "num_candidates": n,
            "num_passing": c,
            "best_median_time": best_time,
        })

    # Pass@1 unbiased estimator
    k = 1
    pass_at_1_scores = []
    for stats in pass_counts.values():
        n, c = stats["n"], stats["c"]
        score = 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)
        pass_at_1_scores.append(score)
    pass_at_1 = sum(pass_at_1_scores) / len(pass_at_1_scores) if pass_at_1_scores else 0.0

    overall_median_time = statistics.median(median_times) if median_times else None
    problems_with_passing = sum(1 for s in pass_counts.values() if s["c"] > 0)
    avg_gen_latency = gen_time / len(problems) if problems else None

    peak_mem_mb = None
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"\n{'='*50}")
    print(f"Run: {args.run_name}")
    print(f"Problems evaluated:           {len(problems)}")
    print(f"Problems with ≥1 passing:     {problems_with_passing} / {len(problems)}")
    print(f"Pass@1 (unbiased):            {pass_at_1:.4f}")
    print(f"Median execution time (s):    {overall_median_time:.6f}" if overall_median_time else "Median execution time:        N/A")
    print(f"Avg generation latency (s):   {avg_gen_latency:.2f}" if avg_gen_latency else "")
    if peak_mem_mb:
        print(f"Peak CUDA memory (MB):        {peak_mem_mb:.1f}")
    print(f"{'='*50}")

    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={
                "model_path": args.model_path,
                "num_candidates": args.num_candidates,
                "num_problems": len(problems),
                "temperature": args.temperature,
            },
        )
        log_dict = {
            "pass_at_1": pass_at_1,
            "problems_with_passing": problems_with_passing,
            "avg_generation_latency_s": avg_gen_latency,
        }
        if overall_median_time is not None:
            log_dict["median_execution_time_s"] = overall_median_time
        if peak_mem_mb is not None:
            log_dict["peak_cuda_memory_mb"] = peak_mem_mb
        wandb.log(log_dict)
        wandb.finish()

    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            for r in per_problem_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Per-problem results saved to {args.output_path}")


if __name__ == "__main__":
    main()
