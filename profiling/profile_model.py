"""
profile_model.py

Short description:
Profile model generation with PyTorch Profiler on a small representative subset.
Captures CPU/CUDA activity, memory usage, and saves TensorBoard traces.
Optionally logs summary metrics to Weights & Biases.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import wandb
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile generation with PyTorch Profiler.")
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/curated/prototyping/generation_input_20.json",
        help="Path to generation input JSON",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="HF model name",
    )
    parser.add_argument(
        "--trace_dir",
        type=str,
        default="outputs/tb_profiler",
        help="Directory for TensorBoard profiler traces",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of examples to profile",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Max new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Whether to log summary metrics to W&B",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="hpml-efficient-codegen",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="prototype-profiler",
        help="W&B run name",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_examples(path: str, limit: int) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:limit]


def build_prompt(example: Dict[str, Any]) -> str:
    instruction = (example.get("instruction") or "").strip()
    extra_input = (example.get("input") or "").strip()

    prompt = (
        "You are a helpful coding assistant.\n"
        "Write a correct Python solution for the following problem.\n"
        "Return only Python code, with no explanation.\n\n"
        f"Problem:\n{instruction}\n"
    )
    if extra_input:
        prompt += f"\nAdditional input:\n{extra_input}\n"
    return prompt


def main() -> None:
    args = parse_args()
    ensure_dir(args.trace_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )

    examples = load_examples(args.input_path, args.limit)

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model_name": args.model_name,
                "limit": args.limit,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "device": device,
                "dtype": str(dtype),
            },
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model.to("cpu")
    model.eval()

    prompts = [build_prompt(ex) for ex in examples]

    latencies = []
    input_lengths = []
    output_lengths = []

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        schedule=schedule(wait=1, warmup=1, active=3),
        on_trace_ready=tensorboard_trace_handler(args.trace_dir, worker_name="worker0"),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for step, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt")
            input_len = inputs["input_ids"].shape[1]

            if device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()

            start = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

            latency = end - start
            output_len = outputs.shape[1] - input_len

            latencies.append(latency)
            input_lengths.append(input_len)
            output_lengths.append(output_len)

            prof.step()

            print(
                f"[{step + 1}/{len(prompts)}] "
                f"input_len={input_len} output_len={output_len} latency={latency:.4f}s"
            )

    avg_latency = sum(latencies) / len(latencies) if latencies else None
    avg_input_len = sum(input_lengths) / len(input_lengths) if input_lengths else None
    avg_output_len = sum(output_lengths) / len(output_lengths) if output_lengths else None

    peak_mem_mb = None
    if device == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print("\nSummary:")
    print(f"Average latency:      {avg_latency}")
    print(f"Average input length: {avg_input_len}")
    print(f"Average output len:   {avg_output_len}")
    print(f"Trace directory:      {args.trace_dir}")
    if peak_mem_mb is not None:
        print(f"Peak CUDA memory MB:  {peak_mem_mb:.2f}")

    if args.use_wandb:
        log_dict = {
            "avg_latency_s": avg_latency,
            "avg_input_len": avg_input_len,
            "avg_output_len": avg_output_len,
        }
        if peak_mem_mb is not None:
            log_dict["peak_cuda_memory_mb"] = peak_mem_mb
        wandb.log(log_dict)
        wandb.finish()


if __name__ == "__main__":
    main()