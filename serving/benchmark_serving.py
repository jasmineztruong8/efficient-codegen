"""Run a simple serving benchmark with either Hugging Face or vLLM."""

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "Write a correct Python solution optimized for fast execution time. "
    "Use efficient algorithms and data structures to minimize runtime. "
    "Return only the code with no explanation."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark serving with HF or vLLM.")
    parser.add_argument("--backend", choices=["hf", "vllm"], required=True)
    parser.add_argument("--model_path", type=str, required=True, help="HF model name or local checkpoint path")
    parser.add_argument("--input_path", type=str, required=True, help="Path to dataset_clean.json")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save JSON results")
    parser.add_argument("--limit", type=int, default=None, help="Limit prompts for smoke tests")
    parser.add_argument("--batch_size", type=int, default=8, help="Prompts per batch")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--gpu_sample_interval", type=float, default=0.5)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="hpml-efficient-codegen")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()


def build_messages(instruction, input_content):
    content = instruction
    if input_content:
        content += f"\n\nInput:\n{input_content}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def load_prompts(input_path, limit, tokenizer):
    with open(input_path, "r", encoding="utf-8") as f:
        problems = json.load(f)
    if limit:
        problems = problems[:limit]

    return [
        tokenizer.apply_chat_template(
            build_messages(p["instruction"], p.get("input", "")),
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in problems
    ]


def poll_gpu_stats(samples, stop_event, interval):
    while not stop_event.is_set():
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
            if output:
                first_line = output.splitlines()[0]
                util_str, mem_str = [x.strip() for x in first_line.split(",")]
                samples.append((float(util_str), float(mem_str)))
        except Exception:
            pass
        time.sleep(interval)


def summarize_gpu_samples(samples):
    if not samples:
        return {
            "avg_gpu_util_pct": None,
            "max_gpu_util_pct": None,
            "avg_gpu_mem_mb": None,
            "max_gpu_mem_mb": None,
        }

    utils = [u for u, _ in samples]
    mems = [m for _, m in samples]
    return {
        "avg_gpu_util_pct": sum(utils) / len(utils),
        "max_gpu_util_pct": max(utils),
        "avg_gpu_mem_mb": sum(mems) / len(mems),
        "max_gpu_mem_mb": max(mems),
    }


def get_torch_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def start_gpu_monitor(interval):
    samples = []
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=poll_gpu_stats,
        args=(samples, stop_event, interval),
        daemon=True,
    )
    monitor.start()
    return samples, stop_event, monitor


def stop_gpu_monitor(samples, stop_event, monitor):
    stop_event.set()
    monitor.join(timeout=1)
    return summarize_gpu_samples(samples)


def build_result(args, num_prompts, total_time, total_generated_tokens, batch_latencies, gpu_stats, peak_mem_mb):
    return {
        "backend": args.backend,
        "model_path": args.model_path,
        "num_prompts": num_prompts,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "avg_latency_per_prompt_s": total_time / num_prompts if num_prompts else None,
        "avg_batch_latency_s": sum(batch_latencies) / len(batch_latencies) if batch_latencies else None,
        "throughput_prompts_per_s": num_prompts / total_time if total_time > 0 else None,
        "throughput_output_tokens_per_s": total_generated_tokens / total_time if total_time > 0 else None,
        "peak_cuda_memory_mb": peak_mem_mb,
        **gpu_stats,
    }


def benchmark_hf(args):
    dtype = get_torch_dtype()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    prompts = load_prompts(args.input_path, args.limit, tokenizer)
    print(f"Loaded {len(prompts)} prompts")
    print(f"Running Hugging Face benchmark for {args.model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    gpu_samples, stop_event, monitor = start_gpu_monitor(args.gpu_sample_interval)

    total_generated_tokens = 0
    batch_latencies = []

    start_total = time.perf_counter()
    with torch.inference_mode():
        for batch_start in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[batch_start: batch_start + args.batch_size]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(model.device)

            input_len = inputs["input_ids"].shape[1]
            batch_start_time = time.perf_counter()
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            batch_latencies.append(time.perf_counter() - batch_start_time)
            new_tokens_per_prompt = outputs.shape[1] - input_len
            total_generated_tokens += new_tokens_per_prompt * len(batch_prompts)
    total_time = time.perf_counter() - start_total

    gpu_stats = stop_gpu_monitor(gpu_samples, stop_event, monitor)
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None
    return build_result(args, len(prompts), total_time, total_generated_tokens, batch_latencies, gpu_stats, peak_mem_mb)


def benchmark_vllm(args):
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompts = load_prompts(args.input_path, args.limit, tokenizer)
    print(f"Loaded {len(prompts)} prompts")
    print(f"Running vLLM benchmark for {args.model_path}")

    dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype=dtype,
        gpu_memory_utilization=0.90,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    gpu_samples, stop_event, monitor = start_gpu_monitor(args.gpu_sample_interval)

    total_generated_tokens = 0
    batch_latencies = []

    start_total = time.perf_counter()
    for batch_start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[batch_start: batch_start + args.batch_size]
        batch_start_time = time.perf_counter()
        outputs = llm.generate(batch_prompts, sampling_params)
        batch_latencies.append(time.perf_counter() - batch_start_time)
        total_generated_tokens += sum(len(o.outputs[0].token_ids) for o in outputs)
    total_time = time.perf_counter() - start_total

    gpu_stats = stop_gpu_monitor(gpu_samples, stop_event, monitor)
    return build_result(
        args,
        len(prompts),
        total_time,
        total_generated_tokens,
        batch_latencies,
        gpu_stats,
        gpu_stats["max_gpu_mem_mb"],
    )


def save_result(result, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def log_to_wandb(args, result):
    import wandb

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        config={
            "experiment_type": "serving_optimization",
            "backend": args.backend,
            "model_path": args.model_path,
            "num_prompts": result["num_prompts"],
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
    )
    wandb.log(result)
    wandb.finish()


def main():
    args = parse_args()
    result = benchmark_hf(args) if args.backend == "hf" else benchmark_vllm(args)

    output_path = Path(args.output_path)
    save_result(result, output_path)

    print(json.dumps(result, indent=2))
    print(f"Results saved to {output_path}")

    if args.use_wandb:
        log_to_wandb(args, result)


if __name__ == "__main__":
    main()
