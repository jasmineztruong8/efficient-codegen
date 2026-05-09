"""
profile_operators.py

Operator-level profiling with torch.profiler. Produces:
  - operators.csv                  (top-N ops by time)
  - bottleneck_report.txt
  - roofline_report.txt + roofline.png
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer

from generation import build_prompt, ensure_dir, generate_with_annotations, load_examples
from roofline import compute_roofline_metrics, format_roofline_report, plot_roofline


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Operator-level torch.profiler trace for LLM generation.")
    p.add_argument("--input_path",  default="data/curated/prototyping/generation_input_20.json")
    p.add_argument("--model_name",  default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    p.add_argument("--output_dir",  default="outputs/operator_profile")
    p.add_argument("--limit",       type=int,   default=5)
    p.add_argument("--batch_size",  type=int,   default=1)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p",       type=float, default=0.95)
    p.add_argument("--top_ops",     type=int,   default=30)
    p.add_argument("--decode_steps_to_trace", type=int, default=5)
    p.add_argument("--gpu_name",    default="T4")
    p.add_argument("--gpu_peak_tflops",        type=float, default=65.0)
    p.add_argument("--gpu_peak_bandwidth_gbs", type=float, default=300.0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Bottleneck analysis
# ---------------------------------------------------------------------------

_SYNC_OPS     = {"aten::item", "aten::_local_scalar_dense", "cudaDeviceSynchronize",
                 "cudaStreamSynchronize", "aten::copy_"}
_ATTENTION_OPS = {"aten::softmax", "aten::_softmax", "aten::scaled_dot_product_attention",
                  "aten::baddbmm", "aten::bmm"}
_LINEAR_OPS   = {"aten::linear", "aten::mm", "aten::addmm", "aten::matmul"}
_MEMORY_OPS   = {"aten::to", "aten::copy_", "aten::contiguous", "aten::clone"}

_CHECKLIST = """
  [1] Mixed precision     — bfloat16 / float16 with torch.autocast
  [2] torch.compile       — mode='reduce-overhead' fuses elementwise ops
  [3] FlashAttention/SDPA — torch.backends.cuda.enable_flash_sdp(True)
  [4] Batch size          — larger batches amortise launch overhead
  [5] No per-token syncs  — avoid .item() / .numpy() inside the decode loop
  [6] KV-cache dtype      — store past_key_values in bfloat16, not float32
"""


def analyze_bottlenecks(key_avgs, top_n: int, device: str, time_key: str = "") -> str:
    lines: List[str] = ["=" * 72, "OPERATOR-LEVEL BOTTLENECK REPORT", "=" * 72]

    if not time_key:
        time_key = "cuda_time_total" if device == "cuda" else "cpu_time_total"

    sorted_ops = sorted(key_avgs, key=lambda e: getattr(e, time_key, 0), reverse=True)
    total_time = sum(getattr(e, time_key, 0) for e in sorted_ops) or 1
    time_label = "CUDA" if time_key == "cuda_time_total" else "CPU"

    lines.append(f"\nTop {top_n} operators by {time_label} self-time:\n")
    header = f"{'Operator':<55} {'CUDA ms':>10} {'CPU ms':>10} {'Calls':>8} {'% total':>8}"
    lines += [header, "-" * len(header)]
    for e in sorted_ops[:top_n]:
        pct = getattr(e, time_key, 0) / total_time * 100
        lines.append(
            f"{e.key:<55} {getattr(e,'cuda_time_total',0)/1e3:>10.3f} "
            f"{e.cpu_time_total/1e3:>10.3f} {e.count:>8} {pct:>7.1f}%"
        )

    op_map = {e.key: e for e in key_avgs}
    bottlenecks: List[tuple[str, str, str]] = []

    def _time(ops_set):
        return sum(getattr(op_map[k], time_key, 0) for k in ops_set if k in op_map)

    if _time(_SYNC_OPS) / total_time > 0.02:
        bottlenecks.append((
            "CPU-GPU synchronisations",
            f"{_time(_SYNC_OPS)/total_time*100:.1f}% of time in sync ops.",
            "Avoid .item()/.cpu()/numpy() inside the generation loop.",
        ))
    if _time(_MEMORY_OPS) / total_time > 0.05:
        bottlenecks.append((
            "Excessive memory copies",
            f"{_time(_MEMORY_OPS)/total_time*100:.1f}% of time in memory ops.",
            "Pre-move tensors to device once; avoid unnecessary .contiguous() calls.",
        ))
    if _time(_ATTENTION_OPS) / total_time > 0.15:
        bottlenecks.append((
            "Attention is a dominant kernel",
            f"{_time(_ATTENTION_OPS)/total_time*100:.1f}% of time in attention ops.",
            "Enable FlashAttention: torch.backends.cuda.enable_flash_sdp(True).",
        ))
    if _time(_LINEAR_OPS) / total_time > 0.30:
        bottlenecks.append((
            "Linear/matmul kernels dominate",
            f"{_time(_LINEAR_OPS)/total_time*100:.1f}% of time in GEMM ops.",
            "Ensure bfloat16; consider torch.compile(mode='reduce-overhead').",
        ))

    prefill_t = sum(getattr(e, time_key, 0) for e in key_avgs if "prefill"     in e.key)
    decode_t  = sum(getattr(e, time_key, 0) for e in key_avgs if "decode_step" in e.key)
    if prefill_t and decode_t and prefill_t / (decode_t + 1e-9) > 5:
        ratio = prefill_t / (decode_t + 1e-9)
        bottlenecks.append((
            "Prefill >> decode (long-prompt regime)",
            f"Prefill is ~{ratio:.1f}x slower than the first decode steps.",
            "Consider prompt compression, prefix caching, or speculative decoding.",
        ))

    lines += ["\n" + "=" * 72, f"IDENTIFIED BOTTLENECKS ({len(bottlenecks)} found)", "=" * 72]
    if not bottlenecks:
        lines.append("No major bottlenecks detected.")
    for i, (name, obs, rec) in enumerate(bottlenecks, 1):
        lines += [f"\nBottleneck {i}: {name}", f"  Observation : {obs}", f"  Suggestion  : {rec}"]

    lines += ["\n" + "=" * 72, "GENERAL OPTIMISATION CHECKLIST", "=" * 72, _CHECKLIST]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
              else torch.float16 if torch.cuda.is_available() else torch.float32)
    print(f"Device : {device}  |  dtype : {dtype}\nOutput : {out_dir}")

    examples = load_examples(args.input_path, args.limit)
    prompts  = [build_prompt(ex) for ex in examples]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True,
        torch_dtype=dtype, device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    print("\nWarming up...")
    with torch.no_grad():
        warm = tokenizer(prompts[0], return_tensors="pt")
        if device == "cuda":
            warm = {k: v.to(device) for k, v in warm.items()}
        model(**warm)
    if device == "cuda":
        torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == "cuda" else [])
    profiler_kwargs: Dict[str, Any] = dict(
        activities=activities, record_shapes=True, profile_memory=True, with_stack=False,
    )
    try:
        profiler_kwargs["with_flops"] = True
    except TypeError:
        pass

    bs      = args.batch_size
    batches = [prompts[i : i + bs] for i in range(0, len(prompts), bs)]
    print(f"\nProfiling {len(prompts)} example(s) in {len(batches)} batch(es) of {bs}...\n")

    with profile(**profiler_kwargs) as prof:
        for step, batch_prompts in enumerate(batches):
            with record_function("tokenization"):
                enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
            with record_function("host_to_device"):
                if device == "cuda":
                    enc = {k: v.to(device) for k, v in enc.items()}

            input_ids      = enc["input_ids"]
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()

            t0 = time.perf_counter()
            with record_function("full_generate"):
                generated = generate_with_annotations(
                    model, tokenizer, input_ids, attention_mask,
                    args.max_new_tokens, args.temperature, args.top_p,
                    args.decode_steps_to_trace,
                )
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            out_len = generated.shape[1] - input_ids.shape[1]
            print(f"[batch {step+1}/{len(batches)}]  bs={input_ids.shape[0]}  "
                  f"in={input_ids.shape[1]}  out={out_len}  "
                  f"latency={t1-t0:.3f}s  tok/s={out_len*input_ids.shape[0]/(t1-t0):.1f}")
            prof.step()

    # Operator CSV + bottleneck report
    key_avgs = prof.key_averages(group_by_input_shape=False)
    has_cuda = any(getattr(e, "cuda_time_total", 0) > 0 for e in key_avgs)
    time_key = "cuda_time_total" if (device == "cuda" and has_cuda) else "cpu_time_total"

    csv_path = str(out_dir / "operators.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["operator", "cuda_time_ms", "cpu_time_ms", "calls",
                    "cuda_memory_usage_mb", "self_cuda_time_ms"])
        for e in sorted(key_avgs, key=lambda e: getattr(e, time_key, 0), reverse=True)[:args.top_ops]:
            w.writerow([e.key,
                        f"{getattr(e,'cuda_time_total',0)/1e3:.4f}",
                        f"{e.cpu_time_total/1e3:.4f}",
                        e.count,
                        f"{getattr(e,'cuda_memory_usage',0)/1e6:.4f}",
                        f"{getattr(e,'self_cuda_time_total',0)/1e3:.4f}"])
    print(f"Operator CSV → {csv_path}")

    report = analyze_bottlenecks(key_avgs, top_n=args.top_ops, device=device, time_key=time_key)
    Path(out_dir / "bottleneck_report.txt").write_text(report)
    print(f"Bottleneck report → {out_dir / 'bottleneck_report.txt'}")

    # Roofline
    gpu_peak_flops     = args.gpu_peak_tflops * 1e12
    gpu_peak_bandwidth = args.gpu_peak_bandwidth_gbs * 1e9
    key_avgs_shaped    = prof.key_averages(group_by_input_shape=True)
    roofline_metrics   = compute_roofline_metrics(
        key_avgs_shaped, gpu_peak_flops, gpu_peak_bandwidth, analytical_dtype_bytes=2,
    )
    roofline_report = format_roofline_report(
        roofline_metrics, gpu_peak_flops, gpu_peak_bandwidth, args.gpu_name, key_avgs,
    )
    Path(out_dir / "roofline_report.txt").write_text(roofline_report)
    print(f"Roofline report → {out_dir / 'roofline_report.txt'}")

    if roofline_metrics:
        plot_roofline(roofline_metrics, gpu_peak_flops, gpu_peak_bandwidth,
                      args.gpu_name, str(out_dir / "roofline.png"))
    else:
        print("No FLOPs data — roofline plot skipped.")

    print(f"\nTensorBoard: tensorboard --logdir {tb_dir}")


if __name__ == "__main__":
    main()
