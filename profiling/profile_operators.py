"""
profile_operators.py

Operator-level profiling with torch.profiler.

Produces:
  - Chrome trace JSON    (open in chrome://tracing or ui.perfetto.dev)
  - TensorBoard trace    (existing pipeline)
  - operators.csv        (top-N ops sorted by CUDA self-time)
  - bottleneck_report.txt
  - roofline_report.txt  (arithmetic intensity, MFU, compute- vs memory-bound classification)
  - roofline.png         (log-log roofline chart)

Key instrumentation:
  - "host_to_device"  : tokenizer tensors moved to GPU
  - "prefill"         : first forward pass (prompt encoding, KV-cache fill)
  - "decode_step_N"   : each autoregressive decode step (first 5 captured)
  - "full_generate"   : end-to-end model.generate()
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, cast, Dict, List

import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe on headless GPU nodes
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Operator-level torch.profiler trace for LLM generation.")
    p.add_argument("--input_path", default="data/curated/prototyping/generation_input_20.json")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    p.add_argument("--output_dir", default="outputs/operator_profile",
                   help="All artefacts (trace, CSV, report) go here")
    p.add_argument("--limit", type=int, default=5,
                   help="Number of examples to profile (keep small; 3-5 is enough)")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_ops", type=int, default=30,
                   help="How many operators to print/save in the table")
    p.add_argument("--decode_steps_to_trace", type=int, default=5,
                   help="How many individual decode steps to annotate with record_function")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Number of prompts to process simultaneously. "
                        "Larger values improve GPU utilization (T4/G4 recommendation).")
    # Roofline model parameters — set to match your actual GPU
    p.add_argument("--gpu_name", default="T4",
                   help="GPU label shown on the roofline plot (e.g. T4, A100, V100)")
    p.add_argument("--gpu_peak_tflops", type=float, default=65.0,
                   help="Peak FP16 FLOPs in TFLOPS. T4=65, A100=312, V100=125, A10G=125")
    p.add_argument("--gpu_peak_bandwidth_gbs", type=float, default=300.0,
                   help="Peak HBM bandwidth in GB/s. T4=300, A100=2000, V100=900, A10G=600")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_examples(path: str, limit: int) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:limit]


def build_prompt(example: Dict[str, Any]) -> str:
    instruction = (example.get("instruction") or "").strip()
    extra = (example.get("input") or "").strip()
    prompt = (
        "You are a helpful coding assistant.\n"
        "Write a correct Python solution for the following problem.\n"
        "Return only Python code, with no explanation.\n\n"
        f"Problem:\n{instruction}\n"
    )
    if extra:
        prompt += f"\nAdditional input:\n{extra}\n"
    return prompt


# ---------------------------------------------------------------------------
# Custom generation loop with per-step record_function annotations
# ---------------------------------------------------------------------------

def generate_with_annotations(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    decode_steps_to_trace: int,
) -> torch.Tensor:
    """
    Autoregressive generation with explicit record_function regions:
      - 'prefill'         : single forward pass over the full prompt
      - 'decode_step_N'   : Nth token generation (first decode_steps_to_trace steps)
      - 'decode_remaining': all steps after the annotated ones
    """
    past_key_values = None
    generated = input_ids.clone()
    cur_attn = attention_mask.clone()

    # ---- prefill --------------------------------------------------------
    with record_function("prefill"):
        with torch.no_grad():
            out = model(
                input_ids=generated,
                attention_mask=cur_attn,
                use_cache=True,
            )
        past_key_values = out.past_key_values
        logits = out.logits[:, -1, :]  # (B, V)

    # sample first token for each item in the batch
    next_token = _sample(logits, temperature, top_p)          # (B, 1)
    generated = torch.cat([generated, next_token], dim=-1)
    cur_attn = torch.cat([cur_attn, torch.ones_like(next_token)], dim=-1)
    finished = (next_token.squeeze(-1) == tokenizer.eos_token_id)  # (B,)

    # ---- decode steps ---------------------------------------------------
    for step in range(1, max_new_tokens):
        if finished.all():
            break

        label = f"decode_step_{step}" if step <= decode_steps_to_trace else "decode_remaining"
        with record_function(label):
            with torch.no_grad():
                out = model(
                    input_ids=next_token,
                    attention_mask=cur_attn,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :]

        next_token = _sample(logits, temperature, top_p)      # (B, 1)
        # mask finished sequences so they emit pad tokens (no effect on output)
        next_token[finished] = tokenizer.pad_token_id
        finished |= (next_token.squeeze(-1) == tokenizer.eos_token_id)

        generated = torch.cat([generated, next_token], dim=-1)
        cur_attn = torch.cat([cur_attn, torch.ones_like(next_token)], dim=-1)

    return generated


def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Minimal top-p (nucleus) sampling — avoids heavy HF overhead in profiling."""
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    # top-p filter
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = (cumsum - sorted_probs) > top_p
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

    next_token_idx = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx.gather(-1, next_token_idx)


# ---------------------------------------------------------------------------
# Bottleneck analysis
# ---------------------------------------------------------------------------

# Operator name fragments associated with known patterns
_SYNC_OPS = {"aten::item", "aten::_local_scalar_dense", "cudaDeviceSynchronize",
             "cudaStreamSynchronize", "aten::copy_"}
_ATTENTION_OPS = {"aten::softmax", "aten::_softmax", "aten::scaled_dot_product_attention",
                  "aten::baddbmm", "aten::bmm"}
_LINEAR_OPS = {"aten::linear", "aten::mm", "aten::addmm", "aten::matmul"}
_MEMORY_OPS = {"aten::to", "aten::copy_", "aten::contiguous", "aten::clone"}


def analyze_bottlenecks(key_avgs, top_n: int, device: str, time_key: str = "") -> str:
    """
    Walk the key_averages list and produce a human-readable bottleneck report
    with concrete optimization suggestions.
    """
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("OPERATOR-LEVEL BOTTLENECK REPORT")
    lines.append("=" * 72)

    use_cuda = device == "cuda"
    if not time_key:
        time_key = "cuda_time_total" if use_cuda else "cpu_time_total"

    # Sort by CUDA (or CPU) self-time
    sorted_ops = sorted(key_avgs, key=lambda e: getattr(e, time_key, 0), reverse=True)
    total_time = sum(getattr(e, time_key, 0) for e in sorted_ops) or 1

    # ---- Top-N table -----
    time_label = "CUDA" if time_key == "cuda_time_total" else "CPU"
    lines.append(f"\nTop {top_n} operators by {time_label} self-time:\n")
    header = f"{'Operator':<55} {'CUDA ms':>10} {'CPU ms':>10} {'Calls':>8} {'% total':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for e in sorted_ops[:top_n]:
        cuda_ms = getattr(e, "cuda_time_total", 0) / 1e3
        cpu_ms = e.cpu_time_total / 1e3
        pct = getattr(e, time_key, 0) / total_time * 100
        lines.append(
            f"{e.key:<55} {cuda_ms:>10.3f} {cpu_ms:>10.3f} {e.count:>8} {pct:>7.1f}%"
        )

    # ---- Bottleneck detection ----
    bottlenecks: List[tuple[str, str, str]] = []  # (name, observation, recommendation)

    op_map = {e.key: e for e in key_avgs}

    # 1. Synchronisation ops (CPU-GPU syncs)
    sync_hits = [(k, op_map[k]) for k in _SYNC_OPS if k in op_map]
    sync_time = sum(getattr(e, time_key, 0) for _, e in sync_hits)
    if sync_time / total_time > 0.02:
        names = ", ".join(k for k, _ in sync_hits[:4])
        bottlenecks.append((
            "Unnecessary CPU-GPU synchronisations",
            f"Ops [{names}] account for {sync_time/total_time*100:.1f}% of total time.",
            "Avoid calling .item(), .cpu(), or numpy() inside the generation loop. "
            "Batch stopping-criterion checks or use model.generate() with synced_gpus=False.",
        ))

    # 2. Memory copies / contiguous calls
    mem_hits = [(k, op_map[k]) for k in _MEMORY_OPS if k in op_map]
    mem_time = sum(getattr(e, time_key, 0) for _, e in mem_hits)
    if mem_time / total_time > 0.05:
        bottlenecks.append((
            "Excessive memory copies / layout conversions",
            f"Memory ops account for {mem_time/total_time*100:.1f}% of total time.",
            "Pre-move all tensors to device once before the generation loop. "
            "Use contiguous layouts from the tokenizer (return_tensors='pt' already does this). "
            "Avoid explicit .contiguous() calls unless strictly required.",
        ))

    # 3. Attention ops — suggest FlashAttention / SDPA
    attn_hits = [(k, op_map[k]) for k in _ATTENTION_OPS if k in op_map]
    attn_time = sum(getattr(e, time_key, 0) for _, e in attn_hits)
    if attn_time / total_time > 0.15:
        bottlenecks.append((
            "Attention is a dominant kernel",
            f"Attention ops account for {attn_time/total_time*100:.1f}% of total time.",
            "Enable torch.backends.cuda.enable_flash_sdp(True) or use "
            "F.scaled_dot_product_attention with the xFormers / Flash-Attention backend. "
            "For long prompts, consider sliding-window or sparse attention.",
        ))

    # 4. Linear / matmul — suggest mixed precision
    lin_hits = [(k, op_map[k]) for k in _LINEAR_OPS if k in op_map]
    lin_time = sum(getattr(e, time_key, 0) for _, e in lin_hits)
    if lin_time / total_time > 0.30:
        bottlenecks.append((
            "Linear/matmul kernels dominate",
            f"GEMM ops account for {lin_time/total_time*100:.1f}% of total time.",
            "Ensure the model runs in bfloat16 or float16 (Tensor Core path). "
            "Wrap inference with torch.autocast('cuda', dtype=torch.bfloat16) even if the "
            "model is already loaded in bfloat16, as this also covers intermediate ops. "
            "Consider torch.compile() with mode='reduce-overhead' for static shapes.",
        ))

    # 5. Prefill dominates over decode (check via record_function events)
    prefill_ops = [e for e in key_avgs if "prefill" in e.key]
    decode_ops  = [e for e in key_avgs if "decode_step" in e.key]
    if prefill_ops and decode_ops:
        prefill_t = sum(getattr(e, time_key, 0) for e in prefill_ops)
        decode_t  = sum(getattr(e, time_key, 0) for e in decode_ops)
        ratio = prefill_t / (decode_t + 1e-9)
        if ratio > 5:
            bottlenecks.append((
                "Prefill latency >> decode latency (long-prompt regime)",
                f"Prefill is ~{ratio:.1f}x slower than the first {len(decode_ops)} decode steps.",
                "Reduce prompt length via prompt compression or prefix caching. "
                "For batch inference, pad to the nearest power-of-2 bucket to avoid excessive "
                "padding. Consider speculative decoding to overlap prefill and decode.",
            ))

    # ---- Print bottlenecks ----
    lines.append("\n" + "=" * 72)
    lines.append(f"IDENTIFIED BOTTLENECKS ({len(bottlenecks)} found)")
    lines.append("=" * 72)
    if not bottlenecks:
        lines.append("No major bottlenecks automatically detected.")
        lines.append("Inspect the Chrome trace in ui.perfetto.dev for manual analysis.")
    for i, (name, obs, rec) in enumerate(bottlenecks, 1):
        lines.append(f"\nBottleneck {i}: {name}")
        lines.append(f"  Observation : {obs}")
        lines.append(f"  Suggestion  : {rec}")

    lines.append("\n" + "=" * 72)
    lines.append("GENERAL OPTIMISATION CHECKLIST")
    lines.append("=" * 72)
    lines.append("""
  [1] Mixed precision
      Use bfloat16 (Ampere+) or float16 with torch.autocast. Tensor Cores give
      2-4x GEMM throughput vs float32.

  [2] torch.compile
      model = torch.compile(model, mode='reduce-overhead')
      Fuses element-wise ops, eliminates Python overhead in the decode loop.
      Disable dynamic shapes with dynamic=False for fixed-length batches.

  [3] FlashAttention / SDPA
      torch.backends.cuda.enable_flash_sdp(True)
      Reduces attention memory from O(n^2) to O(n) and improves throughput for
      sequences > 512 tokens.

  [4] Batch-size / sequence-length trade-off
      Larger batches amortise fixed launch overheads.
      Longer sequences shift the bottleneck from memory-bandwidth to compute.
      Profile both regimes and choose the crossover point for your workload.

  [5] Avoid per-token CPU-GPU syncs
      Do NOT call .item() / .numpy() inside the generation loop.
      Use logits.argmax(dim=-1) on GPU and pass the tensor directly.

  [6] KV-cache dtype
      If storing past_key_values in float32, cast to float16/bfloat16 to halve
      the memory footprint and reduce memory-bandwidth pressure during decode.

  [7] Dataloader / tokenization
      Tokenise the entire dataset once and cache to disk (datasets.save_to_disk).
      For batched generation, sort inputs by length to minimise padding.
""")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Roofline analysis
# ---------------------------------------------------------------------------

def compute_roofline_metrics(
    key_avgs,
    gpu_peak_flops: float,
    gpu_peak_bandwidth: float,
    time_key: str = "",
) -> List[Dict[str, Any]]:
    """
    Compute arithmetic intensity and attainable performance per profiled operator.

    AI = FLOPs / bytes.  We use cuda_memory_usage as a proxy for HBM traffic.
    For hardware-accurate values, use NVIDIA Nsight Compute (l1tex__t_bytes.sum).

    time_key: "cuda_time_total" or "cpu_time_total". When empty, auto-detects
    by checking whether any op has non-zero cuda_time_total (some PyTorch builds
    report 0 for all CUDA times despite running on GPU).
    """
    if not time_key:
        has_cuda = any(getattr(e, "cuda_time_total", 0) > 0 for e in key_avgs)
        time_key = "cuda_time_total" if has_cuda else "cpu_time_total"

    ridge_point = gpu_peak_flops / gpu_peak_bandwidth
    metrics: List[Dict[str, Any]] = []
    for e in key_avgs:
        flops = getattr(e, "flops", 0) or 0
        if flops <= 0:
            continue  # skip ops with no FLOPs estimate (activations, norms, etc.)

        mem_bytes = abs(getattr(e, "cuda_memory_usage", 0) or 0)
        if mem_bytes == 0:
            # Conservative fallback: place op at the ridge point
            mem_bytes = flops / ridge_point

        time_us = getattr(e, time_key, 0) or 0
        if time_us <= 0:
            continue

        ai = flops / mem_bytes
        perf_gflops = flops / (time_us / 1e6) / 1e9
        metrics.append({
            "name": e.key,
            "ai": ai,
            "perf_gflops": perf_gflops,
            "bound": "compute" if ai >= ridge_point else "memory",
            "cuda_ms": time_us / 1e3,
            "flops": flops,
            "mem_bytes": mem_bytes,
            "count": e.count,
            "time_key": time_key,
        })
    return sorted(metrics, key=lambda x: x["cuda_ms"], reverse=True)


def plot_roofline(
    metrics: List[Dict[str, Any]],
    gpu_peak_flops: float,
    gpu_peak_bandwidth: float,
    gpu_name: str,
    out_path: str,
) -> None:
    """Save a log-log roofline chart with each profiled operator as a scatter point."""
    peak_gflops = gpu_peak_flops / 1e9
    peak_gbps   = gpu_peak_bandwidth / 1e9
    ridge       = gpu_peak_flops / gpu_peak_bandwidth

    _, ax = plt.subplots(figsize=(13, 7))

    # Roofline envelope
    ai_lo  = min(m["ai"] for m in metrics) * 0.1 if metrics else 1e-2
    ai_hi  = max(m["ai"] for m in metrics) * 10  if metrics else ridge * 10
    ai_range = np.logspace(np.log10(max(ai_lo, 1e-3)), np.log10(ai_hi), 600)
    attainable = np.minimum(gpu_peak_bandwidth * ai_range, gpu_peak_flops) / 1e9
    ax.plot(ai_range, attainable, "k-", linewidth=2.5, label="Roofline ceiling", zorder=3)

    # Ridge point
    ax.axvline(x=ridge, color="gray", linestyle="--", alpha=0.6,
               label=f"Ridge point ({ridge:.0f} FLOPs/byte)")

    # Regime shading
    ax.axvspan(ai_lo * 0.5, ridge, alpha=0.04, color="steelblue")
    ax.axvspan(ridge, ai_hi * 2, alpha=0.04, color="crimson")
    ax.text(ridge * 0.05, peak_gflops * 0.3, "Memory-bound", fontsize=9,
            color="steelblue", alpha=0.7, ha="left")
    ax.text(ridge * 2.0, peak_gflops * 0.3, "Compute-bound", fontsize=9,
            color="crimson", alpha=0.7, ha="left")

    # Scatter operators
    for bound, color, marker in [("memory", "steelblue", "o"), ("compute", "crimson", "^")]:
        ops = [m for m in metrics if m["bound"] == bound and m["perf_gflops"] > 0]
        if not ops:
            continue
        xs    = [m["ai"]         for m in ops]
        ys    = [m["perf_gflops"] for m in ops]
        names = [m["name"]       for m in ops]
        label = "Memory-bound ops" if bound == "memory" else "Compute-bound ops"
        ax.scatter(xs, ys, c=color, s=110, marker=cast(Any, marker), zorder=5,
                   label=label, alpha=0.85, edgecolors="white", linewidths=0.6)
        for x, y, name in zip(xs, ys, names):
            short = name.replace("aten::", "").replace("cuda::", "")[:18]
            ax.annotate(short, (x, y), fontsize=6.5, alpha=0.75,
                        xytext=(5, 4), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity  (FLOPs / byte)", fontsize=13)
    ax.set_ylabel("Attainable Performance  (GFLOPs/s)", fontsize=13)
    ax.set_title(
        f"Roofline Model — {gpu_name}\n"
        f"Peak compute: {peak_gflops / 1e3:.0f} TFLOPs/s  |  "
        f"Peak BW: {peak_gbps:.0f} GB/s  |  "
        f"Ridge: {ridge:.0f} FLOPs/byte",
        fontsize=13,
    )
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Roofline plot saved → {out_path}")


def format_roofline_report(
    metrics: List[Dict[str, Any]],
    gpu_peak_flops: float,
    gpu_peak_bandwidth: float,
    gpu_name: str,
    key_avgs,
) -> str:
    """Build a human-readable roofline + MFU report."""
    ridge = gpu_peak_flops / gpu_peak_bandwidth
    lines: List[str] = []

    lines.append("=" * 72)
    lines.append("ROOFLINE ANALYSIS")
    lines.append(f"  GPU              : {gpu_name}")
    lines.append(f"  Peak compute     : {gpu_peak_flops / 1e12:.0f} TFLOPs/s")
    lines.append(f"  Peak bandwidth   : {gpu_peak_bandwidth / 1e9:.0f} GB/s")
    lines.append(f"  Ridge point      : {ridge:.1f} FLOPs/byte")
    time_source = metrics[0]["time_key"] if metrics else "cpu_time_total"
    timing_note = (
        "cuda_time_total" if time_source == "cuda_time_total"
        else "cpu_time_total (CUDA times unavailable in this PyTorch build)"
    )
    lines.append(f"  Timing source    : {timing_note}")
    lines.append(
        "  Memory bytes src : cuda_memory_usage (allocation proxy; not raw HBM bandwidth).\n"
        "                     For hardware-accurate values use NVIDIA Nsight Compute."
    )
    lines.append("=" * 72)

    mem_ops = [m for m in metrics if m["bound"] == "memory"]
    cmp_ops = [m for m in metrics if m["bound"] == "compute"]
    lines.append(f"\n  {len(mem_ops)} memory-bound op(s)  |  {len(cmp_ops)} compute-bound op(s)\n")

    header = (
        f"  {'Operator':<45} {'AI (F/B)':>10} {'GFLOPs/s':>10} "
        f"{'Bound':>14} {'CUDA ms':>9}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for m in metrics[:20]:
        lines.append(
            f"  {m['name']:<45} {m['ai']:>10.2f} {m['perf_gflops']:>10.1f} "
            f"  {m['bound']:>12}  {m['cuda_ms']:>8.3f}"
        )

    # MFU
    total_flops  = sum(getattr(e, "flops", 0) or 0 for e in key_avgs)
    total_cuda_s = sum(getattr(e, "cuda_time_total", 0) or 0 for e in key_avgs) / 1e6
    lines.append("")
    if total_cuda_s > 0 and total_flops > 0:
        mfu = total_flops / total_cuda_s / gpu_peak_flops * 100
        lines.append(f"  Model FLOP Utilization (MFU): {mfu:.2f}%")
        if mfu < 10:
            lines.append(
                f"  → MFU < 10%: memory bandwidth is the dominant bottleneck.\n"
                f"    Increasing batch size (e.g. vLLM continuous batching) raises\n"
                f"    arithmetic intensity and will improve GPU utilization."
            )
        elif mfu < 50:
            lines.append(
                f"  → MFU {mfu:.1f}%: mixed bottleneck. Profile prefill vs. decode\n"
                f"    separately to identify which phase limits throughput."
            )
        else:
            lines.append(
                f"  → MFU > 50%: strong compute utilization; approaching the compute ceiling."
            )
    else:
        lines.append("  MFU: insufficient FLOPs data (re-run on a CUDA device).")

    lines.append("\n" + "=" * 72)
    lines.append("INTERPRETATION FOR LLM INFERENCE")
    lines.append("=" * 72)
    lines.append(f"""
  Prefill (full prompt processed in parallel):
    • Large GEMM: M = prompt_length (often >> 1) → high arithmetic intensity
    • Tends to be COMPUTE-bound (above the ridge point)
    • Optimise with: bfloat16 Tensor Cores, FlashAttention, torch.compile

  Decode (one token at a time):
    • Loads the entire weight matrix to produce 1 output row (M = batch_size)
    • AI ≈ 2 / bytes_per_param ≈ 1–4 FLOPs/byte — far below ridge ({ridge:.0f} F/B)
    • Strongly MEMORY-BOUND; adding more compute does NOT help
    • Optimise with:
        - Larger effective batch size → continuous batching (vLLM PagedAttention)
        - Quantization: INT8/INT4 halves weight bytes → doubles AI
        - KV-cache in lower precision (reduces decode memory traffic)
        - Speculative decoding: amortises decode cost across multiple accepted tokens

  Why vLLM outperforms HuggingFace on your serving benchmark:
    • Continuous batching keeps decode batch size high → AI increases
    • PagedAttention reduces KV-cache waste → room for larger batches
    • Both push decode arithmetic intensity toward the ridge point
""")
    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    tb_dir  = ensure_dir(str(out_dir / "tb_trace"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16 if torch.cuda.is_available()
        else torch.float32
    )

    print(f"Device : {device}  |  dtype : {dtype}")
    print(f"Output : {out_dir}")

    examples = load_examples(args.input_path, args.limit)
    prompts  = [build_prompt(ex) for ex in examples]

    # ---- Load model ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    # ---- Warm-up (not profiled) ----
    print("\nWarming up (1 example, not profiled)...")
    with torch.no_grad():
        warm_inputs = tokenizer(prompts[0], return_tensors="pt")
        if device == "cuda":
            warm_inputs = {k: v.to(device) for k, v in warm_inputs.items()}
        _ = model(**warm_inputs)
    if device == "cuda":
        torch.cuda.synchronize()
    print("Warm-up done.\n")

    # ---- Profiled run ----
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    # No schedule: record all steps continuously so key_averages() sees the full
    # data after the context manager exits (a schedule resets events after each
    # active window, leaving key_averages() empty).
    # TensorBoard export is triggered manually after profiling instead.
    profiler_kwargs: Dict[str, Any] = dict(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,  # with_stack=True interferes with CUDA timing attribution
    )
    try:
        profiler_kwargs["with_flops"] = True
    except TypeError:
        pass  # older PyTorch

    # Group prompts into batches
    bs = args.batch_size
    batches = [prompts[i : i + bs] for i in range(0, len(prompts), bs)]
    print(f"Profiling {len(prompts)} example(s) in {len(batches)} batch(es) of up to {bs}...\n")

    with profile(**profiler_kwargs) as prof:
        for step, batch_prompts in enumerate(batches):
            # Tokenise with padding so all sequences in a batch are the same length
            with record_function("tokenization"):
                enc = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )

            # Host → device transfer
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
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    decode_steps_to_trace=args.decode_steps_to_trace,
                )
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            input_len  = input_ids.shape[1]
            output_len = generated.shape[1] - input_len
            total_new_tokens = output_len * input_ids.shape[0]
            print(
                f"[batch {step + 1}/{len(batches)}]  "
                f"batch_size={input_ids.shape[0]}  "
                f"input_len={input_len}  output_len={output_len}  "
                f"latency={t1 - t0:.3f}s  "
                f"tok/s={total_new_tokens / (t1 - t0):.1f}"
            )

            prof.step()

    # ---- Export Chrome trace ----
    # kineto_results.save() can only be called once, so export_chrome_trace is
    # called first, then the file is copied to the TensorBoard directory with
    # the *.pt.trace.json filename pattern that TensorBoard's profiler plugin expects.
    chrome_path = str(out_dir / "chrome_trace.json")
    prof.export_chrome_trace(chrome_path)
    print(f"\nChrome trace saved → {chrome_path}")
    print("  View at: ui.perfetto.dev  or  chrome://tracing")

    # ---- Copy to TensorBoard directory ----
    tb_filename = f"worker0.{int(time.time() * 1000)}.pt.trace.json"
    shutil.copy(chrome_path, str(tb_dir / tb_filename))
    print(f"TensorBoard trace  → {tb_dir / tb_filename}")

    # ---- Operator table ----
    key_avgs = prof.key_averages(group_by_input_shape=False)
    # Use CUDA times only if they are actually populated (some PyTorch builds report 0)
    has_cuda_times = any(getattr(e, "cuda_time_total", 0) > 0 for e in key_avgs)
    time_key = "cuda_time_total" if (device == "cuda" and has_cuda_times) else "cpu_time_total"
    if device == "cuda" and not has_cuda_times:
        print("\nNote: CUDA times are zero — reporting CPU times instead. "
              "This is normal when CUDA kernels run asynchronously and attribution "
              "is unavailable in this PyTorch build.")
    sorted_ops = sorted(key_avgs, key=lambda e: getattr(e, time_key, 0), reverse=True)

    csv_path = str(out_dir / "operators.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["operator", "cuda_time_ms", "cpu_time_ms", "calls",
                         "cuda_memory_usage_mb", "self_cuda_time_ms"])
        for e in sorted_ops[: args.top_ops]:
            writer.writerow([
                e.key,
                f"{getattr(e, 'cuda_time_total', 0) / 1e3:.4f}",
                f"{e.cpu_time_total / 1e3:.4f}",
                e.count,
                f"{getattr(e, 'cuda_memory_usage', 0) / 1e6:.4f}",
                f"{getattr(e, 'self_cuda_time_total', 0) / 1e3:.4f}",
            ])
    print(f"Operator CSV saved  → {csv_path}")

    # ---- Bottleneck report ----
    report = analyze_bottlenecks(key_avgs, top_n=args.top_ops, device=device, time_key=time_key)
    print("\n" + report)

    report_path = str(out_dir / "bottleneck_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nBottleneck report   → {report_path}")

    # ---- Roofline analysis ----
    gpu_peak_flops = args.gpu_peak_tflops * 1e12
    gpu_peak_bandwidth = args.gpu_peak_bandwidth_gbs * 1e9

    roofline_metrics = compute_roofline_metrics(key_avgs, gpu_peak_flops, gpu_peak_bandwidth)
    roofline_report = format_roofline_report(
        roofline_metrics, gpu_peak_flops, gpu_peak_bandwidth, args.gpu_name, key_avgs
    )
    print("\n" + roofline_report)

    roofline_report_path = str(out_dir / "roofline_report.txt")
    with open(roofline_report_path, "w") as f:
        f.write(roofline_report)
    print(f"Roofline report     → {roofline_report_path}")

    roofline_plot_path = str(out_dir / "roofline.png")
    if roofline_metrics:
        plot_roofline(
            roofline_metrics, gpu_peak_flops, gpu_peak_bandwidth,
            args.gpu_name, roofline_plot_path,
        )
    else:
        print("No FLOPs data available for roofline plot — re-run on a CUDA device.")

    print(f"TensorBoard traces  → {tb_dir}")
    print("\nTo view in TensorBoard:")
    print(f"  tensorboard --logdir {tb_dir}")


if __name__ == "__main__":
    main()
