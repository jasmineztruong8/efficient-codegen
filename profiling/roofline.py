"""roofline.py — analytical roofline model for LLM operator profiling."""

from __future__ import annotations

from typing import Any, cast, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def analytical_mem_bytes(op_name: str, input_shapes, dtype_bytes: int = 2) -> int:
    """
    Analytically estimate HBM traffic (bytes read + written) for common ops.

    Uses a read-once / write-once model. dtype_bytes=2 for bfloat16/float16.
    Returns 0 for unrecognised ops or malformed shapes.
    """
    if not input_shapes:
        return 0
    shapes = [list(s) for s in input_shapes if s]

    def prod(dims: list) -> int:
        r = 1
        for d in dims:
            r *= max(int(d), 1)
        return r

    try:
        if op_name == "aten::mm":
            if len(shapes) >= 2 and len(shapes[0]) == 2 and len(shapes[1]) == 2:
                M, K = shapes[0]; _, N = shapes[1]
                return (M * K + K * N + M * N) * dtype_bytes

        elif op_name == "aten::bmm":
            if len(shapes) >= 2 and len(shapes[0]) == 3 and len(shapes[1]) == 3:
                B, M, K = shapes[0]; _, _, N = shapes[1]
                return B * (M * K + K * N + M * N) * dtype_bytes

        elif op_name == "aten::addmm":
            if len(shapes) >= 3 and len(shapes[1]) == 2 and len(shapes[2]) == 2:
                M, K = shapes[1]; _, N = shapes[2]
                bias_elems = shapes[0][0] if shapes[0] else N
                return (M * K + K * N + bias_elems + M * N) * dtype_bytes

        elif op_name == "aten::linear":
            if len(shapes) >= 2 and len(shapes[1]) == 2:
                in_shape = shapes[0]; out_f, in_f = shapes[1]
                M = prod(in_shape[:-1]) if len(in_shape) > 1 else 1
                bias_elems = shapes[2][0] if len(shapes) > 2 and shapes[2] else 0
                return (M * in_f + out_f * in_f + bias_elems + M * out_f) * dtype_bytes

        elif op_name in ("aten::mul", "aten::add", "aten::sub", "aten::div"):
            if shapes:
                return 3 * prod(shapes[0]) * dtype_bytes

        elif op_name in ("aten::relu", "aten::gelu", "aten::silu", "aten::sigmoid"):
            if shapes:
                return 2 * prod(shapes[0]) * dtype_bytes

    except (IndexError, TypeError, ValueError, ZeroDivisionError):
        pass

    return 0


def compute_roofline_metrics(
    key_avgs,
    gpu_peak_flops: float,
    gpu_peak_bandwidth: float,
    time_key: str = "",
    analytical_dtype_bytes: int = 2,
) -> List[Dict[str, Any]]:
    """
    Compute arithmetic intensity and attainable performance per operator.

    Memory bytes priority:
      1. Analytical shape estimate (accurate for matmul / elementwise ops)
      2. cuda_memory_usage allocation proxy
      3. Ridge-point fallback

    Pass key_avgs from prof.key_averages(group_by_input_shape=True) to enable
    analytical estimation.
    """
    if not time_key:
        has_cuda = any(getattr(e, "cuda_time_total", 0) > 0 for e in key_avgs)
        time_key = "cuda_time_total" if has_cuda else "cpu_time_total"

    ridge_point = gpu_peak_flops / gpu_peak_bandwidth
    metrics: List[Dict[str, Any]] = []
    for e in key_avgs:
        flops = getattr(e, "flops", 0) or 0
        if flops <= 0:
            continue

        input_shapes = getattr(e, "input_shapes", None)
        mem_bytes = analytical_mem_bytes(e.key, input_shapes, analytical_dtype_bytes)
        mem_source = "analytical"

        if mem_bytes == 0:
            mem_bytes = abs(getattr(e, "cuda_memory_usage", 0) or 0)
            mem_source = "allocation_proxy"

        if mem_bytes == 0:
            mem_bytes = flops / ridge_point
            mem_source = "ridge_fallback"

        time_us = getattr(e, time_key, 0) or 0
        if time_us <= 0:
            continue

        ai = flops / mem_bytes
        metrics.append({
            "name":       e.key,
            "ai":         ai,
            "perf_gflops": flops / (time_us / 1e6) / 1e9,
            "bound":      "compute" if ai >= ridge_point else "memory",
            "cuda_ms":    time_us / 1e3,
            "flops":      flops,
            "mem_bytes":  mem_bytes,
            "mem_source": mem_source,
            "count":      e.count,
            "time_key":   time_key,
        })
    return sorted(metrics, key=lambda x: x["cuda_ms"], reverse=True)


def _aggregate_by_name(metrics: List[Dict[str, Any]], ridge: float) -> List[Dict[str, Any]]:
    """Collapse per-shape entries into one point per op name (sum FLOPs + time, recompute AI)."""
    agg: Dict[str, Dict[str, Any]] = {}
    for m in metrics:
        name = m["name"]
        if name not in agg:
            agg[name] = {"flops": 0, "mem_bytes": 0, "cuda_ms": 0.0, "mem_source": m["mem_source"]}
        agg[name]["flops"]     += m["flops"]
        agg[name]["mem_bytes"] += m["mem_bytes"]
        agg[name]["cuda_ms"]   += m["cuda_ms"]

    result = []
    for name, v in agg.items():
        ai         = v["flops"] / v["mem_bytes"] if v["mem_bytes"] > 0 else ridge
        cuda_s     = v["cuda_ms"] / 1e3
        perf_gflops = v["flops"] / cuda_s / 1e9 if cuda_s > 0 else 0
        result.append({
            "name":        name,
            "ai":          ai,
            "perf_gflops": perf_gflops,
            "bound":       "compute" if ai >= ridge else "memory",
            "cuda_ms":     v["cuda_ms"],
            "flops":       v["flops"],
            "mem_bytes":   v["mem_bytes"],
            "mem_source":  v["mem_source"],
        })
    return result


def plot_roofline(
    metrics: List[Dict[str, Any]],
    gpu_peak_flops: float,
    gpu_peak_bandwidth: float,
    gpu_name: str,
    out_path: str,
) -> None:
    """Save a log-log roofline chart — one point per op name (shapes aggregated)."""
    ridge       = gpu_peak_flops / gpu_peak_bandwidth
    metrics     = _aggregate_by_name(metrics, ridge)
    peak_gflops = gpu_peak_flops / 1e9
    peak_gbps   = gpu_peak_bandwidth / 1e9

    _, ax = plt.subplots(figsize=(13, 7))

    ai_lo    = min(m["ai"] for m in metrics) * 0.1 if metrics else 1e-2
    ai_hi    = max(m["ai"] for m in metrics) * 10  if metrics else ridge * 10
    ai_range = np.logspace(np.log10(max(ai_lo, 1e-3)), np.log10(ai_hi), 600)
    ax.plot(ai_range, np.minimum(gpu_peak_bandwidth * ai_range, gpu_peak_flops) / 1e9,
            "k-", linewidth=2.5, label="Roofline ceiling", zorder=3)

    ax.axvline(x=ridge, color="gray", linestyle="--", alpha=0.6,
               label=f"Ridge point ({ridge:.0f} FLOPs/byte)")

    total_flops    = sum(m["flops"]     for m in metrics)
    total_membytes = sum(m["mem_bytes"] for m in metrics)
    if total_membytes > 0:
        eff_ai  = total_flops / total_membytes
        overall = "memory-bound" if eff_ai < ridge else "compute-bound"
        ax.axvline(x=eff_ai, color="darkorange", linestyle=":", linewidth=2, alpha=0.9,
                   label=f"Effective AI ({eff_ai:.1f} F/B) — {overall}")

    ax.axvspan(ai_lo * 0.5, ridge,  alpha=0.04, color="steelblue")
    ax.axvspan(ridge, ai_hi * 2,    alpha=0.04, color="crimson")
    ax.text(ridge * 0.05, peak_gflops * 0.3, "Memory-bound",  fontsize=9, color="steelblue", alpha=0.7)
    ax.text(ridge * 2.0,  peak_gflops * 0.3, "Compute-bound", fontsize=9, color="crimson",   alpha=0.7)

    for bound, color, marker in [("memory", "steelblue", "o"), ("compute", "crimson", "^")]:
        ops = [m for m in metrics if m["bound"] == bound and m["perf_gflops"] > 0]
        if not ops:
            continue
        xs = [m["ai"] for m in ops]; ys = [m["perf_gflops"] for m in ops]
        ax.scatter(xs, ys, c=color, s=110, marker=cast(Any, marker), zorder=5,
                   label=f"{'Memory' if bound == 'memory' else 'Compute'}-bound ops",
                   alpha=0.85, edgecolors="white", linewidths=0.6)
        for x, y, m in zip(xs, ys, ops):
            short = m["name"].replace("aten::", "").replace("cuda::", "")[:18]
            ax.annotate(short, (x, y), fontsize=6.5, alpha=0.75, xytext=(5, 4),
                        textcoords="offset points")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity  (FLOPs / byte)", fontsize=13)
    ax.set_ylabel("Attainable Performance  (GFLOPs/s)",   fontsize=13)
    ax.set_title(
        f"Roofline Model — {gpu_name}\n"
        f"Peak compute: {peak_gflops/1e3:.0f} TFLOPs/s  |  "
        f"Peak BW: {peak_gbps:.0f} GB/s  |  Ridge: {ridge:.0f} FLOPs/byte",
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
    """Build a human-readable roofline + MFU summary."""
    ridge = gpu_peak_flops / gpu_peak_bandwidth
    lines: List[str] = []

    n_analytical = sum(1 for m in metrics if m.get("mem_source") == "analytical")
    n_proxy      = sum(1 for m in metrics if m.get("mem_source") == "allocation_proxy")
    n_fallback   = sum(1 for m in metrics if m.get("mem_source") == "ridge_fallback")
    time_source  = metrics[0]["time_key"] if metrics else "cpu_time_total"
    timing_note  = ("cuda_time_total" if time_source == "cuda_time_total"
                    else "cpu_time_total (CUDA times unavailable in this PyTorch build)")

    lines += [
        "=" * 72, "ROOFLINE ANALYSIS",
        f"  GPU              : {gpu_name}",
        f"  Peak compute     : {gpu_peak_flops/1e12:.0f} TFLOPs/s",
        f"  Peak bandwidth   : {gpu_peak_bandwidth/1e9:.0f} GB/s",
        f"  Ridge point      : {ridge:.1f} FLOPs/byte",
        f"  Timing source    : {timing_note}",
        f"  Memory bytes src : analytical ({n_analytical} ops), "
        f"allocation proxy ({n_proxy}), ridge fallback ({n_fallback}).\n"
        "                     For hardware-accurate values use NVIDIA Nsight Compute.",
        "=" * 72,
    ]

    mem_ops = [m for m in metrics if m["bound"] == "memory"]
    cmp_ops = [m for m in metrics if m["bound"] == "compute"]
    mem_ms  = sum(m["cuda_ms"] for m in mem_ops)
    cmp_ms  = sum(m["cuda_ms"] for m in cmp_ops)
    total_ms = mem_ms + cmp_ms or 1.0

    total_flops    = sum(m["flops"]     for m in metrics)
    total_membytes = sum(m["mem_bytes"] for m in metrics)
    eff_ai  = total_flops / total_membytes if total_membytes > 0 else 0.0
    overall = "MEMORY-BOUND" if eff_ai < ridge else "COMPUTE-BOUND"

    lines += [
        f"\n  {len(mem_ops)} memory-bound op(s)  |  {len(cmp_ops)} compute-bound op(s)",
        f"\n  Time in memory-bound ops : {mem_ms:8.1f} ms  ({mem_ms/total_ms*100:.1f}%)",
        f"  Time in compute-bound ops: {cmp_ms:8.1f} ms  ({cmp_ms/total_ms*100:.1f}%)",
        f"\n  Effective arithmetic intensity : {eff_ai:.1f} FLOPs/byte",
        f"  Ridge point                   : {ridge:.1f} FLOPs/byte",
        f"  Overall workload              : {overall}\n",
    ]

    header = (f"  {'Operator':<45} {'AI (F/B)':>10} {'GFLOPs/s':>10} "
              f"{'Bound':>14} {'CUDA ms':>9} {'Mem src':>16}")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for m in metrics[:20]:
        lines.append(
            f"  {m['name']:<45} {m['ai']:>10.2f} {m['perf_gflops']:>10.1f} "
            f"  {m['bound']:>12}  {m['cuda_ms']:>8.3f}  {m.get('mem_source',''):>14}"
        )

    total_flops_all  = sum(getattr(e, "flops", 0) or 0 for e in key_avgs)
    total_cuda_s     = sum(getattr(e, "cuda_time_total", 0) or 0 for e in key_avgs) / 1e6
    lines.append("")
    if total_cuda_s > 0 and total_flops_all > 0:
        mfu = total_flops_all / total_cuda_s / gpu_peak_flops * 100
        lines.append(f"  Model FLOP Utilization (MFU): {mfu:.2f}%")
        if mfu < 10:
            lines.append("  → MFU < 10%: memory bandwidth is the dominant bottleneck.")
        elif mfu < 50:
            lines.append(f"  → MFU {mfu:.1f}%: mixed bottleneck.")
        else:
            lines.append("  → MFU > 50%: strong compute utilization.")
    else:
        lines.append("  MFU: insufficient FLOPs data (re-run on a CUDA device).")

    lines += [
        "\n" + "=" * 72, "INTERPRETATION FOR LLM INFERENCE", "=" * 72,
        f"""
  Prefill (full prompt processed in parallel):
    • Large GEMM: M = prompt_length → high arithmetic intensity → COMPUTE-bound
    • Optimise with: bfloat16 Tensor Cores, FlashAttention, torch.compile

  Decode (one token at a time, small batch):
    • AI ≈ 2 / bytes_per_param ≈ 1–4 FLOPs/byte — far below ridge ({ridge:.0f} F/B)
    • Strongly MEMORY-BOUND; adding compute does NOT help
    • Optimise with: larger batch (vLLM continuous batching), INT8/INT4 quantization,
      KV-cache in lower precision, speculative decoding

  Why vLLM outperforms HuggingFace on decode:
    • Continuous batching + PagedAttention raise effective batch size → AI increases
""",
        "=" * 72,
    ]
    return "\n".join(lines)
