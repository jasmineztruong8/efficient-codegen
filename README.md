# efficient-codegen
**Optimization of SLMs for Efficient Code Generation**

COMS 6998E High Performance Machine Learning — Columbia University, Spring 2026

Team: Jasmine Truong, Yingxin Zhang, Arnav Mahajan, Jianyi Gao

---

## Overview

We fine-tune `Qwen2.5-Coder-1.5B-Instruct` to generate not just correct code, but *fast* code. Standard code LLMs optimize for Pass@1; our goal is to reduce execution time of generated programs while maintaining near-baseline correctness.

**Approach:**
1. Curate a clean subset of the [EffiCoder dataset](https://arxiv.org/abs/2410.10209)
2. Generate multiple candidate solutions per problem using an SLM
3. Select the fastest correct candidate per problem to build a runtime-aware fine-tuning dataset
4. Fine-tune with qLoRA and compare against a control SFT and base SLM

---

## Repository Structure

```
efficient-codegen/
├── data/
│   ├── raw/                        # Raw EffiCoder dataset (not committed)
│   └── curated/
│       ├── scale_full/             # Full ~6.6k clean dataset
│       ├── scale1k/                # 1k subset for validation
│       └── prototyping/            # 20-sample profiling set
├── scripts/
│   ├── select_dataset.py           # Filter and score problems from raw dataset
│   ├── filter_passing.py           # Keep only problems whose reference solution passes tests
│   ├── expand_candidates.py        # Expand a base set with additional candidates (prototyping)
│   ├── merge_candidates.py         # Merge base + expansion batch (prototyping)
│   ├── review_candidates.py        # Inspect candidate scores (utility)
│   ├── quick_check.py              # Sanity check dataset (utility)
│   ├── run_pipeline_full.sh        # End-to-end pipeline: full ~6.9k
│   ├── run_pipeline_1k.sh          # Slice 1k from full dataset
│   └── run_pipeline_20.sh          # Slice 20-sample profiling set
├── generation/
│   └── generate_candidates.py      # Generate N candidate solutions per problem via LLM
├── execution/
│   ├── benchmark.py                # Benchmark reference solutions (subprocess-isolated)
│   ├── evaluate_candidates.py      # Check correctness of generated candidates
│   ├── filter_passing_candidates.py# Keep only passing candidates
│   └── benchmark_candidates.py     # Time passing candidates (repeated runs, median)
├── profiling/
│   ├── profile_model.py            # PyTorch Profiler + WandB logging (high-level metrics)
│   └── profile_operators.py        # Operator-level trace with bottleneck analysis
├── outputs/
│   ├── generated_candidates_full.jsonl   # 6,646 problems × 5 candidates (committed)
│   ├── evaluated_candidates_full.jsonl   # Pass/fail per candidate (committed)
│   └── benchmarked_candidates_full.jsonl # Median runtime for passing candidates (committed)
└── notebooks/
    └── prototype_pipeline_colab.ipynb  # Full Colab pipeline (generation → benchmark)
```

---

## Reproducing the Dataset (Local)

### Setup

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> Activate the venv (`source venv/bin/activate`) before running any script in this repo.

### Step 1: Obtain the raw dataset

Download `efficoder.json` and place it at `data/raw/efficoder.json`.

### Step 2: Run the data pipeline

```bash
# Build the full clean dataset (~6.6k problems, takes several hours)
bash scripts/run_pipeline_full.sh

# Slice subsets
bash scripts/run_pipeline_1k.sh    # → data/curated/scale1k/dataset_clean.json
bash scripts/run_pipeline_20.sh    # → data/curated/prototyping/prototype_final_20_clean.json
```

**Pipeline steps inside `run_pipeline_full.sh`:**
1. `select_dataset.py` — filters and scores all ~9.4k EffiCoder problems, keeps ~6.9k eligible
2. `benchmark.py` — runs each reference solution 7× in an isolated subprocess, records median runtime
3. `filter_passing.py` — keeps only problems whose reference solution passes all tests (~96% pass rate)

**Output:** `data/curated/scale_full/dataset_clean.json` (~6.6k problems)

---

## Generation + Profiling (Colab)

Open `notebooks/prototype_pipeline_colab.ipynb` in Google Colab. Use a G4 (NVIDIA RTX PRO 6000 Blackwell, 94GB) or A100 for generation. T4/L4 is sufficient for profiling only.

The notebook covers:
1. Environment setup and repo clone
2. Smoke test (2 problems, 2 candidates)
3. Full generation run — generates `NUM_CANDIDATES` solutions per problem via `Qwen2.5-Coder-1.5B-Instruct`
4. Correctness evaluation
5. Filter passing candidates
6. Runtime benchmarking of passing candidates
7. PyTorch Profiler run with WandB logging

**Recommended batch size:** `--batch_size 64` on G4 (improves GPU utilization from 34% → 52% vs batch=8)

**Required Colab secrets:** `GITHUB_TOKEN`, `WANDB_API_KEY`

---

## Key Scripts

| Script | Purpose | Key Args |
|--------|---------|----------|
| `scripts/select_dataset.py` | Filter & score problems | `--limit`, `--output` |
| `execution/benchmark.py` | Benchmark reference solutions | `--input`, `--output`, `--repeats` |
| `scripts/filter_passing.py` | Keep passing problems | `--bench`, `--candidates`, `--output`, `--limit` |
| `generation/generate_candidates.py` | Generate candidate solutions | `--input_path`, `--num_candidates`, `--model_name`, `--limit` |
| `execution/evaluate_candidates.py` | Check candidate correctness | `--input_path`, `--output_path`, `--limit` |
| `execution/benchmark_candidates.py` | Time passing candidates | `--input_path`, `--output_path`, `--num_runs`, `--warmup_runs` |
| `profiling/profile_model.py` | PyTorch Profiler + WandB | `--input_path`, `--limit`, `--batch_size`, `--use_wandb` |
| `profiling/profile_operators.py` | Operator-level trace, bottleneck report, Chrome trace | `--input_path`, `--limit`, `--batch_size`, `--output_dir` |

---

## Profiling Results (Baseline)

Profiled on NVIDIA RTX PRO 6000 Blackwell (94GB, CC 12.0) using `Qwen2.5-Coder-1.5B-Instruct` in bfloat16.

### batch_size=8 vs batch_size=64

Profiled on 20 samples (batch=8) and 320 samples / 5 batches (batch=64).

| Metric | batch=8 | batch=64 |
|--------|---------|---------|
| GPU Utilization | 34.33% | 42.71% |
| Est. SM Efficiency | 17.8% | 35.37% |
| Est. Achieved Occupancy | 9.74% | 31.73% |
| Kernel share of step time | 34.3% | 42.76% |
| CPU Exec share of step time | 53.6% | 50.86% |
| Tensor Core utilization | ~1% | ~6.7% |

### Key bottlenecks identified

1. **CPU-bound autoregressive decode loop** — CPU execution accounts for 50.9% of step time at batch=64. `cudaLaunchKernel` is called 811k times across 5 batches, indicating high Python dispatch overhead per token.
2. **Per-token CPU-GPU synchronizations** — `aten::item` and `cudaStreamSynchronize` account for 17.8% of total profiled time (2,550 calls), triggered by EOS token detection inside the generate loop.
3. **Low Tensor Core utilization** — only 6.7% of kernel time uses Tensor Cores despite bfloat16. During decode, M-dimension equals batch size (64), which is too small to saturate Tensor Core tiles.
4. **No data-loading bottleneck** — tokenization (<5ms per batch) and H2D transfer are negligible.

### Proposed optimizations
- `torch.compile(model, mode='reduce-overhead')` — fuses elementwise ops, eliminates Python dispatch overhead in decode loop
- `synced_gpus=False` in `model.generate()` — reduces unnecessary CPU-GPU syncs during EOS detection
- batch_size ≥ 64 for inference — already validated above

Full operator-level trace: `outputs/operator_profile/bottleneck_report.txt`

---

## Outputs

Key output files committed to the repo (generated on G4/Colab using `Qwen2.5-Coder-1.5B-Instruct`):

| File | Description |
|------|-------------|
| `outputs/generated_candidates_full.jsonl` | 6,646 problems × 5 candidates = 33,230 generated solutions |
| `outputs/evaluated_candidates_full.jsonl` | Pass/fail result for each candidate (Pass@1=34.2%, Pass@5=46.0%) |
| `outputs/benchmarked_candidates_full.jsonl` | Median runtime for each passing candidate (7 runs, 1 warmup) |

Intermediate files (gitignored, re-derivable):
- `outputs/passing_candidates_full.jsonl` — filtered subset of evaluated, passing only

---

## Experiment Tracking

WandB project: [hpml-efficient-codegen](https://wandb.ai/yz3202-columbia-university/hpml-efficient-codegen)

---

## References

- Huang et al. [EffiCoder](https://arxiv.org/abs/2410.10209), 2025
- Nichols et al. [Performance-Aligned LLMs for Generating Fast Code](https://arxiv.org/abs/2404.18864), 2024
- Hui et al. [Qwen2.5-Coder Technical Report](https://arxiv.org/abs/2409.12186), 2024
- Dettmers et al. [QLoRA](https://arxiv.org/abs/2305.14314), 2023
