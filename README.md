# efficient-codegen
**Optimization of SLMs for Efficient Code Generation**

COMS 6998E High Performance Machine Learning — Columbia University, Spring 2026

Team: Jasmine Truong, Yingxin Zhang, Arnav Mahajan, Jianyi Gao

---

## Overview

We fine-tune `Qwen2.5-Coder-1.5B-Instruct` to generate not just correct code, but *fast* code. Standard code LLMs optimize for Pass@1; our goal is to reduce execution time of generated programs while maintaining near-baseline correctness.

**Approach:**
1. Curate a clean subset of the [EffiCoder dataset](https://arxiv.org/abs/2410.10209)
2. Generate multiple candidate solutions per problem using the base SLM
3. Select the fastest correct candidate per problem to build a runtime-aware fine-tuning dataset
4. Fine-tune with qLoRA and compare against a control SFT and base SLM
5. Profile GPU performance and benchmark serving throughput (HuggingFace vs vLLM)

---

## Repository Structure

```
efficient-codegen/
├── data/
│   ├── raw/                        # Raw EffiCoder dataset (not committed)
│   └── curated/
│       ├── scale_full/             # Full ~6.6k clean dataset
│       ├── scale1k/                # 1k subset for ablation evaluation
│       └── prototyping/            # 20-sample profiling set
├── scripts/
│   ├── select_dataset.py           # Filter and score problems from raw dataset
│   ├── filter_passing.py           # Keep only problems whose reference solution passes tests
│   ├── expand_candidates.py        # Expand a base set with additional candidates
│   ├── merge_candidates.py         # Merge base + expansion batch
│   ├── review_candidates.py        # Inspect candidate scores (utility)
│   ├── quick_check.py              # Sanity check dataset (utility)
│   ├── run_pipeline_full.sh        # End-to-end pipeline: full ~6.6k
│   ├── run_pipeline_1k.sh          # Slice 1k from full dataset
│   └── run_pipeline_20.sh          # Slice 20-sample profiling set
├── generation/
│   └── generate_candidates.py      # Generate N candidate solutions per problem via LLM
├── execution/
│   ├── benchmark.py                # Benchmark reference solutions (subprocess-isolated)
│   ├── evaluate_candidates.py      # Check correctness of generated candidates
│   ├── filter_passing_candidates.py# Keep only passing candidates
│   └── benchmark_candidates.py     # Time passing candidates (repeated runs, median)
├── training/
│   ├── train.py                    # qLoRA fine-tuning with SFTTrainer
│   ├── evaluate_model.py           # Generate + evaluate candidates for a given model
│   └── select_training_data.py     # Build runtime_aware.jsonl and control.jsonl
├── serving/
│   ├── merge_checkpoint.py         # Merge LoRA adapter into base model weights
│   └── benchmark_serving.py        # Benchmark HuggingFace, vLLM, and SGLang serving backends
├── profiling/
│   ├── profile_model.py            # PyTorch Profiler + W&B logging (high-level metrics)
│   └── profile_operators.py        # Operator-level trace with bottleneck analysis
├── outputs/
│   ├── generated_candidates_full.jsonl     # 6,646 problems x 5 candidates
│   ├── evaluated_candidates_full.jsonl     # Pass/fail per candidate
│   ├── benchmarked_candidates_full.jsonl   # Median runtime for passing candidates
│   ├── serving_full_results.csv            # HF vs vLLM serving benchmark results
│   ├── wandb_ablation_runs_2026-04-20.csv  # Ablation study W&B export
│   └── wandb_profiled_runs_2026-04-20.csv  # Profiling W&B export
└── notebooks/
    ├── base_data_generation_evaluation_pipeline.ipynb  # Generation → evaluation → benchmark (Vast.ai)
    ├── base_model_gpu_profiling.ipynb                  # Base model profiling with profile_model.py + profile_operators.py
    ├── gpu_profiling_comparison.ipynb                  # 3-model GPU profiling comparison (base, control, runtime-aware)
    ├── results.ipynb                                   # Ablation + serving results summary tables
    ├── efficient_codegen_experiments.ipynb             # Training runs (Colab)
    └── efficient_codegen_system_optimization.ipynb     # Serving benchmarks (Colab)
```

---

## Phase 1: Data Curation (Local)

### Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 1: Obtain the raw dataset

Download `efficoder.json` and place it at `data/raw/efficoder.json`.

### Step 2: Run the data pipeline

```bash
bash scripts/run_pipeline_full.sh   # -> data/curated/scale_full/dataset_clean.json (~6.6k problems)
bash scripts/run_pipeline_1k.sh     # -> data/curated/scale1k/dataset_clean.json (1k subset)
bash scripts/run_pipeline_20.sh     # -> data/curated/prototyping/prototype_final_20_clean.json
```

**Pipeline steps:**
1. `select_dataset.py` — filters ~9.4k EffiCoder problems, keeps ~6.9k eligible
2. `benchmark.py` — runs each reference solution 7x in an isolated subprocess, records median runtime
3. `filter_passing.py` — keeps only problems whose reference solution passes all tests (~96% pass rate)

---

## Phase 2: Candidate Generation & Benchmarking (Vast.ai / A100)

Use a Vast.ai instance with a CUDA/PyTorch image (A100 recommended, minimum 80GB disk).

### `notebooks/base_data_generation_evaluation_pipeline.ipynb`

Runs the full generation and evaluation pipeline on `scale_full` (6,646 problems):

1. Generate 5 candidate solutions per problem using `Qwen2.5-Coder-1.5B-Instruct`
2. Evaluate candidate correctness → `outputs/evaluated_candidates_full.jsonl` (33,230 candidates)
3. Filter passing candidates
4. Benchmark passing candidates (7 runs, 1 warmup, median) → `outputs/benchmarked_candidates_full.jsonl` (7,992 passing)

**Output used for training:** 7,992 benchmarked candidates → 2,110 training examples per model (via `select_training_data.py`)

---

## Phase 3: Fine-Tuning (GCP T4 + Colab)

### `notebooks/efficient_codegen_experiments.ipynb`

Model checkpoints (control and runtime-aware) were trained by Arnav on GCP T4 and stored on Google Drive. The notebook covers smoke tests and ablation evaluation on Colab.

#### Build training datasets

```bash
python training/select_training_data.py \
    --candidates_path outputs/benchmarked_candidates_full.jsonl \
    --dataset_path data/curated/scale_full/dataset_clean.json \
    --output_dir training/data
```

Outputs:
- `training/data/runtime_aware.jsonl` — fastest correct candidate per problem (2,110 examples)
- `training/data/control.jsonl` — first correct candidate per problem (2,110 examples)

#### Train models

```bash
# Control SFT
python training/train.py \
    --mode control \
    --data_path training/data/control.jsonl \
    --output_dir checkpoints/control_full \
    --use_wandb --wandb_project hpml-efficient-codegen

# Runtime-Aware SFT
python training/train.py \
    --mode runtime_aware \
    --data_path training/data/runtime_aware.jsonl \
    --output_dir checkpoints/runtime_aware_full \
    --use_wandb --wandb_project hpml-efficient-codegen
```

Checkpoints are stored on Google Drive and are not committed (~800MB LoRA adapters).

#### Ablation evaluation (scale1k, 1,000 problems)

```bash
python training/evaluate_model.py --model_path Qwen/Qwen2.5-Coder-1.5B-Instruct --run_name base_slm ...
python training/evaluate_model.py --model_path checkpoints/control_full --run_name control_sft ...
python training/evaluate_model.py --model_path checkpoints/runtime_aware_full --run_name runtime_aware_sft ...
```

---

## Phase 4: Serving Optimization (Colab)

### `notebooks/efficient_codegen_system_optimization.ipynb`

Compares HuggingFace Transformers, vLLM, and SGLang on base and runtime-aware models.

#### Merge LoRA adapter before vLLM

```bash
python serving/merge_checkpoint.py \
    --adapter_path checkpoints/runtime_aware_full \
    --base_model_name Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --output_dir checkpoints/runtime_aware_merged
```

#### Benchmark serving

```bash
python serving/benchmark_serving.py \
    --backend hf \
    --model_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --input_path data/curated/scale1k/dataset_clean.json \
    --limit 1000 --batch_size 8 \
    --output_path outputs/serving/base_hf_full.json
```

Use `--backend vllm` or `--backend sglang` to run the same benchmark against optimized serving engines.

Results saved to `outputs/serving_full_results.csv`.

---

## Phase 5: GPU Profiling (Vast.ai / A100)

### `notebooks/base_model_gpu_profiling.ipynb`

Profiles the base model (`Qwen2.5-Coder-1.5B-Instruct`) on the full dataset using:
- `profiling/profile_model.py` — PyTorch Profiler with W&B logging (latency, memory, throughput)
- `profiling/profile_operators.py` — operator-level trace with bottleneck report and Chrome trace

### `notebooks/gpu_profiling_comparison.ipynb`

Profiles all three models (base, control, runtime-aware) with identical settings for direct comparison:
- Merges LoRA adapters before profiling
- Saves TensorBoard traces to `outputs/gpu_profiling/<model>/`
- Produces side-by-side operator breakdown and bottleneck comparison

---

## Results

Ablation metrics match `training/evaluate_model.py` and W&B: median execution time is **seconds** (wall time for the fastest passing candidate’s tests per problem, aggregated as the median across problems).

### Ablation Study (scale1k, 1,000 problems)

| Model | Pass@1 | Median Exec Time (s) | Avg Gen Latency (s) |
|-------|--------|----------------------|---------------------|
| Base SLM | 0.369 | 0.0890 | 0.890 |
| Control SFT | 0.709 | 0.0906 | 0.753 |
| Runtime-Aware SFT | 0.702 | 0.0897 | 0.742 |

### Serving Performance (scale1k, 1,000 prompts, A100)

| Model | Backend | Throughput (tokens/s) | Avg GPU Util (%) |
|-------|---------|----------------------|-----------------|
| Base SLM | HuggingFace | 233.5 | 37.6% |
| Base SLM | vLLM | 1781.6 | 97.2% |
| Runtime-Aware SFT | HuggingFace | 228.6 | 37.3% |
| Runtime-Aware SFT | vLLM | 1891.1 | 96.8% |

### Key Profiling Findings

- **CPU-bound decode loop** — CPU execution accounts for ~51% of step time; `cudaLaunchKernel` called ~923k times across 5 batches
- **Per-token CPU-GPU syncs** — `aten::item` + `cudaStreamSynchronize` account for 6.7% of total time (EOS detection inside generate loop)
- **Low Tensor Core utilization** — ~6.7% of kernel time during decode due to small M-dimension (batch size)
- **vLLM provides 7.6x throughput improvement** over HuggingFace via PagedAttention and continuous batching

---

## Experiment Tracking

W&B project: [hpml-efficient-codegen](https://wandb.ai/efficient-codegen/hpml-efficient-codegen)

---

## References

- Huang et al. [EffiCoder](https://arxiv.org/abs/2410.10209), 2025
- Nichols et al. [Performance-Aligned LLMs for Generating Fast Code](https://arxiv.org/abs/2404.18864), 2024
- Hui et al. [Qwen2.5-Coder Technical Report](https://arxiv.org/abs/2409.12186), 2024
- Dettmers et al. [QLoRA](https://arxiv.org/abs/2305.14314), 2023
- Guo et al. [DeepSeek-Coder](https://arxiv.org/abs/2401.14196), 2024
