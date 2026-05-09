# HPML Final Project: Optimization of SLMs for Efficient Code Generation

> **Course:** High Performance Machine Learning
> **Semester:** Spring 2026
> **Instructor:** Dr. Kaoutar El Maghraoui

---

## Team Information

- **Team Name:** efficient-codegen
- **Members:**
  - Arnav Mahajan ([UNI]) — *role / area of contribution*
  - Jasmine Truong ([UNI]) — *role / area of contribution*
  - Jianyi Gao ([UNI]) — *role / area of contribution*
  - Yingxin Zhang (yz3202) — *data pipeline,  profiling pipeline and results analysis, integrated W&B/TensorBoard tracking, report writing*

## Submission

- **GitHub repository:** [https://github.com/jasmineztruong8/efficient-codegen](https://github.com/jasmineztruong8/efficient-codegen)
- **Final report:** [`deliverables/Optimizationof SLMs for Efficient Code Generation.pdf`](deliverables/Optimizationof SLMs for Efficient Code Generation.pdf)
- **Final presentation:** [`deliverables/HPML_Final_Presentation.pptx`](deliverables/HPML_Final_Presentation.pptx)
- **Experiment-tracking dashboard:** [https://wandb.ai/efficient-codegen/hpml-efficient-codegen](https://wandb.ai/efficient-codegen/hpml-efficient-codegen)

The final report PDF and the presentation file are checked into the `deliverables/` folder of this repository **and** uploaded to CourseWorks.

---

## 1. Problem Statement

Standard code-generation LLMs are trained to maximize correctness (Pass@1) but are blind to the runtime efficiency of the code they produce — a correct O(n²) solution and a correct O(n log n) solution look identical to the model. We fine-tune `Qwen2.5-Coder-1.5B-Instruct`, a 1.5B-parameter small language model (SLM), to generate not just correct code but *fast* code, targeting **inference-time** optimization of both the generated programs and the model's own serving throughput. The primary bottlenecks we address are (1) training-data bias toward correctness over performance and (2) memory-bandwidth saturation during autoregressive decode, which limits serving throughput under HuggingFace's default generation loop.

---

## 2. Model/Application Description

- **Model architecture:** `Qwen2.5-Coder-1.5B-Instruct` — a 1.5B-parameter decoder-only transformer optimized for code. Fine-tuned with qLoRA (4-bit NF4 quantization + rank-16 LoRA adapters via `peft` and `trl`).
- **Framework:** PyTorch 2.10–2.11 (cu128), HuggingFace Transformers, PEFT, TRL (SFTTrainer), vLLM.
- **Dataset:** [EffiCoder](https://arxiv.org/abs/2410.10209) — ~9.4k competitive-programming problems with reference solutions and test cases. We curate a clean subset of ~6.6k problems, generate 5 candidate solutions per problem, and select the fastest correct candidate to build a runtime-aware fine-tuning dataset of 2,110 examples.
- **Custom modifications:** Runtime-aware training data selection — instead of taking any passing candidate, we rank by measured median execution time (7 timed subprocess runs) and train on the fastest. A control SFT model is trained on the first passing candidate as an ablation.
- **Hardware target:** NVIDIA A100-SXM4-40GB (Vast.ai / GCP) for generation, training, profiling, and qLoRA training.

---

## 3. Final Results Summary

### Ablation Study (clean test split, 666 problems)

| Metric | Base SLM | Control SFT | Runtime-Aware SFT |
|---|---|---|---|
| Pass@1 | 0.345 | 0.642 | 0.639 |
| Median Exec Time (ms) | 0.0819 | 0.0807 | 0.009 |
| Avg Generation Latency (s) | 0.787 | 0.679 | 0.687 |

### Serving Throughput (1,000 prompts, A100; HF batch_size=8, vLLM batch_size=32, max_new_tokens=256)

| Model | Backend | Throughput (tok/s) | Avg Latency/Prompt (s) | Avg GPU Util |
|---|---|---|---|---|
| Base SLM | HuggingFace | 233.5 | 0.689 | 37.6% |
| Base SLM | vLLM | 1,781.6 | 0.032 | 97.2% |
| Control SFT | HuggingFace | 221.9 | 0.696 | 36.7% |
| Control SFT | vLLM | 1,557.5 | 0.039 | 78.0% |
| Runtime-Aware SFT | HuggingFace | 228.6 | 0.670 | 37.3% |
| Runtime-Aware SFT | vLLM | 1,891.1 | 0.032 | 96.8% |

**Hardware:** 1× NVIDIA A100-SXM4-40GB, CUDA 12.x, PyTorch 2.11, Ubuntu 22.04

**Headline result:** Fine-tuning on runtime-selected candidates (runtime-aware SFT) achieves near-identical correctness to control SFT (Pass@1 0.639 vs 0.642) while generating slightly faster code, and switching from HuggingFace to vLLM serving delivers an **8.3× throughput gain** (228.6 → 1,891.1 tok/s) by eliminating the memory-bandwidth bottleneck in the decode loop via PagedAttention and continuous batching.

---

## 4. Repository Structure

```
efficient-codegen/
├── README.md
├── requirements.txt
├── deliverables/
│   ├── HPML_Final_Report.pdf
│   └── HPML_Final_Presentation.pptx
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
│   ├── run_pipeline_full.sh        # End-to-end data pipeline: full ~6.6k
│   ├── run_pipeline_1k.sh          # 1k subset pipeline
│   └── run_pipeline_20.sh          # 20-sample profiling set
├── generation/
│   └── generate_candidates.py      # Generate N candidate solutions per problem via LLM
├── execution/
│   ├── benchmark.py                # Benchmark reference solutions (subprocess-isolated)
│   ├── evaluate_candidates.py      # Check correctness of generated candidates
│   ├── filter_passing_candidates.py
│   └── benchmark_candidates.py     # Time passing candidates (7 runs, median)
├── training/
│   ├── train.py                    # qLoRA fine-tuning with SFTTrainer
│   ├── evaluate_model.py           # Generate + evaluate candidates for a given model
│   └── select_training_data.py     # Build runtime_aware.jsonl and control.jsonl
├── serving/
│   ├── merge_checkpoint.py         # Merge LoRA adapter into base model weights
│   └── benchmark_serving.py        # Benchmark HuggingFace and vLLM serving backends
├── profiling/
│   ├── profile_model.py            # PyTorch Profiler + W&B logging (high-level metrics)
│   ├── profile_operators.py        # Operator-level profiling, bottleneck report (entry point)
│   ├── generation.py               # Prompt helpers and annotated autoregressive generation loop
│   └── roofline.py                 # Analytical roofline model: memory estimation, plot, report
├── outputs/
│   ├── generated_candidates_full.jsonl
│   ├── evaluated_candidates_full.jsonl
│   ├── benchmarked_candidates_full.jsonl
│   ├── serving_full_results.csv
│   ├── wandb_ablation_runs_2026-04-20.csv
│   ├── wandb_profiled_runs_2026-04-20.csv
│   └── operator_profiling/
│       ├── base/
│       │   ├── roofline_base.png         # Log-log roofline chart — base model
│       │   └── roofline_report_base.txt  # Arithmetic intensity + MFU summary — base model
│       ├── control/
│       │   ├── roofline_control.png
│       │   └── roofline_report_control.txt
│       └── runtime_aware/
│           ├── roofline_runtime_aware.png
│           └── roofline_report_runtime_aware.txt
└── notebooks/
    ├── base_data_generation_evaluation_pipeline.ipynb
    ├── base_model_gpu_profiling.ipynb
    ├── gpu_profiling_comparison.ipynb   # 3-model profiling + roofline analysis
    ├── results.ipynb
    ├── efficient_codegen_experiments.ipynb
    └── efficient_codegen_system_optimization.ipynb
```

---

## 5. Reproducibility Instructions

### A. Environment Setup

```bash
git clone https://github.com/jasmineztruong8/efficient-codegen.git
cd efficient-codegen

python -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**System requirements:** Python 3.10+, CUDA 12.x, ≥ 24 GB GPU memory for generation and serving. Training (qLoRA) runs on a T4 (16 GB) or larger. See `requirements.txt` for pinned package versions.

### B. Experiment Tracking Dashboard

> **🔗 Dashboard:** [https://wandb.ai/efficient-codegen/hpml-efficient-codegen](https://wandb.ai/efficient-codegen/hpml-efficient-codegen)
>
> *Platform:* Weights & Biases

The dashboard includes training curves, ablation evaluation metrics, GPU profiling summaries (latency, memory, throughput), and serving benchmark results for all three model variants.

### C. Dataset

Download `efficoder.json` from the [EffiCoder project](https://arxiv.org/abs/2410.10209) and place it at `data/raw/efficoder.json`. Then run the data pipeline:

```bash
bash scripts/run_pipeline_full.sh   # -> data/curated/scale_full/dataset_clean.json (~6.6k)
bash scripts/run_pipeline_1k.sh     # -> data/curated/scale1k/dataset_clean.json (1k subset)
bash scripts/run_pipeline_20.sh     # -> data/curated/prototyping/ (20-sample set)
```

**Pipeline steps:**
1. `select_dataset.py` — filters ~9.4k problems, keeps ~6.9k eligible
2. `benchmark.py` — runs each reference solution 7× in an isolated subprocess, records median runtime
3. `filter_passing.py` — keeps problems whose reference solution passes all tests (~96% pass rate)

### D. Training

Build the fine-tuning datasets from benchmarked candidates:

```bash
python training/select_training_data.py \
    --candidates_path outputs/benchmarked_candidates_full.jsonl \
    --dataset_path data/curated/scale_full/dataset_clean.json \
    --output_dir training/data
```

Train both models (requires ≥ 16 GB GPU):

```bash
# Control SFT (first correct candidate)
python training/train.py \
    --mode control \
    --data_path training/data/control.jsonl \
    --output_dir checkpoints/control_full \
    --use_wandb --wandb_project hpml-efficient-codegen

# Runtime-Aware SFT (fastest correct candidate)
python training/train.py \
    --mode runtime_aware \
    --data_path training/data/runtime_aware.jsonl \
    --output_dir checkpoints/runtime_aware_full \
    --use_wandb --wandb_project hpml-efficient-codegen
```

Checkpoints are LoRA adapters (~800 MB each) stored on Google Drive and are not committed to the repository.

### E. Evaluation

```bash
bash scripts/run_eval.sh
# Runs evaluate_model.py for base, control, and runtime_aware on scale1k
```

Or individually:

```bash
python training/evaluate_model.py \
    --model_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --run_name base_slm \
    --dataset_path data/curated/scale1k/dataset_clean.json
```

### F. Serving Benchmark

Merge LoRA adapter before vLLM serving:

```bash
python serving/merge_checkpoint.py \
    --adapter_path checkpoints/runtime_aware_full \
    --base_model_name Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --output_dir checkpoints/runtime_aware_merged
```

Benchmark HuggingFace vs vLLM:

```bash
python serving/benchmark_serving.py \
    --backend hf \
    --model_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --input_path data/curated/scale1k/dataset_clean.json \
    --limit 1000 --batch_size 8 \
    --output_path outputs/serving/base_hf.json

python serving/benchmark_serving.py \
    --backend vllm \
    --model_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --input_path data/curated/scale1k/dataset_clean.json \
    --limit 1000 \
    --output_path outputs/serving/base_vllm.json
```

### G. Profiling

High-level profiling (latency, memory, throughput) with W&B logging:

```bash
python profiling/profile_model.py \
    --model_name Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --input_path data/curated/scale1k/dataset_clean.json \
    --limit 320 --batch_size 64 \
    --use_wandb --wandb_project hpml-efficient-codegen
```

Operator-level profiling with roofline analysis (outputs `operators.csv`, `bottleneck_report.txt`, `roofline_report.txt`, and `roofline.png`):

```bash
python profiling/profile_operators.py \
    --model_name Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --input_path data/curated/scale1k/dataset_clean.json \
    --limit 5 --batch_size 1 \
    --output_dir outputs/operator_profile \
    --gpu_name "A100 SXM4-40GB" \
    --gpu_peak_tflops 312 \
    --gpu_peak_bandwidth_gbs 1555
```

Full 3-model profiling comparison (base, control, runtime-aware) is automated in `notebooks/gpu_profiling_comparison.ipynb`.

### H. Quickstart: Reproduce the Headline Serving Result

The following reproduces the vLLM vs HuggingFace throughput comparison (≈ 30 minutes on an A100):

```bash
# 1. Set up environment
pip install -r requirements.txt

# 2. Merge the runtime-aware checkpoint
python serving/merge_checkpoint.py \
    --adapter_path checkpoints/runtime_aware_full \
    --base_model_name Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --output_dir checkpoints/runtime_aware_merged

# 3. Benchmark both backends
python serving/benchmark_serving.py --backend hf \
    --model_path checkpoints/runtime_aware_merged \
    --input_path data/curated/scale1k/dataset_clean.json \
    --limit 1000 --batch_size 8

python serving/benchmark_serving.py --backend vllm \
    --model_path checkpoints/runtime_aware_merged \
    --input_path data/curated/scale1k/dataset_clean.json \
    --limit 1000
```

---

## 6. Results and Observations

- **Runtime-aware SFT matches control SFT correctness** (Pass@1 0.639 vs 0.642, Δ < 1pp) with no architecture change — demonstrating that training-data selection alone is sufficient to steer code quality without sacrificing correctness.
- **vLLM delivers 8.3× serving throughput** over HuggingFace (1,891.1 vs 228.6 tok/s on the runtime-aware model) at 96.8% GPU utilization vs 37.3%. Roofline analysis (see `outputs/operator_profiling/`) confirms why: element-wise and attention ops during decode sit at arithmetic intensity 1–10 FLOPs/byte — far below the A100's ridge point (~200 FLOPs/byte) — making the decode phase strongly memory-bandwidth-bound. The effective AI across all ops appears compute-bound only because prefill dominates total FLOPs; at batch_size=1 decode is the serving bottleneck. vLLM eliminates this with PagedAttention and continuous batching, raising effective batch size and arithmetic intensity.
- **Per-token CPU-GPU synchronizations are a measurable bottleneck** — `aten::item` + `cudaStreamSynchronize` account for ~6.7% of total operator time in HuggingFace's `model.generate()` loop due to EOS-token detection; vLLM eliminates this with asynchronous scheduling.
- **qLoRA training is memory-efficient** — 4-bit NF4 quantization + rank-16 LoRA enables fine-tuning a 1.5B model on a single T4 (16 GB) with no degradation in Pass@1 relative to the base model after control SFT.
- **What did not work:** Execution-time improvements from runtime-aware training were marginal (0.0807 ms vs 0.0819 ms baseline) — likely because at the 1.5B scale the model lacks sufficient capacity to consistently learn and apply algorithmic improvements; the dataset also skews toward already-fast reference solutions, leaving little room for further optimization.

---

## 7. Notes

- Trained checkpoints (LoRA adapters, ~800 MB each) are stored on Google Drive and not committed to the repository.
- All secrets (W&B API keys) should be loaded from environment variables. Never commit API keys to the repository.
- The raw EffiCoder dataset is not committed; download it separately and place at `data/raw/efficoder.json`.

<<<<<<< HEAD
### AI Use Disclosure

*Per the HPML AI Use Policy (posted on CourseWorks). Required for every submission.*
=======
### Ablation Study (clean test split, 666 problems)

| Model | Pass@1 | Median Exec Time (s) | Avg Gen Latency (s) |
|-------|--------|----------------------|---------------------|
| Base SLM | 0.345 | 0.0000819 | 0.787 |
| Control SFT | 0.642 | 0.0000807 | 0.679 |
| Runtime-Aware SFT | 0.639 | 0.0000809 | 0.687 |

### Historical Ablation Study (legacy scale1k, 1,000 problems — pre data-split fix)
>>>>>>> d9af511 (reran training and eval)

**Did your team use any AI tool in completing this project?**

- [ ] No, we did not use any AI tool.
- [x] Yes, we used AI assistance as described below.

**Tool(s) used:** Claude, GitHub Copilot, Codex, ChatGPT

**Specific purpose:** [PLACEHOLDER — e.g., debugging CUDA profiling code, clarifying roofline model concepts, drafting README prose]

**Sections affected:** [PLACEHOLDER — e.g., profiling/profile_operators.py roofline functions, README]

**How we verified correctness:** [PLACEHOLDER — e.g., re-ran all reported experiments ourselves; confirmed profiler trace interpretations against raw traces; reviewed and tested all AI-suggested code before committing]

By submitting this project, the team confirms that the analysis, interpretations, and conclusions are our own, and that any AI assistance is fully disclosed above. The same disclosure block appears as an appendix in the final report.

### License

Released under the MIT License. See [`LICENSE`](LICENSE).

### Citation

If you build on this work, please cite:

```bibtex
@misc{truong2026efficientcodegen,
  title  = {Optimization of SLMs for Efficient Code Generation},
  author = {Truong, Jasmine and Zhang, Yingxin and Mahajan, Arnav and Gao, Jianyi},
  year   = {2026},
  note   = {HPML Spring 2026 Final Project, Columbia University},
  url    = {https://github.com/jasmineztruong8/efficient-codegen}
}
```

### Contact

Open a GitHub Issue or email the team via Columbia email.

---

## References

- Huang et al. [EffiCoder](https://arxiv.org/abs/2410.10209), 2025
- Nichols et al. [Performance-Aligned LLMs for Generating Fast Code](https://arxiv.org/abs/2404.18864), 2024
- Hui et al. [Qwen2.5-Coder Technical Report](https://arxiv.org/abs/2409.12186), 2024
- Dettmers et al. [QLoRA](https://arxiv.org/abs/2305.14314), 2023
- Kwon et al. [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180), 2023

---

*HPML Spring 2026 — Dr. Kaoutar El Maghraoui — Columbia University*
