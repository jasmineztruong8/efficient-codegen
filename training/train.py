"""
train.py

Fine-tune Qwen2.5-Coder-1.5B-Instruct with qLoRA using trl.SFTTrainer.

Modes:
  runtime_aware — train on fastest-correct candidates (main experiment)
  control       — train on first-correct candidates (ablation baseline)

For profiling: pass --profile to capture a PyTorch Profiler trace during
training. The ProfilingCallback steps the profiler after each training step,
matching the pattern in profiling/profile_model.py. Traces are saved to
--trace_dir and viewable in TensorBoard.

Usage:
  python training/train.py \
    --mode runtime_aware \
    --data_path training/data/runtime_aware.jsonl \
    --output_dir training/checkpoints/runtime_aware \
    --use_wandb

  # With profiling:
  python training/train.py \
    --mode runtime_aware \
    --data_path training/data/runtime_aware.jsonl \
    --output_dir training/checkpoints/runtime_aware \
    --profile \
    --trace_dir outputs/tb_profiler/train
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

# Must match SYSTEM_PROMPT in generation/generate_candidates.py and
# training/select_training_data.py so training format == inference format.
SYSTEM_PROMPT = (
    "Write a correct Python solution optimized for fast execution time. "
    "Use efficient algorithms and data structures to minimize runtime. "
    "Return only the code with no explanation."
)

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# LoRA target modules for Qwen2.5
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# Profiling support
# ---------------------------------------------------------------------------

class ProfilingCallback(TrainerCallback):
    """
    Steps the PyTorch Profiler after each training step.

    To add profiling: pass --profile. The profiler context is managed in
    main() and this callback advances it per step, matching the manual
    prof.step() pattern used in profiling/profile_model.py.

    To extend: you can log per-step GPU stats here by adding to on_step_end.
    """

    def __init__(self, profiler):
        self.profiler = profiler

    def on_step_end(self, args, state, control, **kwargs):
        self.profiler.step()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(data_path: str, tokenizer, max_seq_length: int, limit: int | None) -> Dataset:
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    if limit:
        records = records[:limit]

    # Apply the model's chat template to format each (instruction, code) pair.
    # add_generation_prompt=False includes the assistant turn in the text,
    # which is what SFTTrainer trains on.
    texts = [
        tokenizer.apply_chat_template(
            r["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        for r in records
    ]
    return Dataset.from_dict({"text": texts})


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="qLoRA fine-tuning via SFTTrainer.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["runtime_aware", "control"],
        required=True,
        help="runtime_aware: train on fastest-correct candidates. control: train on first-correct (ablation).",
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to training JSONL")
    parser.add_argument("--output_dir", type=str, required=True, help="Checkpoint save directory")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    # Training hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    # LoRA hyperparameters
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    # WandB
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="hpml-efficient-codegen")
    # Profiling — see ProfilingCallback above
    parser.add_argument("--profile", action="store_true", help="Enable PyTorch Profiler during training")
    parser.add_argument("--trace_dir", type=str, default="outputs/tb_profiler/train",
                        help="Directory for TensorBoard profiler traces (used with --profile)")
    # Debug
    parser.add_argument("--limit", type=int, default=None, help="Limit training examples (debug)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.use_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    # 4-bit quantization for qLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # SFT requires right padding

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # required with gradient checkpointing

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_dataset = load_dataset(args.data_path, tokenizer, args.max_seq_length, args.limit)
    print(f"Training on {len(train_dataset)} examples  (mode={args.mode})")


    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="wandb" if args.use_wandb else "none",
        run_name=f"efficient-codegen-{args.mode}" if args.use_wandb else None,
        dataset_text_field="text",
        max_length=args.max_seq_length,
    )

    # --- Profiling setup ---
    # ProfilingCallback steps the profiler each training step.
    # Traces land in --trace_dir; view with TensorBoard or upload to WandB.
    # To extend: add per-step GPU memory logging inside ProfilingCallback.on_step_end.
    callbacks = []
    profiler_ctx = None

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
            on_trace_ready=tensorboard_trace_handler(args.trace_dir, worker_name="train"),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        profiler_ctx.__enter__()
        callbacks.append(ProfilingCallback(profiler_ctx))
        print(f"Profiling enabled — traces will be saved to {args.trace_dir}")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        peft_config=lora_config,
        args=training_args,
        callbacks=callbacks if callbacks else None,
    )

    trainer.train()

    if profiler_ctx is not None:
        profiler_ctx.__exit__(None, None, None)
        print(f"Profiler traces saved to {args.trace_dir}")

    if args.use_wandb:
        import wandb
        log = {"num_train_examples": len(train_dataset)}
        if torch.cuda.is_available():
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            log["peak_cuda_memory_mb"] = peak_mem_mb
            print(f"Peak CUDA memory: {peak_mem_mb:.1f} MB")
        wandb.log(log)

    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
