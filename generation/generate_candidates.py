"""
generate_candidates.py

Generate multiple code candidates for each example using a Hugging Face
causal language model, and save the outputs as JSONL.

Each output record contains:
- dataset_index, task, instruction, input, tests, reference_output, function_name
- candidates: list of generated code strings (length = num_candidates)

Pipeline: problem prompt -> multiple code candidates -> correctness/runtime evaluation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# Keep this prompt identical to the one used in training/select_training_data.py
# and training/train.py so that inference format matches training format.
SYSTEM_PROMPT = (
    "Write a correct Python solution optimized for fast execution time. "
    "Use efficient algorithms and data structures to minimize runtime. "
    "Return only the code with no explanation."
)


def build_messages(instruction: str, input_content: str) -> list:
    content = instruction
    if input_content:
        content += f"\n\nInput:\n{input_content}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--num_candidates", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8, help="Number of problems per batch")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading examples from: {args.input_path}")
    with open(args.input_path, "r", encoding="utf-8") as f:
        problems = json.load(f)
    if args.limit:
        problems = problems[:args.limit]
    print(f"Loaded {len(problems)} examples")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading model: {args.model_name}  device={device}  dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        device_map=device,
    )
    model.eval()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as outfile, torch.inference_mode():
        for batch_start in tqdm(range(0, len(problems), args.batch_size), desc="Generating"):
            batch = problems[batch_start: batch_start + args.batch_size]
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

            # Generate num_candidates sequences per prompt in one forward pass
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.num_candidates,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # output_ids shape: (batch_size * num_candidates, seq_len)
            input_lengths = inputs["input_ids"].shape[1]
            decoded = tokenizer.batch_decode(
                output_ids[:, input_lengths:],
                skip_special_tokens=True,
            )

            if torch.cuda.is_available() and batch_start % (args.batch_size * 10) == 0:
                mem_used = torch.cuda.memory_allocated() / 1e9
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"GPU memory: {mem_used:.1f}/{mem_total:.1f} GB")

            # Group candidates back per problem
            for i, problem in enumerate(batch):
                candidates = decoded[i * args.num_candidates: (i + 1) * args.num_candidates]
                result = {
                    "dataset_index": problem["dataset_index"],
                    "task": problem.get("task"),
                    "instruction": problem["instruction"],
                    "input": problem.get("input", ""),
                    "tests": problem["tests"],
                    "reference_output": problem.get("reference_output") or problem.get("output", ""),
                    "function_name": problem.get("function_name"),
                    "candidates": candidates,
                }
                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Generated candidates saved to {output_path}")


if __name__ == "__main__":
    main()
