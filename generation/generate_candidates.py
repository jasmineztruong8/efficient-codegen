"""
generate_candidates.py

Short description:
Generate multiple code candidates for each example in generation_input_20.json
using a Hugging Face causal language model, and save the outputs as JSONL.

Each output record contains:
- dataset_index
- task
- model name
- sampling settings
- candidate_id
- prompt
- generated_code

This script is meant for the candidate-generation stage of the pipeline:
problem prompt -> multiple code candidates -> later correctness/runtime evaluation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

def generate_one(
    model,
    tokenizer,
    instruction,
    input_content,
    generation_config,
    max_new_tokens=256,
):
    if input_content:
        prompt = f"Instruction: {instruction}\nInput: {input_content}\nOutput:"
    else:
        prompt = f"Instruction: {instruction}\nOutput:"

    inputs = tokenizer(prompt, return_tensors="pt")

    # Move all tensors in the inputs dictionary to the model's device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs, # Pass the dictionary of inputs, which now includes input_ids on GPU
        generation_config=generation_config,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
    )
    # Decode the generated output, skipping the input prompt
    # Use len(inputs["input_ids"][0]) as input_ids is now part of the moved inputs dict
    generated_text = tokenizer.decode(
        output_ids[0][len(inputs["input_ids"][0]) :],
        skip_special_tokens=True,
    )
    return generated_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input JSON file containing problems.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output JSONL file to save generated candidates.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="Hugging Face model name or path.",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=1,
        help="Number of candidate solutions to generate per problem.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of problems to process for testing.",
    )
    args = parser.parse_args()

    print(f"Loading examples from: {args.input_path}")
    with open(args.input_path, "r", encoding="utf-8") as f:
        problems = json.load(f)
    
    if args.limit:
        problems = problems[:args.limit]
    print(f"Loaded {len(problems)} examples")

    print(f"Loading model: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using device={{device}}, dtype={{dtype}}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()

    # Set up generation config
    generation_config = GenerationConfig(
        temperature=0.2,
        top_p=0.95,
        do_sample=True,
        num_return_sequences=1,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as outfile:
        for problem in tqdm(problems, desc="Generating candidates"):
            instruction = problem["instruction"]
            input_content = problem.get("input", "") # Use .get to handle missing 'input' key

            candidates = []
            for _ in range(args.num_candidates):
                generated_code = generate_one(
                    model,
                    tokenizer,
                    instruction,
                    input_content,
                    generation_config,
                    max_new_tokens=args.max_new_tokens,
                )
                candidates.append(generated_code)

            result = {
                "dataset_index": problem["dataset_index"],
                "task": problem["task"],
                "instruction": problem["instruction"],
                "input": problem.get("input", ""),
                "tests": problem["tests"],
                "reference_output": problem["reference_output"],
                "function_name": problem.get("function_name"),
                "candidates": candidates,
            }
            outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Generated candidates saved to {output_path}")


if __name__ == "__main__":
    main()
