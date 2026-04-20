"""Merge a PEFT adapter into a standalone model checkpoint."""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Merge a PEFT adapter into a standalone checkpoint.")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to PEFT adapter checkpoint")
    parser.add_argument("--base_model_name", type=str, required=True, help="Base HF model name")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save merged model")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16 if torch.cuda.is_available()
        else torch.float32
    )

    print(f"Loading base model: {args.base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    print(f"Loading adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(base, args.adapter_path)

    print("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
