"""generation.py — prompt helpers and annotated autoregressive generation loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.profiler import record_function


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
    Autoregressive generation with record_function regions:
      - 'prefill'          : first forward pass (prompt encoding, KV-cache fill)
      - 'decode_step_N'    : Nth token (first decode_steps_to_trace steps)
      - 'decode_remaining' : all remaining steps
    """
    past_key_values = None
    generated = input_ids.clone()
    cur_attn = attention_mask.clone()

    with record_function("prefill"):
        with torch.no_grad():
            out = model(input_ids=generated, attention_mask=cur_attn, use_cache=True)
        past_key_values = out.past_key_values
        logits = out.logits[:, -1, :]

    next_token = _sample(logits, temperature, top_p)
    generated = torch.cat([generated, next_token], dim=-1)
    cur_attn = torch.cat([cur_attn, torch.ones_like(next_token)], dim=-1)
    finished = (next_token.squeeze(-1) == tokenizer.eos_token_id)

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

        next_token = _sample(logits, temperature, top_p)
        next_token[finished] = tokenizer.pad_token_id
        finished |= (next_token.squeeze(-1) == tokenizer.eos_token_id)
        generated = torch.cat([generated, next_token], dim=-1)
        cur_attn = torch.cat([cur_attn, torch.ones_like(next_token)], dim=-1)

    return generated


def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Minimal top-p nucleus sampling — avoids HF overhead inside the profiler."""
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    sorted_probs[(cumsum - sorted_probs) > top_p] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    return sorted_idx.gather(-1, torch.multinomial(sorted_probs, num_samples=1))
