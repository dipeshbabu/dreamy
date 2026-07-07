"""Behavioral evaluation helpers for prompt suppression experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import torch


def continuation_logprob(model, tokenizer, prompt: str, continuation: str) -> dict:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt + continuation, add_special_tokens=False)
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("Continuation produced no additional tokens")
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=model.device)
    with torch.no_grad():
        logits = model(input_ids).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)

    cont_ids = full_ids[len(prompt_ids) :]
    start = len(prompt_ids) - 1
    token_scores = []
    for offset, token_id in enumerate(cont_ids):
        token_scores.append(float(log_probs[start + offset, token_id].detach().cpu()))
    return {
        "logprob": float(sum(token_scores)),
        "avg_logprob": float(sum(token_scores) / len(token_scores)),
        "n_tokens": len(token_scores),
    }


def score_continuations(model, tokenizer, evals: Sequence[dict]) -> list[dict]:
    rows = []
    for i, item in enumerate(evals):
        prompt = item["prompt"]
        continuations = item["continuations"]
        for name, continuation in continuations.items():
            score = continuation_logprob(model, tokenizer, prompt, continuation)
            rows.append(
                {
                    "prompt_id": item.get("id", i),
                    "continuation": name,
                    "text": continuation,
                    **score,
                }
            )
    return rows


def load_behavior_evals(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data["evals"]
    return data


def write_behavior_rows(rows: Sequence[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
