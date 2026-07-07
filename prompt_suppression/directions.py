"""Residual direction fitting and layer sweep utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from prompt_suppression.activations.probes import collect_residual_states


def load_contrast_pairs(path: str | Path) -> tuple[list[str], list[str]]:
    path = Path(path)
    a_texts = []
    b_texts = []
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows = data["pairs"] if isinstance(data, dict) else data

    for row in rows:
        if "a" in row and "b" in row:
            a_texts.append(row["a"])
            b_texts.append(row["b"])
        else:
            a_texts.append(row["positive"])
            b_texts.append(row["negative"])
    if len(a_texts) != len(b_texts) or not a_texts:
        raise ValueError("Contrast file must contain nonempty paired texts")
    return a_texts, b_texts


def mean_difference_direction(
    a_states: np.ndarray,
    b_states: np.ndarray,
) -> np.ndarray:
    direction = a_states.mean(axis=0) - b_states.mean(axis=0)
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("Contrast direction has zero norm")
    return direction / norm


def projection_gap(
    a_states: np.ndarray,
    b_states: np.ndarray,
    direction: np.ndarray,
) -> float:
    return float((a_states @ direction).mean() - (b_states @ direction).mean())


def fit_direction_for_layer(
    model,
    tokenizer,
    layer: int,
    a_texts: Sequence[str],
    b_texts: Sequence[str],
    *,
    pooling: str = "last",
    max_len: int = 256,
) -> tuple[np.ndarray, dict]:
    a_states = collect_residual_states(
        model, layer, tokenizer, a_texts, max_len=max_len, pooling=pooling
    )
    b_states = collect_residual_states(
        model, layer, tokenizer, b_texts, max_len=max_len, pooling=pooling
    )
    direction = mean_difference_direction(a_states, b_states)
    gap = projection_gap(a_states, b_states, direction)
    row = {
        "layer": int(layer),
        "projection_gap": gap,
        "a_mean_projection": float((a_states @ direction).mean()),
        "b_mean_projection": float((b_states @ direction).mean()),
        "n_pairs": len(a_texts),
        "pooling": pooling,
    }
    return direction, row


def fit_direction_sweep(
    model,
    tokenizer,
    contrast_path: str | Path,
    layers: Sequence[int],
    out_dir: str | Path,
    *,
    name: str,
    pooling: str = "last",
    max_len: int = 256,
) -> list[dict]:
    a_texts, b_texts = load_contrast_pairs(contrast_path)
    out_dir = Path(out_dir)
    vector_dir = out_dir / "vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for layer in layers:
        direction, row = fit_direction_for_layer(
            model,
            tokenizer,
            int(layer),
            a_texts,
            b_texts,
            pooling=pooling,
            max_len=max_len,
        )
        vector_path = vector_dir / f"{name}_L{layer}.npy"
        np.save(vector_path, direction.astype(np.float32))
        row["name"] = name
        row["vector_path"] = str(vector_path)
        rows.append(row)
    write_direction_rows(rows, out_dir / f"{name}_layer_sweep.csv")
    return rows


def write_direction_rows(rows: Sequence[dict], path: str | Path) -> None:
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


def top_direction_specs(rows: Sequence[dict], *, top_k: int = 3) -> list[dict]:
    ranked = sorted(rows, key=lambda r: abs(float(r["projection_gap"])), reverse=True)
    specs = []
    for row in ranked[:top_k]:
        layer = int(row["layer"])
        specs.append(
            {
                "name": f"{row['name']}_residual_L{layer}",
                "type": "residual",
                "layer": layer,
                "vector_path": row["vector_path"],
                "minimize": True,
            }
        )
    return specs


def save_torch_vector(vector: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(vector, dtype=torch.float32), path)
