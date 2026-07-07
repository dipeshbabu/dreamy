# Dreamy

Utilities for gradient-guided prompt optimization against language-model
targets. This repository contains an EPO-style optimizer, target runners for
logits, neurons, residual directions, and attention entries, plus small helpers
for attribution and remote experiment execution.

## Repository Contents

- `dreamy.epo`: evolutionary prompt optimization and Pareto-frontier utilities.
- `dreamy.runners`: target builders for logits, MLP neurons, residual directions,
  and attention entries.
- `dreamy.activations`: helpers for fitting residual-stream directions.
- `dreamy.attribution`: token-resampling utilities for local attribution views.
- `dreamy.experiment`: optional Modal/S3 experiment orchestration helpers.

## Basic Usage

Install dependencies with `uv`:

```bash
uv sync
```

```python
from dreamy.epo import epo, load_model
from dreamy.runners import logit_diff_runner

model, tokenizer = load_model(model_size="70m")
token_id = tokenizer.encode(" dog", add_special_tokens=False)[0]
runner = logit_diff_runner(model, tokenizer, token_id, banned_text="dog")
runner.minimize = True

history = epo(
    runner,
    model,
    tokenizer,
    seq_len=12,
    population_size=8,
    iters=50,
)
```

Pass `model_name` to `load_model` to use another compatible Hugging Face causal
LM:

```python
model, tokenizer = load_model(model_name="microsoft/phi-2")
```

## Notes

The optimizer assumes white-box access to model activations and input-token
gradients. Keep manuscript drafts, generated PDFs, posters, and local result
artifacts outside git; the ignore rules are set up for that workflow.

## Verification

Run the lightweight regression tests with:

```bash
uv run python -m unittest discover -s tests -v
```
