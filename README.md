# Dreamy

Utilities for gradient-guided prompt optimization against language-model
targets. This repository contains an EPO-style optimizer, target runners for
logits, neurons, residual directions, and attention entries, plus small helpers
for attribution and remote experiment execution.

## Repository Contents

- `dreamy.epo`: evolutionary prompt optimization and Pareto-frontier utilities.
- `dreamy.runners`: target builders for logits, MLP neurons, residual directions,
  and attention entries.
- `dreamy.benchmarks`: EPO, GCG, random, and natural text scan baselines.
- `dreamy.results`: candidate records, operating point summaries, and CSV IO.
- `dreamy.robustness`: deterministic prompt variant checks.
- `dreamy.plotting`: standard scatter, Pareto, bar, and robustness figures.
- `dreamy.behavior`: continuation log probability scoring for behavioral tests.
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

## Reproducible experiment workflow

Run a suppression experiment from a JSON target spec:

```bash
uv run dreamy run \
  --spec examples/logit_spec.json \
  --texts examples/text_pool.txt \
  --out runs/example \
  --methods epo random random_search minscan gcg \
  --seeds 0 1 2
```

The command writes:

- `runs/example/candidates.csv`: every scored prompt from search and baselines.
- `runs/example/summary.csv`: method level operating point summaries.

Generate standard figures:

```bash
uv run dreamy plot \
  --records runs/example/candidates.csv \
  --out-dir runs/example/figures
```

Evaluate deterministic robustness variants for the best prompts:

```bash
uv run dreamy robustness \
  --spec examples/logit_spec.json \
  --records runs/example/candidates.csv \
  --out runs/example/robustness.csv \
  --rows-out runs/example/robustness_rows.csv \
  --summary-out runs/example/robustness_summary.csv
```

Score continuation preferences for behavioral checks:

```bash
uv run dreamy behavior \
  --evals examples/behavior_evals.json \
  --out runs/example/behavior.csv
```

Generate target specs:

```bash
uv run dreamy generate-targets \
  --out runs/specs/logits_and_neurons.json \
  --tokens " dog" " answer" \
  --layers 8-10 \
  --neurons 0,32,64 \
  --model-size 70m
```

Fit residual directions across layers and write a residual target spec:

```bash
uv run dreamy fit-directions \
  --contrast examples/contrast_pairs.json \
  --layers 0-6 \
  --out-dir runs/directions \
  --name eval_awareness \
  --spec-out runs/specs/eval_awareness_residuals.json
```

Export a paper table from a summary CSV:

```bash
uv run dreamy latex-table \
  --csv runs/example/summary.csv \
  --out runs/example/summary_table.tex \
  --columns target_name,method,best_target,best_target_xentropy
```

Write starter behavioral evaluation templates:

```bash
uv run dreamy behavior-templates --out runs/specs/behavior_evals.json
```

For CPU only smoke runs, add:

```bash
--device-map cpu --torch-dtype float32
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
