# Prompt Suppression Geometry

Utilities for gradient-guided prompt optimization against language-model
targets. This repository contains an EPO-style optimizer, target runners for
logits, neurons, residual directions, and attention entries, plus small helpers
for attribution and remote experiment execution.

## Repository Contents

- `prompt_suppression.epo`: evolutionary prompt optimization and Pareto-frontier utilities.
- `prompt_suppression.runners`: target builders for logits, MLP neurons, residual directions,
  and attention entries.
- `prompt_suppression.benchmarks`: EPO, GCG, random, and natural text scan baselines.
- `prompt_suppression.results`: candidate records, operating point summaries, and CSV IO.
- `prompt_suppression.robustness`: deterministic prompt variant checks.
- `prompt_suppression.plotting`: standard scatter, Pareto, bar, and robustness figures.
- `prompt_suppression.behavior`: continuation log probability scoring for behavioral tests.
- `prompt_suppression.activations`: helpers for fitting residual-stream directions.
- `prompt_suppression.attribution`: token-resampling utilities for local attribution views.
- `prompt_suppression.experiment`: optional Modal/S3 experiment orchestration helpers.

## Basic Usage

Install dependencies with `uv`:

```bash
uv sync
```

```python
from prompt_suppression.epo import epo, load_model
from prompt_suppression.runners import logit_diff_runner

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

For Gemma 2 2B:

```python
model, tokenizer = load_model(
    model_name="google/gemma-2-2b",
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
)
```

## Reproducible experiment workflow

Run a suppression experiment from a JSON target spec:

```bash
uv run prompt-suppression run \
  --spec examples/logit_spec.json \
  --texts examples/text_pool.txt \
  --out runs/example \
  --methods epo random random_search minscan gcg \
  --seeds 0 1 2
```

Run the same workflow on Gemma 2 2B:

```bash
uv run prompt-suppression run \
  --spec examples/gemma_logit_spec.json \
  --texts examples/text_pool.txt \
  --out runs/gemma_example \
  --methods epo random random_search minscan gcg \
  --seeds 0 1 2 \
  --torch-dtype bfloat16
```

Gemma models may require a Hugging Face token with Google model access enabled.
If loading fails with an authorization error, accept the model terms on Hugging
Face and run `huggingface-cli login`.

The command writes:

- `runs/example/candidates.csv`: every scored prompt from search and baselines.
- `runs/example/summary.csv`: method level operating point summaries.

Generate standard figures:

```bash
uv run prompt-suppression plot \
  --records runs/example/candidates.csv \
  --out-dir runs/example/figures
```

Evaluate deterministic robustness variants for the best prompts:

```bash
uv run prompt-suppression robustness \
  --spec examples/logit_spec.json \
  --records runs/example/candidates.csv \
  --out runs/example/robustness.csv \
  --rows-out runs/example/robustness_rows.csv \
  --summary-out runs/example/robustness_summary.csv
```

Score continuation preferences for behavioral checks:

```bash
uv run prompt-suppression behavior \
  --evals examples/behavior_evals.json \
  --out runs/example/behavior.csv
```

Generate target specs:

```bash
uv run prompt-suppression generate-targets \
  --out runs/specs/logits_and_neurons.json \
  --tokens " dog" " answer" \
  --layers 8-10 \
  --neurons 0,32,64 \
  --model-size 70m
```

Fit residual directions across layers and write a residual target spec:

```bash
uv run prompt-suppression fit-directions \
  --contrast examples/contrast_pairs.json \
  --layers 0-6 \
  --out-dir runs/directions \
  --name eval_awareness \
  --spec-out runs/specs/eval_awareness_residuals.json
```

Export a paper table from a summary CSV:

```bash
uv run prompt-suppression latex-table \
  --csv runs/example/summary.csv \
  --out runs/example/summary_table.tex \
  --columns target_name,method,best_target,best_target_xentropy
```

Write starter behavioral evaluation templates:

```bash
uv run prompt-suppression behavior-templates --out runs/specs/behavior_evals.json
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
