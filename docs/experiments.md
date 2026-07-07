# Full experiment runbook

This is the actual experiment workflow for the paper. The small commands in the
README are minimal examples; use this file for full runs.

The `examples/` files are smoke test fixtures only. For paper results, build a
local frontier prompt bundle under `data/frontier/` and use those paths
throughout the runbook.

The two model settings are:

- Phi 2: `microsoft/phi-2`
- Gemma 2 2B: `google/gemma-2-2b`

Gemma may require Hugging Face access approval. If loading fails with an
authorization error, accept the model terms on Hugging Face and run
`huggingface-cli login`.

## 0. Build frontier prompt data

Install the optional dataset dependencies:

```bash
uv sync --extra remote
```

Build a local benchmark bundle:

```bash
uv run prompt-suppression build-frontier-data \
  --out-dir data/frontier \
  --sources mmlu_pro math500 gpqa_diamond hle \
  --allow-gated \
  --max-items-per-source 500 \
  --train-fraction 0.75 \
  --behavior-limit 300
```

`gpqa_diamond` and `hle` are gated Hugging Face datasets. Accept their terms and
run `huggingface-cli login` before using `--allow-gated`. Do not commit or
redistribute generated benchmark rows; `data/` is ignored by git.

The command writes:

- `data/frontier/text_pools/frontier_train.txt`
- `data/frontier/text_pools/frontier_eval.txt`
- `data/frontier/contrasts/eval_awareness_train.json`
- `data/frontier/contrasts/eval_awareness_eval.json`
- `data/frontier/contrasts/casing_control.json`
- `data/frontier/contrasts/style_control.json`
- `data/frontier/behavior/frontier_behavior.json`

Use `frontier_train.txt` and `eval_awareness_train.json` for fitting directions.
Use `frontier_eval.txt` for suppression baselines and held out reporting.

## 1. Generate target specs

Phi 2 logit and neuron targets:

```bash
uv run prompt-suppression generate-targets \
  --out runs/specs/phi2_logits_neurons.json \
  --model-name microsoft/phi-2 \
  --texts-path data/frontier/text_pools/frontier_eval.txt \
  --tokens " dog" " answer" " safe" " grade" " test" \
  --layers 8,12,16,20,24,28 \
  --neurons 0,64,128,256,512
```

Gemma 2 2B logit and neuron targets:

```bash
uv run prompt-suppression generate-targets \
  --out runs/specs/gemma2_logits_neurons.json \
  --model-name google/gemma-2-2b \
  --attn-implementation eager \
  --texts-path data/frontier/text_pools/frontier_eval.txt \
  --tokens " dog" " answer" " safe" " grade" " test" \
  --layers 6,10,14,18,22 \
  --neurons 0,64,128,256,512
```

## 2. Fit residual directions

Phi 2 evaluation awareness direction sweep:

```bash
uv run prompt-suppression fit-directions \
  --model-name microsoft/phi-2 \
  --contrast data/frontier/contrasts/eval_awareness_train.json \
  --layers 0-31 \
  --out-dir runs/directions/phi2_eval_awareness \
  --name phi2_eval_awareness \
  --spec-out runs/specs/phi2_eval_awareness_residuals.json \
  --top-k 5 \
  --texts-path data/frontier/text_pools/frontier_eval.txt
```

Phi 2 casing direction sweep:

```bash
uv run prompt-suppression fit-directions \
  --model-name microsoft/phi-2 \
  --contrast data/frontier/contrasts/casing_control.json \
  --layers 0-31 \
  --out-dir runs/directions/phi2_casing \
  --name phi2_casing \
  --spec-out runs/specs/phi2_casing_residuals.json \
  --top-k 5 \
  --texts-path data/frontier/text_pools/frontier_eval.txt
```

Gemma 2 2B evaluation awareness direction sweep:

```bash
uv run prompt-suppression fit-directions \
  --model-name google/gemma-2-2b \
  --attn-implementation eager \
  --torch-dtype bfloat16 \
  --contrast data/frontier/contrasts/eval_awareness_train.json \
  --layers 0-25 \
  --out-dir runs/directions/gemma2_eval_awareness \
  --name gemma2_eval_awareness \
  --spec-out runs/specs/gemma2_eval_awareness_residuals.json \
  --top-k 5 \
  --texts-path data/frontier/text_pools/frontier_eval.txt
```

Gemma 2 2B casing direction sweep:

```bash
uv run prompt-suppression fit-directions \
  --model-name google/gemma-2-2b \
  --attn-implementation eager \
  --torch-dtype bfloat16 \
  --contrast data/frontier/contrasts/casing_control.json \
  --layers 0-25 \
  --out-dir runs/directions/gemma2_casing \
  --name gemma2_casing \
  --spec-out runs/specs/gemma2_casing_residuals.json \
  --top-k 5 \
  --texts-path data/frontier/text_pools/frontier_eval.txt
```

## 3. Run suppression experiments

Use the same seeds and method set for all target specs.

Phi 2 logits and neurons:

```bash
uv run prompt-suppression run \
  --spec runs/specs/phi2_logits_neurons.json \
  --texts data/frontier/text_pools/frontier_eval.txt \
  --out runs/phi2/logits_neurons \
  --methods epo gcg random random_search minscan \
  --seeds 0 1 2 3 4 \
  --seq-len 32 \
  --population-size 24 \
  --iters 150 \
  --explore-per-pop 16 \
  --batch-size 64 \
  --topk 256
```

Phi 2 evaluation awareness residual targets:

```bash
uv run prompt-suppression run \
  --spec runs/specs/phi2_eval_awareness_residuals.json \
  --texts data/frontier/text_pools/frontier_eval.txt \
  --out runs/phi2/eval_awareness_residuals \
  --methods epo gcg random random_search minscan \
  --seeds 0 1 2 3 4 \
  --seq-len 32 \
  --population-size 24 \
  --iters 150 \
  --explore-per-pop 16 \
  --batch-size 64 \
  --topk 256
```

Phi 2 casing residual targets:

```bash
uv run prompt-suppression run \
  --spec runs/specs/phi2_casing_residuals.json \
  --texts data/frontier/text_pools/frontier_eval.txt \
  --out runs/phi2/casing_residuals \
  --methods epo gcg random random_search minscan \
  --seeds 0 1 2 3 4 \
  --seq-len 32 \
  --population-size 24 \
  --iters 150 \
  --explore-per-pop 16 \
  --batch-size 64 \
  --topk 256
```

Gemma 2 2B logits and neurons:

```bash
uv run prompt-suppression run \
  --spec runs/specs/gemma2_logits_neurons.json \
  --texts data/frontier/text_pools/frontier_eval.txt \
  --out runs/gemma2/logits_neurons \
  --methods epo gcg random random_search minscan \
  --seeds 0 1 2 3 4 \
  --attn-implementation eager \
  --torch-dtype bfloat16 \
  --seq-len 32 \
  --population-size 24 \
  --iters 150 \
  --explore-per-pop 16 \
  --batch-size 64 \
  --topk 256
```

Gemma 2 2B evaluation awareness residual targets:

```bash
uv run prompt-suppression run \
  --spec runs/specs/gemma2_eval_awareness_residuals.json \
  --texts data/frontier/text_pools/frontier_eval.txt \
  --out runs/gemma2/eval_awareness_residuals \
  --methods epo gcg random random_search minscan \
  --seeds 0 1 2 3 4 \
  --attn-implementation eager \
  --torch-dtype bfloat16 \
  --seq-len 32 \
  --population-size 24 \
  --iters 150 \
  --explore-per-pop 16 \
  --batch-size 64 \
  --topk 256
```

Gemma 2 2B casing residual targets:

```bash
uv run prompt-suppression run \
  --spec runs/specs/gemma2_casing_residuals.json \
  --texts data/frontier/text_pools/frontier_eval.txt \
  --out runs/gemma2/casing_residuals \
  --methods epo gcg random random_search minscan \
  --seeds 0 1 2 3 4 \
  --attn-implementation eager \
  --torch-dtype bfloat16 \
  --seq-len 32 \
  --population-size 24 \
  --iters 150 \
  --explore-per-pop 16 \
  --batch-size 64 \
  --topk 256
```

## 4. Generate plots

Run this for each experiment output directory:

```bash
uv run prompt-suppression plot \
  --records runs/phi2/logits_neurons/candidates.csv \
  --out-dir runs/phi2/logits_neurons/figures
```

Repeat for:

- `runs/phi2/eval_awareness_residuals/candidates.csv`
- `runs/phi2/casing_residuals/candidates.csv`
- `runs/gemma2/logits_neurons/candidates.csv`
- `runs/gemma2/eval_awareness_residuals/candidates.csv`
- `runs/gemma2/casing_residuals/candidates.csv`

## 5. Run robustness checks

Phi 2 example:

```bash
uv run prompt-suppression robustness \
  --spec runs/specs/phi2_logits_neurons.json \
  --records runs/phi2/logits_neurons/candidates.csv \
  --out runs/phi2/logits_neurons/robustness.csv \
  --rows-out runs/phi2/logits_neurons/robustness_rows.csv \
  --summary-out runs/phi2/logits_neurons/robustness_summary.csv \
  --top-n 10
```

Gemma 2 2B example:

```bash
uv run prompt-suppression robustness \
  --spec runs/specs/gemma2_logits_neurons.json \
  --records runs/gemma2/logits_neurons/candidates.csv \
  --out runs/gemma2/logits_neurons/robustness.csv \
  --rows-out runs/gemma2/logits_neurons/robustness_rows.csv \
  --summary-out runs/gemma2/logits_neurons/robustness_summary.csv \
  --top-n 10 \
  --attn-implementation eager \
  --torch-dtype bfloat16
```

Repeat for residual target directories by changing `--spec`, `--records`, and
the output paths.

## 6. Run behavioral scoring

Phi 2:

```bash
uv run prompt-suppression behavior \
  --model-name microsoft/phi-2 \
  --evals data/frontier/behavior/frontier_behavior.json \
  --out runs/phi2/behavior.csv
```

Gemma 2 2B:

```bash
uv run prompt-suppression behavior \
  --model-name google/gemma-2-2b \
  --attn-implementation eager \
  --torch-dtype bfloat16 \
  --evals data/frontier/behavior/frontier_behavior.json \
  --out runs/gemma2/behavior.csv
```

## 7. Export LaTeX tables

Method summary table:

```bash
uv run prompt-suppression latex-table \
  --csv runs/phi2/logits_neurons/summary.csv \
  --out runs/phi2/logits_neurons/summary_table.tex \
  --columns target_name,method,n,best_target,best_target_xentropy,median_target,median_xentropy \
  --caption "Phi 2 logit and neuron suppression summary." \
  --label tab:phi2-logits-neurons
```

Robustness summary table:

```bash
uv run prompt-suppression latex-table \
  --csv runs/phi2/logits_neurons/robustness_summary.csv \
  --out runs/phi2/logits_neurons/robustness_table.tex \
  --columns target_name,base_method,variant,n,survival_rate,median_delta_target,worst_delta_target \
  --caption "Phi 2 robustness summary for top suppressors." \
  --label tab:phi2-robustness
```

Repeat the table export for each run directory you plan to cite in the paper.

## 8. What to copy into the paper

Use only generated artifacts from `runs/`:

- Candidate data: `candidates.csv`
- Method summaries: `summary.csv`
- Robustness summaries: `robustness_summary.csv`
- Behavior scores: `behavior.csv`
- Figures: files under each `figures/` directory
- LaTeX tables: generated `*_table.tex`

Do not manually copy numbers from terminal output.
