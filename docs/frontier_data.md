# Frontier Data Sources

Use `examples/` only for smoke tests. Paper experiments should use local files
generated under `data/frontier/`.

## Primary Sources

The data builder supports these sources:

| Source key | Dataset | Why it belongs |
| --- | --- | --- |
| `hle` | `cais/hle` | Expert level, broad coverage, designed for frontier model evaluation. |
| `gpqa_diamond` | `Idavidrein/gpqa`, `gpqa_diamond` config | Graduate level biology, chemistry, and physics questions designed to be difficult even with web access. |
| `mmlu_pro` | `TIGER-Lab/MMLU-Pro` | More difficult and less saturated than MMLU, with 10 answer options and broad subject coverage. |
| `math500` | `HuggingFaceH4/MATH-500` | Standard competition math control set with exact answers. |

`hle` and `gpqa_diamond` are gated. Accept the Hugging Face terms and run
`huggingface-cli login` before building those files. Do not commit generated
benchmark rows or paste raw examples into public artifacts.

## Build Command

```bash
uv sync --extra remote
uv run prompt-suppression build-frontier-data \
  --out-dir data/frontier \
  --sources mmlu_pro math500 gpqa_diamond hle \
  --allow-gated \
  --max-items-per-source 500 \
  --train-fraction 0.75 \
  --behavior-limit 300
```

The command writes:

- `data/frontier/text_pools/frontier_train.txt`
- `data/frontier/text_pools/frontier_eval.txt`
- `data/frontier/contrasts/eval_awareness_train.json`
- `data/frontier/contrasts/eval_awareness_eval.json`
- `data/frontier/contrasts/casing_control.json`
- `data/frontier/contrasts/style_control.json`
- `data/frontier/behavior/frontier_behavior.json`

Use training contrasts only for fitting residual directions. Use the held out
text pool for natural prompt scans, robustness reporting, and paper tables.

## Recommended Paper Framing

The novelty is not that the benchmark questions are new. The paper should frame
the data as a hard held out substrate for measuring whether optimized prompts
can suppress logit, neuron, and residual direction targets while remaining
fluent. The paired wrappers are synthetic by design because they isolate an
evaluation awareness direction while preserving the underlying question.

Report source counts from `data/frontier/manifest.json`, not from memory.
