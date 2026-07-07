"""Command line tools for suppression experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from prompt_suppression.behavior import (
    load_behavior_evals,
    score_continuations,
    write_behavior_rows,
    write_behavior_templates,
)
from prompt_suppression.benchmarks import (
    epo_suppression_run,
    gcg_suppression_run,
    minscan_baseline,
    random_search_baseline,
    random_token_baseline,
)
from prompt_suppression.directions import fit_direction_sweep, top_direction_specs
from prompt_suppression.epo import load_model
from prompt_suppression.latex import rows_from_csv, rows_to_latex_table
from prompt_suppression.plotting import plot_method_bars, plot_scatter
from prompt_suppression.results import records_from_csv, records_to_csv, rows_to_csv, summarize_by_method
from prompt_suppression.robustness import evaluate_robustness, robustness_rows, robustness_summary_rows
from prompt_suppression.target_generation import (
    logit_specs,
    neuron_specs,
    parse_int_list,
    residual_specs,
    write_spec,
)
from prompt_suppression.target_specs import build_runner_from_spec, target_name


def _load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_texts(path: str | Path | None) -> list[str]:
    if path is None:
        return []
    with Path(path).open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def run_experiments(args) -> None:
    spec = _load_json(args.spec)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.torch_dtype]
    model, tokenizer = load_model(
        model_size=spec.get("model_size", args.model_size),
        model_name=spec.get("model_name", args.model_name),
        tokenizer_name=spec.get("tokenizer_name"),
        attn_implementation=spec.get("attn_implementation", args.attn_implementation),
        device_map=spec.get("device_map", args.device_map),
        torch_dtype=dtype,
    )
    texts = _load_texts(args.texts or spec.get("texts_path"))
    methods = set(args.methods)
    records = []

    for target_spec in spec["targets"]:
        name = target_name(target_spec)
        runner = build_runner_from_spec(model, tokenizer, target_spec)
        for seed in args.seeds:
            if "epo" in methods:
                records.extend(
                    epo_suppression_run(
                        runner,
                        model,
                        tokenizer,
                        target_name=name,
                        seed=seed,
                        seq_len=args.seq_len,
                        population_size=args.population_size,
                        iters=args.iters,
                        explore_per_pop=args.explore_per_pop,
                        batch_size=args.batch_size,
                        topk=args.topk,
                    )
                )
            if "gcg" in methods:
                records.extend(
                    gcg_suppression_run(
                        runner,
                        model,
                        tokenizer,
                        target_name=name,
                        seed=seed,
                        seq_len=args.seq_len,
                        iters=args.iters,
                        batch_size=args.batch_size,
                        topk=args.topk,
                        x_penalty=args.gcg_x_penalty,
                    )
                )
            if "random" in methods:
                records.extend(
                    random_token_baseline(
                        runner,
                        model,
                        tokenizer,
                        target_name=name,
                        seed=seed,
                        n_prompts=args.random_prompts,
                        seq_len=args.seq_len,
                        batch_size=args.batch_size,
                    )
                )
            if "random_search" in methods:
                records.extend(
                    random_search_baseline(
                        runner,
                        model,
                        tokenizer,
                        target_name=name,
                        seed=seed,
                        population_size=args.population_size,
                        iters=args.iters,
                        explore_per_pop=args.explore_per_pop,
                        seq_len=args.seq_len,
                        batch_size=args.batch_size,
                    )
                )
            if "minscan" in methods:
                if not texts:
                    raise ValueError("minscan requires --texts or texts_path in the spec")
                records.extend(
                    minscan_baseline(
                        runner,
                        model,
                        tokenizer,
                        texts,
                        target_name=name,
                        seed=seed,
                        batch_size=args.batch_size,
                        max_length=args.max_length,
                        fluency_quantile=args.minscan_fluency_quantile,
                    )
                )

    out = Path(args.out)
    records_to_csv(records, out / "candidates.csv")
    rows_to_csv(summarize_by_method(records), out / "summary.csv")


def summarize(args) -> None:
    records = records_from_csv(args.records)
    rows = summarize_by_method(
        records,
        threshold=args.threshold,
        fluent_quantile=args.fluent_quantile,
    )
    rows_to_csv(rows, args.out)


def plot(args) -> None:
    records = records_from_csv(args.records)
    out_dir = Path(args.out_dir)
    for target in sorted({r.target_name for r in records}):
        group = [r for r in records if r.target_name == target]
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in target)
        plot_scatter(group, out_dir / f"{safe}_scatter.png", title=target)
        plot_method_bars(group, out_dir / f"{safe}_bars.png", title=target)


def robustness(args) -> None:
    spec = _load_json(args.spec)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.torch_dtype]
    model, tokenizer = load_model(
        model_size=spec.get("model_size", args.model_size),
        model_name=spec.get("model_name", args.model_name),
        tokenizer_name=spec.get("tokenizer_name"),
        attn_implementation=spec.get("attn_implementation", args.attn_implementation),
        device_map=spec.get("device_map", args.device_map),
        torch_dtype=dtype,
    )
    records = records_from_csv(args.records)
    by_name = {target_name(t): t for t in spec["targets"]}
    robust_records = []
    for name, target_spec in by_name.items():
        group = [r for r in records if r.target_name == name]
        if args.top_n:
            group = sorted(group, key=lambda r: (r.target, r.xentropy))[: args.top_n]
        if not group:
            continue
        runner = build_runner_from_spec(model, tokenizer, target_spec)
        robust_records.extend(
            evaluate_robustness(
                runner,
                model,
                tokenizer,
                group,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
        )
    records_to_csv(robust_records, args.out)
    rows_to_csv(robustness_rows(robust_records), args.rows_out)
    if args.summary_out:
        rows_to_csv(
            robustness_summary_rows(
                robust_records,
                target_tolerance=args.target_tolerance,
            ),
            args.summary_out,
        )


def behavior(args) -> None:
    spec = _load_json(args.spec) if args.spec else {}
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.torch_dtype]
    model, tokenizer = load_model(
        model_size=spec.get("model_size", args.model_size),
        model_name=spec.get("model_name", args.model_name),
        tokenizer_name=spec.get("tokenizer_name"),
        attn_implementation=spec.get("attn_implementation", args.attn_implementation),
        device_map=spec.get("device_map", args.device_map),
        torch_dtype=dtype,
    )
    evals = load_behavior_evals(args.evals)
    rows = score_continuations(model, tokenizer, evals)
    write_behavior_rows(rows, args.out)


def generate_targets(args) -> None:
    targets = []
    if args.tokens:
        targets.extend(logit_specs(args.tokens, prefix=args.logit_prefix))
    if args.token_file:
        targets.extend(logit_specs(_load_texts(args.token_file), prefix=args.logit_prefix))
    if args.layers and args.neurons:
        targets.extend(
            neuron_specs(
                parse_int_list(args.layers),
                parse_int_list(args.neurons),
                prefix=args.neuron_prefix,
            )
        )
    if args.vector:
        layer_by_file = None
        if args.vector_layers:
            layer_by_file = {}
            for part in args.vector_layers.split(","):
                if part.strip():
                    name, layer = part.split("=", 1)
                    layer_by_file[name.strip()] = int(layer)
        targets.extend(
            residual_specs(
                args.vector,
                layer_by_file=layer_by_file,
                default_layer=args.default_vector_layer,
                prefix=args.residual_prefix,
            )
        )
    write_spec(
        targets,
        args.out,
        model_name=args.model_name,
        model_size=args.model_size,
        texts_path=args.texts_path,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
    )


def fit_directions(args) -> None:
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.torch_dtype]
    model, tokenizer = load_model(
        model_size=args.model_size,
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        torch_dtype=dtype,
    )
    rows = fit_direction_sweep(
        model,
        tokenizer,
        args.contrast,
        parse_int_list(args.layers),
        args.out_dir,
        name=args.name,
        pooling=args.pooling,
        max_len=args.max_length,
    )
    if args.spec_out:
        write_spec(
            top_direction_specs(rows, top_k=args.top_k),
            args.spec_out,
            model_name=args.model_name,
            model_size=args.model_size,
            texts_path=args.texts_path,
            attn_implementation=args.attn_implementation,
            device_map=args.device_map,
        )


def latex_table(args) -> None:
    rows_to_latex_table(
        rows_from_csv(args.csv),
        args.out,
        columns=args.columns.split(",") if args.columns else None,
        caption=args.caption or "",
        label=args.label or "",
    )


def behavior_templates(args) -> None:
    write_behavior_templates(args.out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prompt-suppression")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run suppression experiments from a JSON spec")
    run.add_argument("--spec", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--texts")
    run.add_argument("--methods", nargs="+", default=["epo", "random", "minscan"])
    run.add_argument("--seeds", nargs="+", type=int, default=[0])
    run.add_argument("--model-size", default="70m")
    run.add_argument("--model-name")
    run.add_argument("--attn-implementation", default=None)
    run.add_argument("--device-map", default="cuda")
    run.add_argument(
        "--torch-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    run.add_argument("--seq-len", type=int, default=24)
    run.add_argument("--population-size", type=int, default=16)
    run.add_argument("--iters", type=int, default=100)
    run.add_argument("--explore-per-pop", type=int, default=16)
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--topk", type=int, default=256)
    run.add_argument("--random-prompts", type=int, default=256)
    run.add_argument("--max-length", type=int, default=128)
    run.add_argument("--minscan-fluency-quantile", type=float, default=0.2)
    run.add_argument("--gcg-x-penalty", type=float, default=1.0)
    run.set_defaults(func=run_experiments)

    summary = sub.add_parser("summarize", help="summarize candidate CSV output")
    summary.add_argument("--records", required=True)
    summary.add_argument("--out", required=True)
    summary.add_argument("--threshold", type=float)
    summary.add_argument("--fluent-quantile", type=float, default=0.25)
    summary.set_defaults(func=summarize)

    plots = sub.add_parser("plot", help="generate standard figures")
    plots.add_argument("--records", required=True)
    plots.add_argument("--out-dir", required=True)
    plots.set_defaults(func=plot)

    robust = sub.add_parser("robustness", help="evaluate deterministic prompt variants")
    robust.add_argument("--spec", required=True)
    robust.add_argument("--records", required=True)
    robust.add_argument("--out", required=True)
    robust.add_argument("--rows-out", required=True)
    robust.add_argument("--summary-out")
    robust.add_argument("--target-tolerance", type=float, default=0.0)
    robust.add_argument("--top-n", type=int, default=10)
    robust.add_argument("--model-size", default="70m")
    robust.add_argument("--model-name")
    robust.add_argument("--attn-implementation", default=None)
    robust.add_argument("--device-map", default="cuda")
    robust.add_argument(
        "--torch-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    robust.add_argument("--batch-size", type=int, default=64)
    robust.add_argument("--max-length", type=int, default=128)
    robust.set_defaults(func=robustness)

    beh = sub.add_parser("behavior", help="score continuation preferences")
    beh.add_argument("--evals", required=True)
    beh.add_argument("--out", required=True)
    beh.add_argument("--spec")
    beh.add_argument("--model-size", default="70m")
    beh.add_argument("--model-name")
    beh.add_argument("--attn-implementation", default=None)
    beh.add_argument("--device-map", default="cuda")
    beh.add_argument(
        "--torch-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    beh.set_defaults(func=behavior)

    gen = sub.add_parser("generate-targets", help="write target specs")
    gen.add_argument("--out", required=True)
    gen.add_argument("--tokens", nargs="*")
    gen.add_argument("--token-file")
    gen.add_argument("--layers")
    gen.add_argument("--neurons")
    gen.add_argument("--vector", nargs="*")
    gen.add_argument("--vector-layers")
    gen.add_argument("--default-vector-layer", type=int)
    gen.add_argument("--logit-prefix", default="logit")
    gen.add_argument("--neuron-prefix", default="neuron")
    gen.add_argument("--residual-prefix", default="residual")
    gen.add_argument("--model-name")
    gen.add_argument("--model-size")
    gen.add_argument("--texts-path")
    gen.add_argument("--attn-implementation")
    gen.add_argument("--device-map")
    gen.set_defaults(func=generate_targets)

    dirs = sub.add_parser("fit-directions", help="fit residual directions across layers")
    dirs.add_argument("--contrast", required=True)
    dirs.add_argument("--layers", required=True)
    dirs.add_argument("--out-dir", required=True)
    dirs.add_argument("--name", required=True)
    dirs.add_argument("--spec-out")
    dirs.add_argument("--top-k", type=int, default=3)
    dirs.add_argument("--texts-path")
    dirs.add_argument("--pooling", choices=["last", "mean"], default="last")
    dirs.add_argument("--max-length", type=int, default=256)
    dirs.add_argument("--model-size", default="70m")
    dirs.add_argument("--model-name")
    dirs.add_argument("--tokenizer-name")
    dirs.add_argument("--attn-implementation", default=None)
    dirs.add_argument("--device-map", default="cuda")
    dirs.add_argument(
        "--torch-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    dirs.set_defaults(func=fit_directions)

    table = sub.add_parser("latex-table", help="convert a CSV summary to a LaTeX table")
    table.add_argument("--csv", required=True)
    table.add_argument("--out", required=True)
    table.add_argument("--columns")
    table.add_argument("--caption")
    table.add_argument("--label")
    table.set_defaults(func=latex_table)

    beh_tpl = sub.add_parser("behavior-templates", help="write starter behavioral eval templates")
    beh_tpl.add_argument("--out", required=True)
    beh_tpl.set_defaults(func=behavior_templates)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
