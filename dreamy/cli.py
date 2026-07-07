"""Command line tools for suppression experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dreamy.behavior import load_behavior_evals, score_continuations, write_behavior_rows
from dreamy.benchmarks import (
    epo_suppression_run,
    gcg_suppression_run,
    minscan_baseline,
    random_token_baseline,
)
from dreamy.epo import load_model
from dreamy.plotting import plot_method_bars, plot_scatter
from dreamy.results import records_from_csv, records_to_csv, rows_to_csv, summarize_by_method
from dreamy.robustness import evaluate_robustness, robustness_rows
from dreamy.target_specs import build_runner_from_spec, target_name


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dreamy")
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
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
