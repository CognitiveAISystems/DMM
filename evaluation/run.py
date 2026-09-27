"""Evaluate DMM on POGEMA or MovingAI with fixed protocols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.assets import EVALUATION_ROOT, materialize
from evaluation.models import MODELS


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=tuple(MODELS))
    parser.add_argument("--benchmark", required=True, choices=("pogema", "movingai"))
    parser.add_argument("--package", type=Path,
                        help="AOTI package; defaults to compiled/<model>-<benchmark>.pt2")
    parser.add_argument("--output-root", type=Path, default=ROOT / "eval_results",
                        help="result directory (default: eval_results/)")
    args = parser.parse_args(argv)
    if args.benchmark not in MODELS[args.model]["benchmarks"]:
        parser.error(f"{args.model} is not a verified {args.benchmark} model")
    package = (args.package or ROOT / "compiled" /
               f"{args.model}-{args.benchmark}.pt2").expanduser().resolve()
    if not package.is_file():
        parser.error(f"AOTI package not found: {package}; compile it or pass --package")
    output_root = args.output_root.expanduser().resolve()
    output_dir = output_root / args.benchmark / args.model
    benchmark_root = materialize(
        args.benchmark, output_root / ".benchmark_cache"
    )

    if args.benchmark == "movingai":
        from evaluation.movingai.run import evaluate

        report = evaluate(
            model=args.model, package=package, output_dir=output_dir,
            manifest=EVALUATION_ROOT / "movingai" / "tasks.tsv",
            movingai_root=benchmark_root,
        )
    else:
        from evaluation.pogema.run import evaluate

        report = evaluate(
            model=args.model, package=package, output_dir=output_dir,
            dataset_root=EVALUATION_ROOT / "pogema",
            instance_root=benchmark_root,
        )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
