"""CLI entry point for ragscale: init, run, replay-audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _err(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def _handle_init(args: argparse.Namespace) -> None:
    from ragscale.presets import write_starter_project

    try:
        config = write_starter_project(args.output, preset=args.preset)
    except (FileExistsError, ValueError) as error:
        _err(str(error))
    print(f"Starter project written to {config.parent}")
    print(f"Run: ragscale run {config}")


def _handle_run(args: argparse.Namespace) -> None:
    from ragscale.runner import run_config

    path = Path(args.config)
    if not path.exists():
        _err(f"File not found: {path}")
    try:
        summary = run_config(path, output_dir=args.output)
    except (ImportError, KeyError, RuntimeError, TypeError, ValueError) as error:
        _err(str(error))
    print(f"Audit written to {summary['output_dir']}")
    for result in summary["metrics"]:
        print(
            f"  {result['metric']}: {result['outcome']} "
            f"(raw upgrade {100 * result['raw_upgrade']:+.1f}pp, "
            f"compressed {100 * result['compressed_upgrade']:+.1f}pp)"
        )
    blocked = [
        result for result in summary["metrics"]
        if result["outcome"] in summary["release_gate"]["fail_on_outcomes"]
    ]
    if blocked:
        names = ", ".join(f"{result['metric']}={result['outcome']}" for result in blocked)
        print(f"Release gate failed: {names}", file=sys.stderr)
        sys.exit(2)


def _handle_replay_audit(args: argparse.Namespace) -> None:
    from ragscale.replay import replay_audit, write_replay_audit

    path = Path(args.input)
    if not path.exists():
        _err(f"File not found: {path}")
    try:
        result = replay_audit(path, splits=args.splits, seed=args.seed, metric=args.metric)
    except (KeyError, ValueError) as error:
        _err(str(error))
    if args.output:
        write_replay_audit(result, args.output)
        print(f"Audit written to {args.output}")
    else:
        print(result.to_markdown())


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ragscale",
        description="Audit whether RAG compression preserves a reader comparison",
    )
    subparsers = parser.add_subparsers(dest="command")

    p_init = subparsers.add_parser("init", help="Create a runnable starter project")
    p_init.add_argument("--output", default="ragscale-starter", help="New starter directory")
    p_init.add_argument(
        "--preset", default="local", choices=["local", "openrouter", "openai"],
        help="local: stub adapters; openrouter/openai: bundled HotpotQA rows, key-only run",
    )

    p_run = subparsers.add_parser("run", help="Run a YAML-configured compile-once reader comparison")
    p_run.add_argument("config", help="Path to ragscale.yaml")
    p_run.add_argument("--output", default=None, help="Override the configured output directory")

    p_replay = subparsers.add_parser(
        "replay-audit", help="Audit a fixed artifact across a shared reader panel from paired scores"
    )
    p_replay.add_argument("--input", required=True, help="Paired fixed-artifact CSV")
    p_replay.add_argument("--output", default=None, help="Output directory")
    p_replay.add_argument(
        "--metric", default=None,
        help="Metric to audit when the input holds several (as paired_scores.csv from `ragscale run` does)",
    )
    p_replay.add_argument("--splits", type=int, default=5000)
    p_replay.add_argument("--seed", type=int, default=20260821)

    args = parser.parse_args()
    handlers = {"init": _handle_init, "run": _handle_run, "replay-audit": _handle_replay_audit}
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    handlers[args.command](args)


if __name__ == "__main__":
    main()
