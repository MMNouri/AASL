"""Command-line entry point: ``aasl <command>``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import AASLError, AwaitingHumanInput

EXIT_WAITING = 3


def _cmd_run(args) -> int:
    from .config import build_run, load_config
    run = build_run(load_config(args.config), verbose=not args.quiet)
    try:
        result = run.run()
    except AwaitingHumanInput as exc:
        print(f"\nPAUSED: {exc}")
        for p in exc.paths:
            print(f"  -> {p}")
        print("Complete the files above, then run the same command again.")
        return EXIT_WAITING
    except AASLError as exc:
        print(f"\nSTOPPED: {exc}", file=sys.stderr)
        return 1
    print(f"\nDone. Final schema v{result.schema.version}: {result.schema.category_names}")
    print(f"Outputs in {run.store.root / 'output'}")
    return 0


def _cmd_status(args) -> int:
    from .config import load_config, _resolve
    cfg = load_config(args.config)
    root = _resolve(cfg["_base"], cfg["run_dir"])
    samples = root / "samples.json"
    if not samples.exists():
        print("No run started yet.")
        return 0
    for name, uids in json.loads(samples.read_text()).items():
        print(f"sample {name}: {len(uids)} units")
    events = root / "events.jsonl"
    if events.exists():
        lines = events.read_text().splitlines()
        print(f"\nlast events ({len(lines)} total):")
        for line in lines[-5:]:
            e = json.loads(line)
            print(f"  {e['time']} {e['event']}")
    return 0


def _cmd_demo(args) -> int:
    from .pipeline import AASLRun, Settings, summarize_codings
    from .simulate import (ScriptedAdjudicator, ScriptedReviser, SimulatedAnnotator,
                           SimulatedModel, make_corpus, seed_schema)
    units, truth = make_corpus(n=args.n, seed=args.seed)
    settings = Settings(tau=0.9, min_support=15, holdout_size=400, calibration_size=400,
                        strata=["sector", "region"], cluster="firm", max_per_cluster=6,
                        seed=args.seed, n_boot=500, backoff=0.0, max_workers=4)
    run = AASLRun(units, seed_schema(), SimulatedModel(truth), SimulatedAnnotator(truth),
                  ScriptedReviser(), ScriptedAdjudicator(), settings, args.run_dir)
    result = run.run()
    print(f"\nFinal schema v{result.schema.version}: {result.schema.category_names}")
    print("Coded units:", summarize_codings(result.codings))
    print(f"Run directory: {Path(args.run_dir).resolve()}")
    return 0


def _cmd_sample_size(args) -> int:
    from .metrics import required_sample_size
    n = required_sample_size(args.p, args.margin, design_effect=args.design_effect)
    print(f"About {n} human-coded units per category to estimate a rate near {args.p} "
          f"within +/-{args.margin} (design effect {args.design_effect}).")
    return 0


def _cmd_codebook(args) -> int:
    from .schema import Schema
    print(Schema.load(args.schema).to_markdown())
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aasl", description="AI-Augmented Supervised Learning")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="run or resume Algorithm 1 from a config file")
    p.add_argument("config")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=_cmd_run)

    p = sub.add_parser("status", help="show samples drawn and recent events")
    p.add_argument("config")
    p.set_defaults(fn=_cmd_status)

    p = sub.add_parser("demo", help="run the full algorithm on a synthetic corpus")
    p.add_argument("--run-dir", default="runs/demo")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(fn=_cmd_demo)

    p = sub.add_parser("sample-size", help="human-coded units needed per category")
    p.add_argument("--p", type=float, default=0.9)
    p.add_argument("--margin", type=float, default=0.05)
    p.add_argument("--design-effect", type=float, default=1.0)
    p.set_defaults(fn=_cmd_sample_size)

    p = sub.add_parser("codebook", help="print a schema file as a coder codebook")
    p.add_argument("schema")
    p.set_defaults(fn=_cmd_codebook)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
