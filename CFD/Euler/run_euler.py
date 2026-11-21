#!/usr/bin/env python3
"""Command-line runner for the unified one-dimensional Euler solver.

Examples
--------
    python run_euler.py
    python run_euler.py --list
    python run_euler.py --case fv_single_sod
    python run_euler.py --config configs/dg_two_sod.yaml
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from euler_unified.config import (
    DEFAULT_REPORT,
    discover_config_paths,
    load_config,
)
from euler_unified.run import run_case, write_report


def build_parser():
    parser = argparse.ArgumentParser(
        description="Unified 1-D Euler solver (FV/DG, single/two-medium)."
    )
    parser.add_argument(
        "--config", nargs="+", default=None,
        help="YAML files or directories; default: configs/*.yaml",
    )
    parser.add_argument(
        "--case", default=None,
        help="comma-separated case names to run (from the selected configs)",
    )
    parser.add_argument("--list", action="store_true", help="list cases and exit")
    parser.add_argument("--report", default=None, help="Markdown report path")
    parser.add_argument("--no-report", action="store_true",
                        help="do not write a Markdown report")
    parser.add_argument("--no-plot", action="store_true",
                        help="skip figure generation")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    try:
        paths = discover_config_paths(args.config)
        configs = [load_config(path) for path in paths]
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.case:
        wanted = [name.strip() for name in args.case.split(",") if name.strip()]
        by_name = {cfg["name"]: cfg for cfg in configs}
        missing = [name for name in wanted if name not in by_name]
        if missing:
            print(f"error: unknown case(s): {', '.join(missing)}", file=sys.stderr)
            print("available: " + ", ".join(by_name), file=sys.stderr)
            return 2
        configs = [by_name[name] for name in wanted]

    if args.list:
        print(f"{'case':<26} {'scheme':<6} {'medium':<8} {'order':<6} config")
        for cfg in configs:
            print(f"{cfg['name']:<26} {cfg['scheme'].upper():<6} "
                  f"{cfg['medium']:<8} {cfg['order']:<6} {cfg['config_path']}")
        return 0

    full_run = args.case is None and args.config is None
    if args.report:
        report_path = Path(args.report)
    elif full_run and not args.no_report:
        report_path = DEFAULT_REPORT
    else:
        report_path = None

    items = []
    failed = []
    for index, cfg in enumerate(configs, start=1):
        if not args.quiet:
            print(f"[{index}/{len(configs)}] running {cfg['name']} ...", flush=True)
        try:
            item = run_case(cfg, make_plots=not args.no_plot)
            items.append(item)
            if not args.quiet:
                res = item["result"]
                print(f"    {res['steps']} steps, dt = {res['dt']:.3e}", flush=True)
        except Exception as exc:
            failed.append((cfg["name"], exc))
            print(f"    FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            if not args.quiet:
                traceback.print_exc()

    if report_path is not None and items:
        path = write_report(items, report_path,
                            [item["cfg"]["config_path"] for item in items])
        if not args.quiet:
            print(f"report written to {path}")

    if failed:
        print(f"{len(failed)} case(s) failed:", file=sys.stderr)
        for name, exc in failed:
            print(f"  - {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
