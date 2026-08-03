from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from phase_c.experiments.e03_random_capacity.commands import run_random_command
from phase_c.experiments.e04_dag_reasoning.commands import run_dag_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase C experiment command line.")
    subparsers = parser.add_subparsers(dest="family", required=True)
    random_parser = subparsers.add_parser(
        "random", help="Random-sequence memory capacity experiments."
    )
    random_parser.add_argument("command")
    random_parser.add_argument("args", nargs=argparse.REMAINDER)
    dag_parser = subparsers.add_parser(
        "dag", help="DAG path-reasoning depth-limit experiments."
    )
    dag_parser.add_argument("command")
    dag_parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(argv)
    if parsed.family == "random":
        result = run_random_command(parsed.command, parsed.args)
        if result is not None:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if parsed.family == "dag":
        result = run_dag_command(parsed.command, parsed.args)
        if result is not None:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    parser.error(f"unknown family: {parsed.family}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
