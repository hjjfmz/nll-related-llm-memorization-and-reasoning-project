from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from phase_c.data.cli import build_parser as build_data_parser
from phase_c.data.cli import command_random_units
from phase_c.experiments.e03_random_capacity.config import (
    make_eval_namespace,
    make_extend_namespace,
    make_resume_namespace,
)
from phase_c.models.inspect import inspect_model
from phase_c.experiments.e03_random_capacity.train import build_parser as build_train_parser
from phase_c.experiments.e03_random_capacity.train import run


def run_random_command(command: str, argv: Sequence[str]) -> dict | None:
    if command == "gen-data":
        return _run_gen_data(argv)
    if command == "inspect":
        return _run_inspect(argv)
    if command == "train":
        args = build_train_parser().parse_args(list(argv))
        args.run_mode = "train"
        args.eval_only = False
        args.extend_lr_policy = None
        return run(args)
    if command == "resume":
        args = _parse_run_dir_command("resume", argv)
        return run(make_resume_namespace(args.run_dir))
    if command == "extend":
        args = _build_extend_parser().parse_args(list(argv))
        namespace = make_extend_namespace(
            args.run_dir,
            extra_epochs=args.extra_epochs,
            lr_policy=args.lr_policy,
            output_dir=args.output_dir,
        )
        return run(namespace)
    if command == "eval":
        args = _build_eval_parser().parse_args(list(argv))
        namespace = make_eval_namespace(args.run_dir, output_dir=args.output_dir)
        if args.eval_size is not None:
            namespace.eval_size = args.eval_size
        return run(namespace)
    raise SystemExit(f"unknown random command: {command}")


def _run_gen_data(argv: Sequence[str]) -> dict | None:
    parser = build_data_parser()
    args = parser.parse_args(["random-units", *list(argv)])
    command_random_units(args)
    return None


def _run_inspect(argv: Sequence[str]) -> dict:
    parser = argparse.ArgumentParser(description="Inspect a model preset.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--V", type=int, default=1024)
    return inspect_model(**vars(parser.parse_args(list(argv))))


def _parse_run_dir_command(name: str, argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{name} a random experiment.")
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args(list(argv))


def _build_extend_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extend a random experiment.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--extra-epochs", type=int, required=True)
    parser.add_argument(
        "--lr-policy",
        choices=("constant-min",),
        default="constant-min",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def _build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a random checkpoint.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--eval-size", type=int, default=None)
    return parser
