from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Iterable

from phase_c_data import (
    DagConfig,
    RandomConfig,
    config_to_dict,
    generate_dag_record,
    generate_random_record,
    generate_records,
    run_admission_checks,
    write_jsonl_gzip_shards,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and validate Phase C random/DAG synthetic data."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preview = subparsers.add_parser("preview", help="print one JSON record")
    _add_family_argument(preview, allow_both=False)
    _add_data_arguments(preview)
    preview.add_argument("--split", choices=("train", "validation", "test"), default="train")
    preview.add_argument("--sample-id", type=int, default=0)
    preview.add_argument("--seed", type=int, default=101)

    validate = subparsers.add_parser(
        "validate", help="stream generated records through all admission checks"
    )
    _add_family_argument(validate, allow_both=True)
    _add_data_arguments(validate)
    validate.add_argument("--train-size", type=int, default=100_000)
    validate.add_argument("--validation-size", type=int, default=10_000)
    validate.add_argument("--test-size", type=int, default=20_000)
    validate.add_argument("--base-seed", type=int, default=20260615)
    validate.add_argument(
        "--output-dir", type=Path, default=Path("phase_c_outputs/admission")
    )

    write = subparsers.add_parser(
        "write", help="write one family/split to compressed JSONL shards"
    )
    _add_family_argument(write, allow_both=False)
    _add_data_arguments(write)
    write.add_argument("--split", choices=("train", "validation", "test"), required=True)
    write.add_argument("--count", type=int, default=100_000)
    write.add_argument("--start-id", type=int, default=0)
    write.add_argument("--seed", type=int, required=True)
    write.add_argument("--shard-size", type=int, default=100_000)
    write.add_argument("--compression-level", type=int, default=6)
    write.add_argument("--output-dir", type=Path, required=True)
    return parser


def _add_family_argument(parser: argparse.ArgumentParser, allow_both: bool) -> None:
    choices = ("random", "dag", "both") if allow_both else ("random", "dag")
    parser.add_argument("--family", choices=choices, required=True)


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--V", type=int, default=1024)
    parser.add_argument("--S", type=int, default=384)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument("--key-seed", type=int, default=0)
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument(
        "--W",
        type=int,
        default=None,
        help="layer width; defaults to d+2",
    )


def _random_config(args: argparse.Namespace) -> RandomConfig:
    return RandomConfig(V=args.V, S=args.S, q=args.q, key_seed=args.key_seed)


def _dag_config(args: argparse.Namespace) -> DagConfig:
    width = args.W if args.W is not None else args.d + 2
    return DagConfig(V=args.V, L=args.L, d=args.d, W=width)


def _seed(base_seed: int, family: str, split: str) -> int:
    family_offset = {"random": 10_000, "dag": 20_000}[family]
    split_offset = {"train": 101, "validation": 202, "test": 303}[split]
    return base_seed + family_offset + split_offset


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def command_preview(args: argparse.Namespace) -> int:
    if args.family == "random":
        record = generate_random_record(
            _random_config(args), args.split, args.sample_id, args.seed
        )
    else:
        record = generate_dag_record(
            _dag_config(args), args.split, args.sample_id, args.seed
        )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


def _validation_records(args: argparse.Namespace) -> Iterable[dict]:
    split_sizes = {
        "train": args.train_size,
        "validation": args.validation_size,
        "test": args.test_size,
    }
    families = ("random", "dag") if args.family == "both" else (args.family,)
    iterators = []
    for family in families:
        config = _random_config(args) if family == "random" else _dag_config(args)
        for split, count in split_sizes.items():
            iterators.append(
                generate_records(
                    family,
                    config,
                    split,
                    count,
                    _seed(args.base_seed, family, split),
                )
            )
    return itertools.chain.from_iterable(iterators)


def command_validate(args: argparse.Namespace) -> int:
    if min(args.train_size, args.validation_size, args.test_size) <= 0:
        raise ValueError("all split sizes must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = run_admission_checks(
        _validation_records(args),
        sqlite_path=args.output_dir / "admission.sqlite",
    )
    report["request"] = {
        "family": args.family,
        "random_config": config_to_dict(_random_config(args)),
        "dag_config": config_to_dict(_dag_config(args)),
        "split_sizes": {
            "train": args.train_size,
            "validation": args.validation_size,
            "test": args.test_size,
        },
        "base_seed": args.base_seed,
    }
    report_path = args.output_dir / "admission_report.json"
    _atomic_write_json(report_path, report)
    print(
        json.dumps(
            {
                "all_checks_passed": report["all_checks_passed"],
                "records_checked": report["records_checked"],
                "report": str(report_path.resolve()),
                "sqlite": str((args.output_dir / "admission.sqlite").resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["all_checks_passed"] else 1


def command_write(args: argparse.Namespace) -> int:
    if args.count <= 0:
        raise ValueError("count must be positive")
    config = _random_config(args) if args.family == "random" else _dag_config(args)
    records = generate_records(
        args.family,
        config,
        args.split,
        args.count,
        args.seed,
        start_id=args.start_id,
    )
    prefix = f"{args.family}_{args.split}"
    manifest = write_jsonl_gzip_shards(
        records,
        output_dir=args.output_dir,
        prefix=prefix,
        shard_size=args.shard_size,
        compression_level=args.compression_level,
    )
    manifest.update(
        {
            "family": args.family,
            "split": args.split,
            "seed": args.seed,
            "start_id": args.start_id,
            "config": config_to_dict(config),
        }
    )
    manifest_path = args.output_dir / f"{prefix}_manifest.json"
    _atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "records": manifest["records"],
                "shards": len(manifest["files"]),
                "manifest": str(manifest_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    commands = {
        "preview": command_preview,
        "validate": command_validate,
        "write": command_write,
    }
    try:
        return commands[args.command](args)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
