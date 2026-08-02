from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

from phase_c.data.core import RandomConfig, config_to_dict, generate_records


def split_seed(base_seed: int, split: str) -> int:
    split_offset = {"train": 101, "validation": 202, "test": 303}[split]
    return base_seed + 10_000 + split_offset


def write_random_units(
    output_dir: Path,
    config: RandomConfig,
    train_units: int,
    test_units: int,
    unit_size: int,
    base_seed: int,
) -> dict:
    if unit_size <= 0:
        raise ValueError("unit-size must be positive")
    if train_units <= 0 or test_units <= 0:
        raise ValueError("train-units and test-units must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_units = {
        "train": train_units,
        "test": test_units,
    }
    split_seeds = {
        "train": split_seed(base_seed, "train"),
        "test": split_seed(base_seed, "test"),
    }
    files: dict[str, list[dict]] = {"train": [], "test": []}
    for split, units in split_units.items():
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for unit_index in range(1, units + 1):
            files[split].append(
                _write_random_unit_file(
                    split_dir / f"{unit_index}.jsonl.gz",
                    config,
                    split,
                    unit_index,
                    unit_size,
                    split_seeds[split],
                )
            )

    manifest = {
        "family": "random",
        "config": config_to_dict(config),
        "unit_size": unit_size,
        "base_seed": base_seed,
        "split_seeds": split_seeds,
        "splits": {
            split: {
                "units": split_units[split],
                "records": split_units[split] * unit_size,
                "files": files[split],
            }
            for split in split_units
        },
    }
    manifest_path = output_dir / "dataset_manifest.json"
    _atomic_write_json(manifest_path, manifest)
    return manifest


def _write_random_unit_file(
    path: Path,
    config: RandomConfig,
    split: str,
    unit_index: int,
    unit_size: int,
    seed: int,
) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    start_id = (unit_index - 1) * unit_size
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(
        temp_path,
        mode="wt",
        encoding="utf-8",
        newline="\n",
        compresslevel=6,
    ) as handle:
        for record in generate_records(
            "random",
            config,
            split,
            unit_size,
            seed,
            start_id=start_id,
        ):
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    os.replace(temp_path, path)
    return {
        "name": path.name,
        "records": unit_size,
        "start_id": start_id,
        "end_id": start_id + unit_size - 1,
        "sha256": _sha256(path),
    }


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
