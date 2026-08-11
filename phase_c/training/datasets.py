from __future__ import annotations

import json
from pathlib import Path

from phase_c.data.core import (
    DagConfig,
    RandomConfig,
    generate_dag_record,
    generate_random_record,
    random_unit_paths,
    read_jsonl_gzip,
)
from phase_c.data.random_units import _sha256


class RandomRecordDataset:
    def __init__(
        self,
        config: RandomConfig,
        split: str,
        size: int,
        seed: int,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self.config = config
        self.split = split
        self.size = size
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < self.size:
            raise IndexError(index)
        return generate_random_record(
            self.config, self.split, sample_id=index, seed=self.seed
        )


class FileRandomRecordDataset:
    def __init__(self, dataset_root: Path, split: str, units: int) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.units = units
        self.paths = random_unit_paths(self.dataset_root, split, units)
        self.records: list[dict] = []
        for path in self.paths:
            self.records.extend(read_jsonl_gzip(path))
        if not self.records:
            raise ValueError("file-backed dataset must not be empty")
        first = self.records[0]
        self.config = RandomConfig(
            V=int(first["V"]),
            S=int(first["answer_length"]),
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < len(self.records):
            raise IndexError(index)
        return self.records[index]


class DagRecordDataset:
    def __init__(
        self,
        config: DagConfig,
        split: str,
        size: int,
        seed: int,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self.config = config
        self.split = split
        self.size = size
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < self.size:
            raise IndexError(index)
        return generate_dag_record(
            self.config, self.split, sample_id=index, seed=self.seed
        )


class FileDagRecordDataset:
    def __init__(self, dataset_root: Path, split: str, units: int) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.units = units
        self.paths = random_unit_paths(self.dataset_root, split, units)
        manifest = json.loads(
            (self.dataset_root / "dataset_manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("family") != "dag":
            raise ValueError("dataset-root is not a DAG unit dataset")
        split_manifest = manifest.get("splits", {}).get(split)
        if not isinstance(split_manifest, dict):
            raise ValueError(f"DAG manifest has no {split!r} split")
        expected_files = split_manifest.get("files", [])[:units]
        if len(expected_files) != units:
            raise ValueError(f"DAG manifest has fewer than {units} {split} units")
        for path, file_info in zip(self.paths, expected_files, strict=True):
            if path.name != file_info.get("name") or _sha256(path) != file_info.get("sha256"):
                raise ValueError(f"DAG unit does not match manifest: {path}")
        self.records: list[dict] = []
        for path in self.paths:
            self.records.extend(read_jsonl_gzip(path))
        if not self.records:
            raise ValueError("file-backed dataset must not be empty")
        first = self.records[0]
        metadata = first["metadata"]
        self.config = DagConfig(
            V=int(first["V"]),
            L=int(metadata["L"]),
            d=int(metadata["d"]),
            W=len(metadata["layers"][0]),
        )
        expected_config = manifest.get("config")
        if expected_config != {
            "V": self.config.V,
            "L": self.config.L,
            "d": self.config.d,
            "W": self.config.W,
        }:
            raise ValueError("DAG dataset config does not match its manifest")
        if any(record.get("split") != split for record in self.records):
            raise ValueError(f"DAG records do not all belong to {split!r}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < len(self.records):
            raise IndexError(index)
        return self.records[index]
