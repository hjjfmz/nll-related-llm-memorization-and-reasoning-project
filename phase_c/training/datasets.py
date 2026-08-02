from __future__ import annotations

from pathlib import Path

from phase_c.data.core import (
    RandomConfig,
    generate_random_record,
    random_unit_paths,
    read_jsonl_gzip,
)


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
        if size > config.samples_per_split:
            raise ValueError("size exceeds unique key capacity for this split")
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
            q=len(first["metadata"]["key"]),
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < len(self.records):
            raise IndexError(index)
        return self.records[index]
