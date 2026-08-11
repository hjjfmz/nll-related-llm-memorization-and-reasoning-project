from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from phase_c.data.core import materialize_dag_task


@dataclass
class AnswerOnlyCollator:
    pad_token_id: int

    def __call__(self, records: Sequence[Mapping[str, object]]) -> dict[str, torch.Tensor]:
        if not records:
            raise ValueError("records must not be empty")
        max_length = max(len(record["input_ids"]) for record in records)
        input_ids = torch.full(
            (len(records), max_length - 1),
            self.pad_token_id,
            dtype=torch.long,
        )
        labels = torch.full(
            (len(records), max_length - 1),
            -100,
            dtype=torch.long,
        )
        supervised_tokens = 0
        for row, record in enumerate(records):
            sequence = torch.tensor(record["input_ids"], dtype=torch.long)
            input_ids[row, : sequence.numel() - 1] = sequence[:-1]
            answer_start = int(record["metadata"]["answer_start"])
            answer_end = int(record["metadata"]["answer_end"])
            labels[row, answer_start - 1:answer_end - 1] = sequence[
                answer_start:answer_end
            ]
            supervised_tokens += answer_end - answer_start
        if supervised_tokens == 0:
            raise ValueError("batch contains no supervised answer tokens")
        return {
            "input_ids": input_ids,
            "labels": labels,
            "supervised_tokens": torch.tensor(supervised_tokens),
        }


@dataclass
class DagTaskCollator:
    pad_token_id: int
    task: str

    def __call__(self, records: Sequence[Mapping[str, object]]) -> dict[str, torch.Tensor]:
        if not records:
            raise ValueError("records must not be empty")
        examples = [materialize_dag_task(record, self.task) for record in records]
        sequences = [
            [*example["prompt_ids"], *example["target_ids"], self.pad_token_id]
            for example in examples
        ]
        max_length = max(len(sequence) for sequence in sequences)
        input_ids = torch.full(
            (len(sequences), max_length - 1), self.pad_token_id, dtype=torch.long
        )
        labels = torch.full_like(input_ids, -100)
        supervised_tokens = 0
        for row, (sequence, example) in enumerate(zip(sequences, examples, strict=True)):
            prompt_length = len(example["prompt_ids"])
            target_length = len(example["target_ids"])
            input_ids[row, : len(sequence) - 1] = torch.tensor(sequence[:-1])
            labels[row, prompt_length - 1 : prompt_length - 1 + target_length] = torch.tensor(
                example["target_ids"], dtype=torch.long
            )
            supervised_tokens += target_length
        return {
            "input_ids": input_ids,
            "labels": labels,
            "supervised_tokens": torch.tensor(supervised_tokens),
        }
