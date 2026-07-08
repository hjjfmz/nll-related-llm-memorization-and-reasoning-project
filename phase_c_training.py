from __future__ import annotations

import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from phase_c_data import RandomConfig, generate_random_record


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


class SampleStream:
    def __init__(self, size: int, seed: int) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self.size = size
        self.seed = seed
        self.epoch = 0
        self.position = 0

    def _permutation_parameters(self) -> tuple[int, int]:
        rng = random.Random((self.seed << 32) ^ self.epoch)
        multiplier = rng.randrange(1, self.size + 1)
        while math.gcd(multiplier, self.size) != 1:
            multiplier = multiplier % self.size + 1
        return multiplier, rng.randrange(self.size)

    def _next_global_ids(self, count: int) -> list[int]:
        if count <= 0:
            raise ValueError("count must be positive")
        result = []
        while len(result) < count:
            multiplier, offset = self._permutation_parameters()
            remaining = min(count - len(result), self.size - self.position)
            result.extend(
                (multiplier * position + offset) % self.size
                for position in range(self.position, self.position + remaining)
            )
            self.position += remaining
            if self.position == self.size:
                self.epoch += 1
                self.position = 0
        return result

    def next_ids(self, count: int) -> list[int]:
        return self._next_global_ids(count)

    def next_ids_for_rank(self, count: int, rank: int, world_size: int) -> list[int]:
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        global_ids = self._next_global_ids(count * world_size)
        start = rank * count
        end = start + count
        return global_ids[start:end]

    def state_dict(self) -> dict[str, int]:
        return {
            "size": self.size,
            "seed": self.seed,
            "epoch": self.epoch,
            "position": self.position,
        }

    def load_state_dict(self, state: Mapping[str, int]) -> None:
        if int(state["size"]) != self.size:
            raise ValueError("sample stream size differs from checkpoint")
        self.seed = int(state["seed"])
        self.epoch = int(state["epoch"])
        self.position = int(state["position"])


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels sequence dimensions must match")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction=reduction,
    )


def random_capacity_metrics(
    total_nll_nats: float,
    supervised_tokens: int,
    num_samples: int,
    H_R_bits_per_sample: float,
    non_embedding_parameters: int,
) -> dict[str, float]:
    total_nll_bits = total_nll_nats / math.log(2.0)
    dataset_entropy_bits = num_samples * H_R_bits_per_sample
    memory_bits = dataset_entropy_bits - total_nll_bits
    return {
        "total_nll_bits": total_nll_bits,
        "nll_bits_per_token": total_nll_bits / supervised_tokens,
        "dataset_entropy_bits": dataset_entropy_bits,
        "memory_bits": memory_bits,
        "bits_per_parameter": memory_bits / non_embedding_parameters,
    }


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


@torch.inference_mode()
def evaluate_random_model(
    model: torch.nn.Module,
    dataset: RandomRecordDataset,
    collator: AnswerOnlyCollator,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    non_embedding_parameters: int,
    max_samples: int = 0,
) -> dict[str, float]:
    sample_count = len(dataset) if max_samples == 0 else min(max_samples, len(dataset))
    model.eval()
    total_nll_nats = 0.0
    supervised_tokens = 0
    for start in range(0, sample_count, batch_size):
        records = [
            dataset[index]
            for index in range(start, min(start + batch_size, sample_count))
        ]
        batch = collator(records)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with _autocast_context(device, dtype):
            logits = model(input_ids)
            loss_sum = causal_lm_loss(logits, labels, reduction="sum")
        total_nll_nats += float(loss_sum)
        supervised_tokens += int(batch["supervised_tokens"])
    metrics = random_capacity_metrics(
        total_nll_nats,
        supervised_tokens,
        sample_count,
        dataset.config.H_R_bits,
        non_embedding_parameters,
    )
    metrics.update(samples=sample_count, supervised_tokens=supervised_tokens)
    return metrics


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    stream: SampleStream,
    run_config: Mapping[str, object],
    scheduler: object | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": int(step),
        "stream": stream.state_dict(),
        "run_config": dict(run_config),
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(state, temp_path)
    os.replace(temp_path, path)


def _coerce_rng_state(state: object) -> torch.Tensor:
    if isinstance(state, torch.Tensor):
        return state.to(dtype=torch.uint8, device="cpu")
    return torch.tensor(state, dtype=torch.uint8)


def _coerce_cuda_rng_state_all(states: object) -> list[torch.Tensor]:
    if not isinstance(states, (list, tuple)):
        raise TypeError("cuda_rng_state_all must be a sequence")
    return [_coerce_rng_state(state) for state in states]


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    stream: SampleStream,
    map_location: str | torch.device,
    scheduler: object | None = None,
) -> dict:
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state["scheduler"] is not None:
        scheduler.load_state_dict(state["scheduler"])
    stream.load_state_dict(state["stream"])
    torch.set_rng_state(_coerce_rng_state(state["torch_rng_state"]))
    random.setstate(state["python_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in state:
        torch.cuda.set_rng_state_all(
            _coerce_cuda_rng_state_all(state["cuda_rng_state_all"])
        )
    return {"step": int(state["step"]), "run_config": state["run_config"]}
