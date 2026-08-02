from __future__ import annotations

import math
from contextlib import nullcontext

import torch

from phase_c.training.collation import AnswerOnlyCollator
from phase_c.training.datasets import RandomRecordDataset
from phase_c.training.losses import causal_lm_loss


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
