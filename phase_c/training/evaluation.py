from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Sequence

import torch

from phase_c.data.core import DagConfig, reorder_dag_edges, solve_unique_path
from phase_c.training.collation import AnswerOnlyCollator
from phase_c.training.datasets import RandomRecordDataset
from phase_c.training.losses import causal_lm_loss


def random_capacity_metrics(
    total_nll_nats: float,
    supervised_tokens: int,
    num_samples: int,
    H_R_bits_per_sample: float,
    parameter_counts: dict[str, int],
) -> dict[str, float]:
    total_nll_bits = total_nll_nats / math.log(2.0)
    dataset_entropy_bits = num_samples * H_R_bits_per_sample
    memory_bits = dataset_entropy_bits - total_nll_bits
    return {
        "total_nll_bits": total_nll_bits,
        "nll_bits_per_token": total_nll_bits / supervised_tokens,
        "dataset_entropy_bits": dataset_entropy_bits,
        "memory_bits": memory_bits,
        "bits_per_parameter": memory_bits / parameter_counts["total"],
        "bits_per_total_parameter": memory_bits / parameter_counts["total"],
        "bits_per_non_embedding_parameter": memory_bits
        / parameter_counts["non_embedding"],
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
    parameter_counts: dict[str, int],
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
        parameter_counts,
    )
    metrics.update(samples=sample_count, supervised_tokens=supervised_tokens)
    return metrics


def dag_reasoning_metrics(
    total_nll_nats: float,
    supervised_tokens: int,
    num_samples: int,
    config: DagConfig,
    em_correct: int,
    marginal_correct: Sequence[int],
    conditional_correct: Sequence[int],
    first_error_sum: int,
    valid_path_count: int,
    solver_correct_count: int,
) -> dict[str, float]:
    """Aggregate per-sample DAG eval counters into reasoning metrics.

    - ``nll_bits_per_sample``: mean answer-token NLL in bits per sample.
    - ``lambda`` = ``(nll_bits_per_sample - H_L) / L``: extra bits per depth
      step beyond the pure-logic lower bound ``H_L = L log2(d)``.
    - ``em``: exact path match rate; ``path_validity_rate``: decoded path is a
      real source->target path; ``solver_em``: deterministic solver upper bound.
    """
    L = config.L
    total_nll_bits = total_nll_nats / math.log(2.0)
    nll_bits_per_sample = total_nll_bits / num_samples
    H_L = config.H_L_bits
    H_R = L * math.log2(config.V)
    return {
        "total_nll_bits": total_nll_bits,
        "nll_bits_per_token": total_nll_bits / supervised_tokens,
        "nll_bits_per_sample": nll_bits_per_sample,
        "H_L_bits": H_L,
        "H_R_bits": H_R,
        "lambda": (nll_bits_per_sample - H_L) / L,
        "memory_bits": num_samples * H_R - total_nll_bits,
        "em": em_correct / num_samples,
        "stepwise_marginal_accuracy": [c / num_samples for c in marginal_correct],
        "stepwise_conditional_accuracy": [c / num_samples for c in conditional_correct],
        "stepwise_marginal_accuracy_mean": sum(marginal_correct) / (num_samples * L),
        "stepwise_conditional_accuracy_mean": sum(conditional_correct)
        / (num_samples * L),
        "first_error_position_mean": first_error_sum / num_samples,
        "path_validity_rate": valid_path_count / num_samples,
        "random_choice_step_accuracy": 1.0 / config.d,
        "random_choice_em_baseline": (1.0 / config.d) ** L,
        "solver_em": solver_correct_count / num_samples,
        "samples": num_samples,
        "supervised_tokens": supervised_tokens,
    }


def _is_valid_decoded_path(record: Mapping, decoded: Sequence[int]) -> bool:
    edges = {tuple(edge) for edge in record["metadata"]["edges"]}
    current = int(record["metadata"]["source"])
    target = int(record["metadata"]["target"])
    for node in decoded:
        if (current, node) not in edges:
            return False
        current = node
    return current == target


def _solver_correct(record: Mapping) -> bool:
    try:
        path = solve_unique_path(
            record["metadata"]["edges"],
            int(record["metadata"]["source"]),
            int(record["metadata"]["target"]),
        )
    except ValueError:
        return False
    return list(path[1:]) == list(record["target_ids"])


@torch.inference_mode()
def evaluate_dag_model(
    model: torch.nn.Module,
    dataset: object,
    collator: AnswerOnlyCollator,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    max_samples: int = 0,
    reorder_edges_seed: int | None = None,
) -> dict[str, float]:
    """Evaluate a model on DAG records.

    Computes answer-token NLL plus greedy-decoding metrics (EM, stepwise,
    first-error position, path validity).  With ``reorder_edges_seed`` every
    record is edge-reordered before evaluation (edge-order invariance control).
    """
    sample_count = len(dataset) if max_samples == 0 else min(max_samples, len(dataset))
    config = dataset.config
    L = config.L
    model.eval()
    total_nll_nats = 0.0
    supervised_tokens = 0
    em_correct = 0
    marginal_correct = [0] * L
    conditional_correct = [0] * L
    first_error_sum = 0
    valid_path_count = 0
    solver_correct_count = 0

    for start in range(0, sample_count, batch_size):
        indices = range(start, min(start + batch_size, sample_count))
        records: list[dict] = []
        for index in indices:
            record = dataset[index]
            if reorder_edges_seed is not None:
                record = reorder_dag_edges(record, reorder_edges_seed + index)
            records.append(record)
        batch = collator(records)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with _autocast_context(device, dtype):
            logits = model(input_ids)
            loss_sum = causal_lm_loss(logits, labels, reduction="sum")
        total_nll_nats += float(loss_sum)
        supervised_tokens += int(batch["supervised_tokens"])
        logits = logits.float()
        for row, record in enumerate(records):
            target = list(record["target_ids"])
            answer_start = int(record["metadata"]["answer_start"])
            decoded = logits[
                row, answer_start - 1 : answer_start - 1 + L
            ].argmax(-1).tolist()
            if decoded == target:
                em_correct += 1
            ok_prefix = True
            for position in range(L):
                if ok_prefix and decoded[position] == target[position]:
                    conditional_correct[position] += 1
                    marginal_correct[position] += 1
                else:
                    ok_prefix = False
                    if decoded[position] == target[position]:
                        marginal_correct[position] += 1
            first_error_sum += _first_error_position(decoded, target, L)
            if _is_valid_decoded_path(record, decoded):
                valid_path_count += 1
            if _solver_correct(record):
                solver_correct_count += 1

    return dag_reasoning_metrics(
        total_nll_nats,
        supervised_tokens,
        sample_count,
        config,
        em_correct,
        marginal_correct,
        conditional_correct,
        first_error_sum,
        valid_path_count,
        solver_correct_count,
    )


def _first_error_position(decoded: Sequence[int], target: Sequence[int], L: int) -> int:
    for position in range(L):
        if decoded[position] != target[position]:
            return position
    return L
