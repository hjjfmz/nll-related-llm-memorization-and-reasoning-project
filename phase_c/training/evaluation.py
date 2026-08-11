from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Sequence

import torch

from phase_c.data.core import (
    DAG_TASKS,
    DagConfig,
    materialize_dag_task,
    reorder_dag_edges,
    solve_unique_path,
)
from phase_c.training.collation import AnswerOnlyCollator, DagTaskCollator
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
        records = [dataset[index] for index in range(start, min(start + batch_size, sample_count))]
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


def _is_valid_generated_trace(record: Mapping, generated: Sequence[int]) -> bool:
    edges = {tuple(edge) for edge in record["metadata"]["edges"]}
    current = int(record["metadata"]["source"])
    target = int(record["metadata"]["target"])
    for node in [*generated, target]:
        if (current, node) not in edges:
            return False
        current = node
    return True


def _solver_correct(record: Mapping) -> bool:
    try:
        path = solve_unique_path(
            record["metadata"]["edges"],
            int(record["metadata"]["source"]),
            int(record["metadata"]["target"]),
        )
    except ValueError:
        return False
    return list(path) == list(record["metadata"]["path"])


@torch.inference_mode()
def _generate_dag_targets(
    model: torch.nn.Module,
    records: Sequence[Mapping[str, object]],
    task: str,
    device: torch.device,
    dtype: torch.dtype,
) -> list[list[int]]:
    examples = [materialize_dag_task(record, task) for record in records]
    prompt_lengths = {len(example["prompt_ids"]) for example in examples}
    target_lengths = {len(example["target_ids"]) for example in examples}
    if len(prompt_lengths) != 1 or len(target_lengths) != 1:
        raise ValueError("DAG free-run batches require matching prompt and target lengths")
    tokens = torch.tensor([example["prompt_ids"] for example in examples], device=device)
    generated: list[list[int]] = [[] for _ in examples]
    for _ in range(target_lengths.pop()):
        with _autocast_context(device, dtype):
            logits = model(tokens)
        next_tokens = logits[:, -1, :].float().argmax(-1)
        for row, token in enumerate(next_tokens.tolist()):
            generated[row].append(token)
        tokens = torch.cat((tokens, next_tokens[:, None]), dim=1)
    return generated


def _first_error_position(decoded: Sequence[int], target: Sequence[int]) -> int:
    for position, (prediction, expected) in enumerate(zip(decoded, target, strict=True)):
        if prediction != expected:
            return position + 1
    return len(target) + 1


@torch.inference_mode()
def evaluate_dag_model(
    model: torch.nn.Module,
    dataset: object,
    collator: DagTaskCollator,
    task: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    max_samples: int = 0,
    reorder_edges_seed: int | None = None,
) -> dict[str, object]:
    """Report teacher-forced NLL separately from true free-run predictions."""
    if task not in DAG_TASKS:
        raise ValueError(f"task must be one of {DAG_TASKS}, got {task!r}")
    sample_count = len(dataset) if max_samples == 0 else min(max_samples, len(dataset))
    config: DagConfig = dataset.config
    model.eval()
    total_nll_nats = 0.0
    supervised_tokens = 0
    first_hop_correct = 0
    trace_em_correct = 0
    stepwise_correct: list[int] | None = None
    first_error_sum = 0
    valid_path_count = 0
    solver_correct_count = 0

    for start in range(0, sample_count, batch_size):
        records = []
        for index in range(start, min(start + batch_size, sample_count)):
            record = dataset[index]
            if reorder_edges_seed is not None:
                record = reorder_dag_edges(record, reorder_edges_seed + index)
            records.append(record)
        batch = collator(records)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with _autocast_context(device, dtype):
            loss_sum = causal_lm_loss(model(input_ids), labels, reduction="sum")
        total_nll_nats += float(loss_sum)
        supervised_tokens += int(batch["supervised_tokens"])

        generated = _generate_dag_targets(model, records, task, device, dtype)
        for record, decoded in zip(records, generated, strict=True):
            target = materialize_dag_task(record, task)["target_ids"]
            if task == "outcome":
                first_hop_correct += int(decoded == target)
            else:
                if stepwise_correct is None:
                    stepwise_correct = [0] * len(target)
                trace_em_correct += int(decoded == target)
                for position, (prediction, expected) in enumerate(zip(decoded, target, strict=True)):
                    stepwise_correct[position] += int(prediction == expected)
                first_error_sum += _first_error_position(decoded, target)
                valid_path_count += int(_is_valid_generated_trace(record, decoded))
            solver_correct_count += int(_solver_correct(record))

    total_nll_bits = total_nll_nats / math.log(2.0)
    metrics: dict[str, object] = {
        "task": task,
        "teacher_forced_total_nll_bits": total_nll_bits,
        "teacher_forced_nll_bits_per_token": total_nll_bits / supervised_tokens,
        "teacher_forced_nll_bits_per_sample": total_nll_bits / sample_count,
        "branching_reference_bits": config.H_L_bits,
        "solver_em": solver_correct_count / sample_count,
        "samples": sample_count,
        "supervised_tokens": supervised_tokens,
    }
    if task == "outcome":
        metrics.update(
            first_hop_accuracy=first_hop_correct / sample_count,
            random_choice_accuracy=1.0 / config.d,
        )
        return metrics

    assert stepwise_correct is not None
    target_length = len(stepwise_correct)
    metrics.update(
        free_run_trace_em=trace_em_correct / sample_count,
        stepwise_accuracy=[count / sample_count for count in stepwise_correct],
        first_error_position_mean=first_error_sum / sample_count,
        path_validity_rate=valid_path_count / sample_count,
        random_choice_trace_em=(1.0 / config.d) ** target_length,
    )
    return metrics
