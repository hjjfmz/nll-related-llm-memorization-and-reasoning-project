"""Shared training-loop helpers used by experiment entry points (E3/E4/E5).

These were consolidated from ``phase_c.experiments.e03_random_capacity.train``
so later experiments can reuse the distributed / device / LR / IO plumbing
without depending on E3 internals.  E3 currently keeps its own copies; new
experiments should import from here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), "utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def distributed_context() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return DistributedContext(
        enabled=world_size > 1,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )


def initialize_distributed(context: DistributedContext, device: torch.device) -> None:
    if not context.enabled:
        return
    if device.type != "cuda":
        raise ValueError("DDP currently requires CUDA for this training script")
    if dist.is_initialized():
        return
    torch.cuda.set_device(context.local_rank)
    backend = "gloo" if os.name == "nt" else "nccl"
    dist.init_process_group(backend=backend)


def destroy_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def resolve_device(requested: str, context: DistributedContext) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", context.local_rank if context.enabled else 0)
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        return torch.device("cuda", context.local_rank if context.enabled else 0)
    if context.enabled:
        raise ValueError("distributed training requires CUDA devices")
    return torch.device(requested)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]
    if device.type == "cpu":
        return torch.float32
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support bfloat16")
    return dtype


def lr_factor(
    step: int, max_steps: int, warmup_steps: int, minimum_ratio: float
) -> float:
    if warmup_steps and step < warmup_steps:
        return max((step + 1) / warmup_steps, 1e-8)
    if max_steps <= warmup_steps:
        return 1.0
    progress = min(
        max((step - warmup_steps) / (max_steps - warmup_steps), 0.0), 1.0
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def ddp_reduce_scalar(
    value: float,
    device: torch.device,
    context: DistributedContext,
    reduction: str,
) -> float:
    if not context.enabled:
        return value
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    op = dist.ReduceOp.SUM if reduction == "mean" else dist.ReduceOp.MAX
    dist.all_reduce(tensor, op=op)
    if reduction == "mean":
        tensor /= context.world_size
    return float(tensor.item())


def get_arg(args: argparse.Namespace, name: str, default: object) -> object:
    return getattr(args, name, default)


def build_run_config(
    args: argparse.Namespace, context: DistributedContext, max_steps: int
) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    } | {
        "resolved_max_steps": max_steps,
        "distributed": {
            "enabled": context.enabled,
            "world_size": context.world_size,
        },
    }
