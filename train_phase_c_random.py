from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from phase_c_data import RandomConfig, special_tokens, total_vocab_size
from phase_c_model import MODEL_PRESETS, DecoderOnlyTransformer, count_parameters
from phase_c_training import (
    AnswerOnlyCollator,
    FileRandomRecordDataset,
    RandomRecordDataset,
    SampleStream,
    causal_lm_loss,
    evaluate_random_model,
    load_checkpoint,
    save_checkpoint,
)


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="From-scratch random-sequence capacity pretraining."
    )
    parser.add_argument("--model", choices=tuple(MODEL_PRESETS), default="L4_H128")
    parser.add_argument("--V", type=int, default=1024)
    parser.add_argument("--S", type=int, default=32)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument("--key-seed", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--validation-size", type=int, default=10_000)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Read fixed random units from this root instead of online generation.",
    )
    parser.add_argument(
        "--train-units",
        type=int,
        default=None,
        help="Number of 1k train unit files to read from dataset-root/train.",
    )
    parser.add_argument(
        "--test-units",
        type=int,
        default=None,
        help="Number of 1k test unit files to read from dataset-root/test.",
    )
    parser.add_argument("--train-seed", type=int, default=20270616)
    parser.add_argument("--validation-seed", type=int, default=20270617)
    parser.add_argument("--sampling-seed", type=int, default=20270618)
    parser.add_argument("--model-seed", type=int, default=20270619)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--eval-size",
        type=int,
        default=10_000,
        help="0 evaluates the complete split and is required for formal capacity.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("phase_c_runs/random_L4_H128_units"),
    )
    return parser


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), "utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _distributed_context() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return DistributedContext(
        enabled=world_size > 1,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )


def _initialize_distributed(context: DistributedContext, device: torch.device) -> None:
    if not context.enabled:
        return
    if device.type != "cuda":
        raise ValueError("DDP currently requires CUDA for this training script")
    if dist.is_initialized():
        return
    torch.cuda.set_device(context.local_rank)
    backend = "gloo" if os.name == "nt" else "nccl"
    dist.init_process_group(backend=backend)


def _destroy_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def _resolve_device(requested: str, context: DistributedContext) -> torch.device:
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


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
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


def _lr_factor(
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


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _ddp_reduce_scalar(
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


def _build_run_config(args: argparse.Namespace, context: DistributedContext, max_steps: int) -> dict:
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


def _build_datasets(
    args: argparse.Namespace,
) -> tuple[RandomRecordDataset | FileRandomRecordDataset, RandomRecordDataset | FileRandomRecordDataset, RandomConfig]:
    if args.dataset_root is None:
        data_config = RandomConfig(args.V, args.S, args.q, args.key_seed)
        train_dataset = RandomRecordDataset(
            data_config, "train", args.train_size, args.train_seed
        )
        test_dataset = RandomRecordDataset(
            data_config, "test", args.validation_size, args.validation_seed
        )
        return train_dataset, test_dataset, data_config

    if args.train_units is None or args.test_units is None:
        raise ValueError("dataset-root requires both --train-units and --test-units")
    train_dataset = FileRandomRecordDataset(args.dataset_root, "train", args.train_units)
    test_dataset = FileRandomRecordDataset(args.dataset_root, "test", args.test_units)
    if train_dataset.config != test_dataset.config:
        raise ValueError("train and test file-backed datasets have different configs")
    return train_dataset, test_dataset, train_dataset.config


def run(args: argparse.Namespace) -> dict | None:
    context = _distributed_context()
    device = _resolve_device(args.device, context)
    _initialize_distributed(context, device)
    try:
        dtype = _resolve_dtype(args.dtype, device)

        # Keep model initialization identical across ranks.
        torch.manual_seed(args.model_seed)
        np.random.seed(args.model_seed % (2**32))
        random.seed(args.model_seed)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.manual_seed_all(args.model_seed)
            torch.set_float32_matmul_precision("high")
            torch.cuda.reset_peak_memory_stats(device)

        train_dataset, test_dataset, data_config = _build_datasets(args)
        model_config = MODEL_PRESETS[args.model]
        sequence_length = 3 + data_config.q + 1 + data_config.S + 1
        if sequence_length > model_config.context_length:
            raise ValueError("data sequence exceeds model context length")

        collator = AnswerOnlyCollator(special_tokens(data_config.V)["PAD"])
        stream = SampleStream(len(train_dataset), args.sampling_seed)
        base_model = DecoderOnlyTransformer(
            model_config, total_vocab_size(data_config.V)
        ).to(device)
        base_model.set_gradient_checkpointing(not args.no_gradient_checkpointing)
        parameter_counts = count_parameters(base_model)

        optimizer_options = {
            "lr": args.learning_rate,
            "betas": (args.beta1, args.beta2),
            "weight_decay": args.weight_decay,
        }
        if device.type == "cuda":
            optimizer_options["fused"] = True
        optimizer = torch.optim.AdamW(base_model.parameters(), **optimizer_options)

        per_rank_effective_batch = args.micro_batch_size * args.gradient_accumulation
        global_effective_batch = per_rank_effective_batch * context.world_size
        steps_per_epoch = math.ceil(len(train_dataset) / global_effective_batch)
        max_steps = args.max_steps or args.epochs * steps_per_epoch
        minimum_ratio = args.minimum_learning_rate / args.learning_rate
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: _lr_factor(
                step, max_steps, args.warmup_steps, minimum_ratio
            ),
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=device.type == "cuda" and dtype == torch.float16
        )
        run_config = _build_run_config(args, context, max_steps)

        if context.enabled:
            model = DistributedDataParallel(
                base_model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
            )
        else:
            model = base_model

        if context.is_main_process:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            _atomic_json(
                args.output_dir / "run_config.json",
                {
                    "arguments": run_config,
                    "model": model_config.to_dict(),
                    "parameters": parameter_counts,
                    "H_R_bits_per_sample": data_config.H_R_bits,
                    "sequence_length": sequence_length,
                    "per_rank_effective_batch_size": per_rank_effective_batch,
                    "global_effective_batch_size": global_effective_batch,
                    "device": str(device),
                    "dtype": str(dtype),
                },
            )

        completed_steps = 0
        if args.resume:
            state = load_checkpoint(
                args.resume,
                _unwrap_model(model),
                optimizer,
                stream,
                device,
                scheduler,
            )
            completed_steps = state["step"]
        if context.enabled:
            dist.barrier()

        started = time.perf_counter()
        for step in range(completed_steps + 1, max_steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for _ in range(args.gradient_accumulation):
                if context.enabled:
                    sample_ids = stream.next_ids_for_rank(
                        args.micro_batch_size,
                        rank=context.rank,
                        world_size=context.world_size,
                    )
                else:
                    sample_ids = stream.next_ids(args.micro_batch_size)
                records = [train_dataset[index] for index in sample_ids]
                batch = collator(records)
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=dtype,
                    enabled=device.type == "cuda" and dtype != torch.float32,
                ):
                    loss = causal_lm_loss(model(input_ids), labels)
                    scaled_loss = loss / args.gradient_accumulation
                scaler.scale(scaled_loss).backward()
                accumulated_loss += float(loss.detach())

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            completed_steps = step

            if step == 1 or step % args.log_interval == 0:
                elapsed = time.perf_counter() - started
                log = {
                    "step": step,
                    "epoch": stream.epoch + stream.position / stream.size,
                    "loss_nats_per_token": _ddp_reduce_scalar(
                        accumulated_loss / args.gradient_accumulation,
                        device,
                        context,
                        reduction="mean",
                    ),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "gradient_norm": _ddp_reduce_scalar(
                        float(gradient_norm),
                        device,
                        context,
                        reduction="max",
                    ),
                    "elapsed_seconds": elapsed,
                }
                if device.type == "cuda":
                    peak_memory_gb = torch.cuda.max_memory_allocated(device) / 2**30
                    log["cuda_peak_allocated_gb"] = _ddp_reduce_scalar(
                        peak_memory_gb,
                        device,
                        context,
                        reduction="max",
                    )
                if context.is_main_process:
                    _append_jsonl(args.output_dir / "train_log.jsonl", log)
                    print(json.dumps(log, ensure_ascii=False), flush=True)

            if step % args.checkpoint_interval == 0 or step == max_steps:
                if context.enabled:
                    dist.barrier()
                if context.is_main_process:
                    save_checkpoint(
                        args.output_dir / "checkpoint_latest.pt",
                        _unwrap_model(model),
                        optimizer,
                        step,
                        stream,
                        run_config,
                        scheduler,
                    )

        if context.enabled:
            dist.barrier()
        if not context.is_main_process:
            return None

        train_metrics = evaluate_random_model(
            _unwrap_model(model),
            train_dataset,
            collator,
            args.eval_batch_size,
            device,
            dtype,
            parameter_counts["non_embedding"],
            args.eval_size,
        )
        test_metrics = evaluate_random_model(
            _unwrap_model(model),
            test_dataset,
            collator,
            args.eval_batch_size,
            device,
            dtype,
            parameter_counts["non_embedding"],
            args.eval_size,
        )
        metrics = {
            "completed_steps": completed_steps,
            "train": train_metrics,
            "test": test_metrics,
            "parameters": parameter_counts,
            "formal_capacity_evaluation": train_metrics["samples"]
            == len(train_dataset),
        }
        _atomic_json(args.output_dir / "final_metrics.json", metrics)
        return metrics
    finally:
        _destroy_distributed(context)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        metrics = run(args)
        if metrics is not None:
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
