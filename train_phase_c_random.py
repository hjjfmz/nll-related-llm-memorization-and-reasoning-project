from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from phase_c_data import RandomConfig, special_tokens, total_vocab_size
from phase_c_model import MODEL_PRESETS, DecoderOnlyTransformer, count_parameters
from phase_c_training import (
    AnswerOnlyCollator,
    RandomRecordDataset,
    SampleStream,
    causal_lm_loss,
    evaluate_random_model,
    load_checkpoint,
    save_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="From-scratch random-sequence capacity pretraining."
    )
    parser.add_argument("--model", choices=tuple(MODEL_PRESETS), default="120m")
    parser.add_argument("--V", type=int, default=1024)
    parser.add_argument("--S", type=int, default=384)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument("--key-seed", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--validation-size", type=int, default=10_000)
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
        default=Path("phase_c_runs/random_120m_N100k"),
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


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
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


def run(args: argparse.Namespace) -> dict:
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)
    torch.manual_seed(args.model_seed)
    np.random.seed(args.model_seed % (2**32))
    random.seed(args.model_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.model_seed)
        torch.set_float32_matmul_precision("high")

    data_config = RandomConfig(args.V, args.S, args.q, args.key_seed)
    model_config = MODEL_PRESETS[args.model]
    sequence_length = 3 + args.q + 1 + args.S + 1
    if sequence_length > model_config.context_length:
        raise ValueError("data sequence exceeds model context length")

    train_dataset = RandomRecordDataset(
        data_config, "train", args.train_size, args.train_seed
    )
    validation_dataset = RandomRecordDataset(
        data_config, "validation", args.validation_size, args.validation_seed
    )
    collator = AnswerOnlyCollator(special_tokens(args.V)["PAD"])
    stream = SampleStream(len(train_dataset), args.sampling_seed)
    model = DecoderOnlyTransformer(
        model_config, total_vocab_size(args.V)
    ).to(device)
    model.set_gradient_checkpointing(not args.no_gradient_checkpointing)
    parameter_counts = count_parameters(model)

    optimizer_options = {
        "lr": args.learning_rate,
        "betas": (args.beta1, args.beta2),
        "weight_decay": args.weight_decay,
    }
    if device.type == "cuda":
        optimizer_options["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_options)

    effective_batch = args.micro_batch_size * args.gradient_accumulation
    steps_per_epoch = math.ceil(args.train_size / effective_batch)
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
    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    } | {"resolved_max_steps": max_steps}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        args.output_dir / "run_config.json",
        {
            "arguments": run_config,
            "model": model_config.to_dict(),
            "parameters": parameter_counts,
            "H_R_bits_per_sample": data_config.H_R_bits,
            "sequence_length": sequence_length,
            "effective_batch_size": effective_batch,
            "device": str(device),
            "dtype": str(dtype),
        },
    )

    completed_steps = 0
    if args.resume:
        state = load_checkpoint(
            args.resume,
            model,
            optimizer,
            stream,
            device,
            scheduler,
        )
        completed_steps = state["step"]

    started = time.perf_counter()
    for step in range(completed_steps + 1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(args.gradient_accumulation):
            records = [
                train_dataset[index]
                for index in stream.next_ids(args.micro_batch_size)
            ]
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
                "loss_nats_per_token": accumulated_loss
                / args.gradient_accumulation,
                "learning_rate": scheduler.get_last_lr()[0],
                "gradient_norm": float(gradient_norm),
                "elapsed_seconds": elapsed,
            }
            if device.type == "cuda":
                log["cuda_peak_allocated_gb"] = (
                    torch.cuda.max_memory_allocated() / 2**30
                )
            _append_jsonl(args.output_dir / "train_log.jsonl", log)
            print(json.dumps(log, ensure_ascii=False), flush=True)

        if step % args.checkpoint_interval == 0 or step == max_steps:
            save_checkpoint(
                args.output_dir / "checkpoint_latest.pt",
                model,
                optimizer,
                step,
                stream,
                run_config,
                scheduler,
            )

    train_metrics = evaluate_random_model(
        model,
        train_dataset,
        collator,
        args.eval_batch_size,
        device,
        dtype,
        parameter_counts["non_embedding"],
        args.eval_size,
    )
    validation_metrics = evaluate_random_model(
        model,
        validation_dataset,
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
        "validation": validation_metrics,
        "parameters": parameter_counts,
        "formal_capacity_evaluation": train_metrics["samples"]
        == len(train_dataset),
    }
    _atomic_json(args.output_dir / "final_metrics.json", metrics)
    return metrics


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        metrics = run(args)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
