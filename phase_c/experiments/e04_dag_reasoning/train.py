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
import torch.distributed as dist

from phase_c.data.core import DagConfig, special_tokens, total_vocab_size
from phase_c.models import MODEL_PRESETS, DecoderOnlyTransformer, count_parameters
from phase_c.training import (
    AnswerOnlyCollator,
    DagRecordDataset,
    FileDagRecordDataset,
    SampleStream,
    causal_lm_loss,
    evaluate_dag_model,
    load_checkpoint,
    save_checkpoint,
)
from phase_c.training.common import (
    append_jsonl,
    atomic_write_json,
    build_run_config,
    ddp_reduce_scalar,
    destroy_distributed,
    distributed_context,
    get_arg,
    initialize_distributed,
    lr_factor,
    resolve_device,
    resolve_dtype,
    unwrap_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="From-scratch DAG path-reasoning depth-limit measurement (E4)."
    )
    parser.add_argument("--model", choices=tuple(MODEL_PRESETS), default="L4_H128")
    parser.add_argument("--V", type=int, default=2048)
    parser.add_argument(
        "--L", type=int, default=4, help="DAG path depth (number of answer tokens)"
    )
    parser.add_argument("--d", type=int, default=2, help="per-node out-degree / branch factor")
    parser.add_argument("--W", type=int, default=None, help="layer width; defaults to d+2")
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--test-size", type=int, default=20_000)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Read fixed DAG units from this root instead of online generation.",
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
    parser.add_argument("--test-seed", type=int, default=20270617)
    parser.add_argument("--sampling-seed", type=int, default=20270618)
    parser.add_argument("--model-seed", type=int, default=20270619)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--eval-size",
        type=int,
        default=10_000,
        help="0 evaluates the complete split and is required for formal metrics.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1_000,
        help="run a test monitor eval every this many steps (grokking signal); 0 disables",
    )
    parser.add_argument(
        "--monitor-eval-size",
        type=int,
        default=2_000,
        help="number of test samples per periodic monitor eval",
    )
    parser.add_argument(
        "--edge-reorder-seed",
        type=int,
        default=None,
        help="if set, also evaluate test with edge-reordered inputs (order-invariance control)",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--run-mode",
        choices=("train", "resume", "extend", "eval"),
        default="train",
        help="Internal execution mode used by the package CLI.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load a checkpoint and write final metrics without training steps.",
    )
    parser.add_argument(
        "--extend-lr-policy",
        choices=("constant-min",),
        default=None,
        help="Learning-rate policy for explicit extension runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("phase_c_runs/dag_L4_H128_depth4_d2"),
    )
    return parser


def _dag_config(args: argparse.Namespace) -> DagConfig:
    width = args.W if args.W is not None else args.d + 2
    return DagConfig(V=args.V, L=args.L, d=args.d, W=width)


def _build_datasets(
    args: argparse.Namespace,
) -> tuple[DagRecordDataset | FileDagRecordDataset, DagRecordDataset | FileDagRecordDataset, DagConfig]:
    if args.dataset_root is None:
        data_config = _dag_config(args)
        train_dataset = DagRecordDataset(
            data_config, "train", args.train_size, args.train_seed
        )
        test_dataset = DagRecordDataset(
            data_config, "test", args.test_size, args.test_seed
        )
        return train_dataset, test_dataset, data_config

    if args.train_units is None or args.test_units is None:
        raise ValueError("dataset-root requires both --train-units and --test-units")
    train_dataset = FileDagRecordDataset(args.dataset_root, "train", args.train_units)
    test_dataset = FileDagRecordDataset(args.dataset_root, "test", args.test_units)
    if train_dataset.config != test_dataset.config:
        raise ValueError("train and test file-backed datasets have different configs")
    return train_dataset, test_dataset, train_dataset.config


def run(args: argparse.Namespace) -> dict | None:
    context = distributed_context()
    device = resolve_device(args.device, context)
    initialize_distributed(context, device)
    try:
        dtype = resolve_dtype(args.dtype, device)

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
        sequence_length = data_config.sequence_length
        if sequence_length > model_config.context_length:
            raise ValueError(
                f"DAG sequence length {sequence_length} exceeds model context "
                f"{model_config.context_length}"
            )

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
        optimizer = torch.optim.Adam(base_model.parameters(), **optimizer_options)

        per_rank_effective_batch = args.micro_batch_size * args.gradient_accumulation
        global_effective_batch = per_rank_effective_batch * context.world_size
        steps_per_epoch = math.ceil(len(train_dataset) / global_effective_batch)
        if args.max_steps is None and args.epochs is None:
            raise ValueError("set either --max-steps or --epochs")
        max_steps = (
            args.max_steps
            if args.max_steps is not None
            else args.epochs * steps_per_epoch
        )
        minimum_ratio = args.minimum_learning_rate / args.learning_rate
        run_mode = str(get_arg(args, "run_mode", "train"))
        eval_only = bool(get_arg(args, "eval_only", False))
        extend_lr_policy = get_arg(args, "extend_lr_policy", None)
        if run_mode == "extend" and extend_lr_policy == "constant-min":
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
            restore_scheduler = False
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda step: lr_factor(
                    step, max(max_steps, 1), args.warmup_steps, minimum_ratio
                ),
            )
            restore_scheduler = True
        scaler = torch.amp.GradScaler(
            "cuda", enabled=device.type == "cuda" and dtype == torch.float16
        )
        run_config = build_run_config(args, context, max_steps)

        if context.enabled:
            model = torch.nn.parallel.DistributedDataParallel(
                base_model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
            )
        else:
            model = base_model

        if context.is_main_process:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                args.output_dir / "run_config.json",
                {
                    "arguments": run_config,
                    "model": model_config.to_dict(),
                    "parameters": parameter_counts,
                    "H_R_bits_per_sample": data_config.L * math.log2(data_config.V),
                    "H_L_bits_per_sample": data_config.H_L_bits,
                    "sequence_length": sequence_length,
                    "per_rank_effective_batch_size": per_rank_effective_batch,
                    "global_effective_batch_size": global_effective_batch,
                    "device": str(device),
                    "dtype": str(dtype),
                },
            )

        completed_steps = 0
        if eval_only and args.resume is None:
            raise ValueError("eval-only requires --resume")
        if args.resume:
            state = load_checkpoint(
                args.resume,
                unwrap_model(model),
                optimizer,
                stream,
                device,
                scheduler if restore_scheduler and not eval_only else None,
            )
            completed_steps = state["step"]
            if run_mode == "extend" and extend_lr_policy == "constant-min":
                for group in optimizer.param_groups:
                    group["lr"] = args.learning_rate
                    group["initial_lr"] = args.learning_rate
        if eval_only:
            max_steps = completed_steps
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
                    "loss_nats_per_token": ddp_reduce_scalar(
                        accumulated_loss / args.gradient_accumulation,
                        device,
                        context,
                        reduction="mean",
                    ),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "gradient_norm": ddp_reduce_scalar(
                        float(gradient_norm),
                        device,
                        context,
                        reduction="max",
                    ),
                    "elapsed_seconds": elapsed,
                }
                if device.type == "cuda":
                    peak_memory_gb = torch.cuda.max_memory_allocated(device) / 2**30
                    log["cuda_peak_allocated_gb"] = ddp_reduce_scalar(
                        peak_memory_gb,
                        device,
                        context,
                        reduction="max",
                    )
                if context.is_main_process:
                    append_jsonl(args.output_dir / "train_log.jsonl", log)
                    print(json.dumps(log, ensure_ascii=False), flush=True)

            if (
                context.is_main_process
                and args.eval_interval
                and step % args.eval_interval == 0
            ):
                monitor = evaluate_dag_model(
                    unwrap_model(model),
                    test_dataset,
                    collator,
                    args.eval_batch_size,
                    device,
                    dtype,
                    args.monitor_eval_size,
                )
                eval_entry = {
                    "step": step,
                    "epoch": stream.epoch + stream.position / stream.size,
                    "nll_bits_per_token": monitor["nll_bits_per_token"],
                    "lambda": monitor["lambda"],
                    "em": monitor["em"],
                    "path_validity_rate": monitor["path_validity_rate"],
                    "elapsed_seconds": time.perf_counter() - started,
                }
                append_jsonl(args.output_dir / "eval_log.jsonl", eval_entry)
                print("EVAL " + json.dumps(eval_entry, ensure_ascii=False), flush=True)

            if step % args.checkpoint_interval == 0 or step == max_steps:
                if context.enabled:
                    dist.barrier()
                if context.is_main_process:
                    save_checkpoint(
                        args.output_dir / "checkpoint_latest.pt",
                        unwrap_model(model),
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

        train_metrics = evaluate_dag_model(
            unwrap_model(model),
            train_dataset,
            collator,
            args.eval_batch_size,
            device,
            dtype,
            args.eval_size,
        )
        test_metrics = evaluate_dag_model(
            unwrap_model(model),
            test_dataset,
            collator,
            args.eval_batch_size,
            device,
            dtype,
            args.eval_size,
        )
        metrics = {
            "completed_steps": completed_steps,
            "train": train_metrics,
            "test": test_metrics,
            "parameters": parameter_counts,
            "config": {
                "V": data_config.V,
                "L": data_config.L,
                "d": data_config.d,
                "W": data_config.W,
                "H_L_bits": data_config.H_L_bits,
                "H_R_bits": data_config.L * math.log2(data_config.V),
                "sequence_length": sequence_length,
            },
            "formal_capacity_evaluation": train_metrics["samples"] == len(train_dataset),
        }
        if args.edge_reorder_seed is not None:
            metrics["test_edge_reordered"] = evaluate_dag_model(
                unwrap_model(model),
                test_dataset,
                collator,
                args.eval_batch_size,
                device,
                dtype,
                args.eval_size,
                reorder_edges_seed=args.edge_reorder_seed,
            )
        atomic_write_json(args.output_dir / "final_metrics.json", metrics)
        return metrics
    finally:
        destroy_distributed(context)


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
