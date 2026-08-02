from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from phase_c.experiments.e03_random_capacity.train import build_parser


class RandomCommandError(ValueError):
    """Raised when a random experiment command would create an unsafe run."""


RESUME_LOCKED_FIELDS = {
    "model",
    "V",
    "S",
    "q",
    "key_seed",
    "train_size",
    "validation_size",
    "dataset_root",
    "train_units",
    "test_units",
    "train_seed",
    "validation_seed",
    "sampling_seed",
    "model_seed",
    "micro_batch_size",
    "gradient_accumulation",
    "epochs",
    "max_steps",
    "learning_rate",
    "minimum_learning_rate",
    "warmup_steps",
    "weight_decay",
    "beta1",
    "beta2",
    "dtype",
}


def write_run_arguments(run_dir: Path, arguments: Mapping[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_config.json"
    path.write_text(
        json.dumps({"arguments": _jsonify_arguments(arguments)}, indent=2),
        encoding="utf-8",
    )


def load_saved_run_arguments(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_config.json"
    if not path.exists():
        raise RandomCommandError(f"missing run config: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    arguments = payload.get("arguments", payload)
    if not isinstance(arguments, dict):
        raise RandomCommandError(f"invalid run config arguments: {path}")
    return dict(arguments)


def make_train_namespace(overrides: Mapping[str, Any] | None = None) -> argparse.Namespace:
    return _namespace_from_arguments(overrides or {})


def make_resume_namespace(
    run_dir: Path, overrides: Mapping[str, Any] | None = None
) -> argparse.Namespace:
    arguments = load_saved_run_arguments(run_dir)
    overrides = dict(overrides or {})
    blocked = sorted(field for field in overrides if field in RESUME_LOCKED_FIELDS)
    if blocked:
        raise RandomCommandError(
            "resume cannot override training-defining fields: " + ", ".join(blocked)
        )
    arguments.update(overrides)
    arguments["output_dir"] = str(run_dir)
    arguments["resume"] = str(run_dir / "checkpoint_latest.pt")
    arguments["run_mode"] = "resume"
    arguments["eval_only"] = False
    return _namespace_from_arguments(arguments)


def make_extend_namespace(
    run_dir: Path,
    extra_epochs: int,
    lr_policy: str = "constant-min",
    output_dir: Path | None = None,
) -> argparse.Namespace:
    if extra_epochs <= 0:
        raise RandomCommandError("extra_epochs must be positive")
    if lr_policy != "constant-min":
        raise RandomCommandError("only constant-min extend policy is currently supported")

    arguments = load_saved_run_arguments(run_dir)
    base_epochs = int(arguments.get("epochs") or 0)
    if base_epochs <= 0:
        raise RandomCommandError("extend requires a saved positive epochs value")
    minimum_lr = float(arguments.get("minimum_learning_rate", 3e-5))
    arguments["epochs"] = base_epochs + extra_epochs
    arguments["max_steps"] = None
    arguments["learning_rate"] = minimum_lr
    arguments["minimum_learning_rate"] = minimum_lr
    arguments["warmup_steps"] = 0
    arguments["resume"] = str(run_dir / "checkpoint_latest.pt")
    arguments["output_dir"] = str(output_dir or run_dir)
    arguments["run_mode"] = "extend"
    arguments["extend_lr_policy"] = lr_policy
    arguments["eval_only"] = False
    return _namespace_from_arguments(arguments)


def make_eval_namespace(
    run_dir: Path, output_dir: Path | None = None
) -> argparse.Namespace:
    arguments = load_saved_run_arguments(run_dir)
    arguments["resume"] = str(run_dir / "checkpoint_latest.pt")
    arguments["output_dir"] = str(output_dir or run_dir)
    arguments["run_mode"] = "eval"
    arguments["eval_only"] = True
    arguments["max_steps"] = 0
    return _namespace_from_arguments(arguments)


def _namespace_from_arguments(arguments: Mapping[str, Any]) -> argparse.Namespace:
    parser = build_parser()
    namespace = parser.parse_args([])
    for key, value in arguments.items():
        normalized_key = key.replace("-", "_")
        setattr(namespace, normalized_key, _coerce_value(normalized_key, value))
    _ensure_new_mode_defaults(namespace)
    return namespace


def _ensure_new_mode_defaults(namespace: argparse.Namespace) -> None:
    if not hasattr(namespace, "run_mode"):
        namespace.run_mode = "train"
    if not hasattr(namespace, "eval_only"):
        namespace.eval_only = False
    if not hasattr(namespace, "extend_lr_policy"):
        namespace.extend_lr_policy = None


def _coerce_value(key: str, value: Any) -> Any:
    if value is None:
        return None
    if key in {"dataset_root", "resume", "output_dir"}:
        return Path(os.fspath(value))
    return value


def _jsonify_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: os.fspath(value) if isinstance(value, Path) else value
        for key, value in arguments.items()
    }
