from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Mapping

import torch

from phase_c.training.stream import SampleStream


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
