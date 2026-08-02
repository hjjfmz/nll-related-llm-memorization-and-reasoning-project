from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    name: str
    n_layers: int
    hidden_size: int
    n_heads: int
    context_length: int = 512
    mlp_ratio: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.n_heads <= 0 or self.hidden_size % self.n_heads:
            raise ValueError("n_heads must evenly divide hidden_size")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict:
        return asdict(self)
