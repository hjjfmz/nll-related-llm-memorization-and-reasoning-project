from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


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


MODEL_PRESETS = {
    "debug": ModelConfig("debug", 4, 128, 4),
    "120m": ModelConfig("120m", 12, 896, 14),
    "350m": ModelConfig("350m", 28, 1024, 16),
    "700m": ModelConfig("700m", 36, 1280, 20),
    "1b": ModelConfig("1b", 36, 1536, 24),
}


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.hidden_size // config.n_heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = x.shape
        query, key, value = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size, sequence_length, self.n_heads, self.head_dim
            ).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, hidden_size
        )
        return self.output(attended)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        intermediate_size = config.hidden_size * config.mlp_ratio
        self.input = nn.Linear(config.hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.output(F.gelu(self.input(x), approximate="tanh")))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.hidden_size)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: ModelConfig, vocab_size: int) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.config = config
        self.vocab_size = vocab_size
        self.gradient_checkpointing = False
        self.token_embedding = nn.Embedding(vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.context_length, config.hidden_size
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize)
        self._scale_residual_projections()

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _scale_residual_projections(self) -> None:
        scale = 0.02 / math.sqrt(2 * self.config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attention.output.weight, mean=0.0, std=scale)
            nn.init.normal_(block.mlp.output.weight, mean=0.0, std=scale)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = enabled

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        sequence_length = input_ids.shape[1]
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds context "
                f"{self.config.context_length}"
            )
        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.dropout(hidden)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        return self.lm_head(self.final_norm(hidden))


def count_parameters(model: DecoderOnlyTransformer) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    embedding_ids = {
        id(model.token_embedding.weight),
        id(model.position_embedding.weight),
    }
    embedding = sum(
        parameter.numel()
        for parameter in model.parameters()
        if id(parameter) in embedding_ids
    )
    return {
        "total": total,
        "embedding": embedding,
        "non_embedding": total - embedding,
    }
