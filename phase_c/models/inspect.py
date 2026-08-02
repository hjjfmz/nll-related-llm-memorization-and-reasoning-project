from __future__ import annotations

from phase_c.data.core import total_vocab_size
from phase_c.models.counting import count_parameters
from phase_c.models.presets import MODEL_PRESETS
from phase_c.models.transformer import DecoderOnlyTransformer


def inspect_model(model: str, V: int = 1024) -> dict:
    if model not in MODEL_PRESETS:
        choices = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(f"unknown model preset {model!r}; choices: {choices}")
    config = MODEL_PRESETS[model]
    instance = DecoderOnlyTransformer(config, total_vocab_size(V))
    return {
        "model": config.to_dict(),
        "V": V,
        "vocab_size": total_vocab_size(V),
        "head_dim": config.hidden_size // config.n_heads,
        "parameters": count_parameters(instance),
        "non_embedding_formula": (
            f"{config.n_layers} * (12 * {config.hidden_size}^2 "
            f"+ 13 * {config.hidden_size}) + 2 * {config.hidden_size}"
        ),
    }
