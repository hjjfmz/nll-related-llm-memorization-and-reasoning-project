from __future__ import annotations

from phase_c.models.transformer import DecoderOnlyTransformer


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
