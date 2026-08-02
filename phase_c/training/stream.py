from __future__ import annotations

import math
import random
from typing import Mapping


class SampleStream:
    def __init__(self, size: int, seed: int) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self.size = size
        self.seed = seed
        self.epoch = 0
        self.position = 0

    def _permutation_parameters(self) -> tuple[int, int]:
        rng = random.Random((self.seed << 32) ^ self.epoch)
        multiplier = rng.randrange(1, self.size + 1)
        while math.gcd(multiplier, self.size) != 1:
            multiplier = multiplier % self.size + 1
        return multiplier, rng.randrange(self.size)

    def _next_global_ids(self, count: int) -> list[int]:
        if count <= 0:
            raise ValueError("count must be positive")
        result = []
        while len(result) < count:
            multiplier, offset = self._permutation_parameters()
            remaining = min(count - len(result), self.size - self.position)
            result.extend(
                (multiplier * position + offset) % self.size
                for position in range(self.position, self.position + remaining)
            )
            self.position += remaining
            if self.position == self.size:
                self.epoch += 1
                self.position = 0
        return result

    def next_ids(self, count: int) -> list[int]:
        return self._next_global_ids(count)

    def next_ids_for_rank(self, count: int, rank: int, world_size: int) -> list[int]:
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        global_ids = self._next_global_ids(count * world_size)
        start = rank * count
        end = start + count
        return global_ids[start:end]

    def state_dict(self) -> dict[str, int]:
        return {
            "size": self.size,
            "seed": self.seed,
            "epoch": self.epoch,
            "position": self.position,
        }

    def load_state_dict(self, state: Mapping[str, int]) -> None:
        if int(state["size"]) != self.size:
            raise ValueError("sample stream size differs from checkpoint")
        self.seed = int(state["seed"])
        self.epoch = int(state["epoch"])
        self.position = int(state["position"])
