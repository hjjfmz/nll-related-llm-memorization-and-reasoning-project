from __future__ import annotations

import hashlib
import gzip
import json
import math
import os
import random
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np


SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}
FAMILY_CODES = {"random": 101, "dag": 211}
SPECIAL_NAMES = (
    "PAD",
    "BOS",
    "EOS",
    "RANDOM",
    "KEY",
    "SEP",
    "GRAPH",
    "QUERY",
    "ANSWER",
)


def special_tokens(V: int) -> dict[str, int]:
    return {name: V + index for index, name in enumerate(SPECIAL_NAMES)}


def total_vocab_size(V: int) -> int:
    return V + len(SPECIAL_NAMES)


@dataclass(frozen=True)
class RandomConfig:
    V: int = 1024
    S: int = 384
    q: int = 4
    key_seed: int = 0

    def __post_init__(self) -> None:
        if self.V < 3:
            raise ValueError("V must be at least 3")
        if self.S <= 0:
            raise ValueError("S must be positive")
        if self.q < 2:
            raise ValueError("q must be at least 2")

    @property
    def H_R_bits(self) -> float:
        return self.S * math.log2(self.V)

    @property
    def samples_per_split(self) -> int:
        return self.V ** (self.q - 1)


@dataclass(frozen=True)
class DagConfig:
    V: int = 1024
    L: int = 4
    d: int = 2
    W: int = 4

    def __post_init__(self) -> None:
        if self.V < 3:
            raise ValueError("V must be at least 3")
        if self.L <= 0:
            raise ValueError("L must be positive")
        if self.d < 2:
            raise ValueError("d must be at least 2")
        if self.W < self.d + 1:
            raise ValueError("W must be at least d + 1")
        required_nodes = (self.L + 1) * self.W
        if required_nodes > self.V:
            raise ValueError(
                f"V={self.V} cannot name {required_nodes} unique graph nodes"
            )

    @property
    def H_L_bits(self) -> float:
        return self.L * math.log2(self.d)

    @property
    def edge_count(self) -> int:
        return self.L * self.W * self.d

    @property
    def sequence_length(self) -> int:
        return 2 * self.edge_count + self.L + 7


def _validate_split(split: str) -> None:
    if split not in SPLIT_CODES:
        raise ValueError(f"split must be one of {tuple(SPLIT_CODES)}, got {split!r}")


def _sample_rng(
    family: str, split: str, sample_id: int, seed: int
) -> np.random.Generator:
    _validate_split(split)
    if sample_id < 0:
        raise ValueError("sample_id must be non-negative")
    sequence = np.random.SeedSequence(
        [int(seed), int(sample_id), FAMILY_CODES[family], SPLIT_CODES[split]]
    )
    return np.random.default_rng(sequence)


def _coprime_multiplier(seed: int, modulus: int) -> int:
    candidate = abs(2 * int(seed) + 1) % modulus
    candidate = max(candidate, 1)
    while math.gcd(candidate, modulus) != 1:
        candidate = (candidate + 1) % modulus
        if candidate == 0:
            candidate = 1
    return candidate


def _encode_base_V(value: int, V: int, width: int) -> list[int]:
    digits = [0] * width
    for index in range(width - 1, -1, -1):
        value, digit = divmod(value, V)
        digits[index] = digit
    if value:
        raise ValueError("value does not fit in requested base-V width")
    return digits


def make_unique_key(config: RandomConfig, split: str, sample_id: int) -> list[int]:
    _validate_split(split)
    if not 0 <= sample_id < config.samples_per_split:
        raise ValueError(
            f"sample_id must be below {config.samples_per_split} for q={config.q}"
        )
    raw = SPLIT_CODES[split] * config.samples_per_split + sample_id
    modulus = config.V ** config.q
    multiplier = _coprime_multiplier(config.key_seed, modulus)
    offset = (
        int.from_bytes(
            hashlib.sha256(str(config.key_seed).encode("ascii")).digest()[:8],
            "big",
        )
        % modulus
    )
    permuted = (multiplier * raw + offset) % modulus
    return _encode_base_V(permuted, config.V, config.q)


def _hash_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def generate_random_record(
    config: RandomConfig,
    split: str,
    sample_id: int,
    seed: int,
) -> dict:
    rng = _sample_rng("random", split, sample_id, seed)
    tokens = special_tokens(config.V)
    key = make_unique_key(config, split, sample_id)
    target = rng.integers(0, config.V, size=config.S, dtype=np.int64).tolist()
    prefix = [
        tokens["BOS"],
        tokens["RANDOM"],
        tokens["KEY"],
        *key,
        tokens["SEP"],
    ]
    answer_start = len(prefix)
    answer_end = answer_start + config.S
    input_ids = [*prefix, *target, tokens["EOS"]]
    return {
        "sample_id": sample_id,
        "family": "random",
        "split": split,
        "seed": seed,
        "V": config.V,
        "answer_length": config.S,
        "input_ids": input_ids,
        "target_ids": target,
        "metadata": {
            "key": key,
            "key_hash": _hash_json(key),
            "target_hash": _hash_json(target),
            "answer_start": answer_start,
            "answer_end": answer_end,
            "H_R_bits": config.H_R_bits,
        },
    }


def _build_dag(
    config: DagConfig, rng: np.random.Generator
) -> tuple[list[list[int]], list[int], list[tuple[int, int]]]:
    node_count = (config.L + 1) * config.W
    nodes = rng.choice(config.V, size=node_count, replace=False)
    layers = nodes.reshape(config.L + 1, config.W).astype(int).tolist()
    path_positions = rng.integers(0, config.W, size=config.L + 1)
    path = [
        layers[layer_index][int(path_positions[layer_index])]
        for layer_index in range(config.L + 1)
    ]

    edges: list[tuple[int, int]] = []
    for layer_index in range(config.L):
        current_path_node = path[layer_index]
        next_path_node = path[layer_index + 1]
        next_nonpath_nodes = [
            node for node in layers[layer_index + 1] if node != next_path_node
        ]
        for source in layers[layer_index]:
            if source == current_path_node:
                distractors = rng.choice(
                    next_nonpath_nodes, size=config.d - 1, replace=False
                ).astype(int).tolist()
                successors = [next_path_node, *distractors]
            else:
                successors = rng.choice(
                    next_nonpath_nodes, size=config.d, replace=False
                ).astype(int).tolist()
            edges.extend((source, target) for target in successors)
    return layers, path, edges


def generate_dag_record(
    config: DagConfig,
    split: str,
    sample_id: int,
    seed: int,
) -> dict:
    rng = _sample_rng("dag", split, sample_id, seed)
    tokens = special_tokens(config.V)
    layers, path, edges = _build_dag(config, rng)
    serialized_edges = list(edges)
    rng.shuffle(serialized_edges)
    flat_edges = [token for edge in serialized_edges for token in edge]
    prefix = [
        tokens["BOS"],
        tokens["GRAPH"],
        *flat_edges,
        tokens["QUERY"],
        path[0],
        path[-1],
        tokens["ANSWER"],
    ]
    target = path[1:]
    answer_start = len(prefix)
    answer_end = answer_start + config.L
    input_ids = [*prefix, *target, tokens["EOS"]]
    graph_identity = {
        "edges": sorted([list(edge) for edge in edges]),
        "source": path[0],
        "target": path[-1],
    }
    return {
        "sample_id": sample_id,
        "family": "dag",
        "split": split,
        "seed": seed,
        "V": config.V,
        "answer_length": config.L,
        "input_ids": input_ids,
        "target_ids": target,
        "metadata": {
            "L": config.L,
            "d": config.d,
            "b_i": [1] * config.L,
            "W": config.W,
            "H_L_bits": config.H_L_bits,
            "layers": layers,
            "path": path,
            "source": path[0],
            "target": path[-1],
            "edges": [list(edge) for edge in edges],
            "serialized_edges": [list(edge) for edge in serialized_edges],
            "graph_hash": _hash_json(graph_identity),
            "answer_start": answer_start,
            "answer_end": answer_end,
            "node_count": node_count_from_layers(layers),
            "edge_count": len(edges),
        },
    }


def node_count_from_layers(layers: Sequence[Sequence[int]]) -> int:
    return sum(len(layer) for layer in layers)


def parse_dag_sequence(input_ids: Sequence[int], V: int) -> dict:
    tokens = special_tokens(V)
    if len(input_ids) < 7:
        raise ValueError("DAG sequence is too short")
    if input_ids[0] != tokens["BOS"] or input_ids[1] != tokens["GRAPH"]:
        raise ValueError("DAG sequence must start with BOS GRAPH")
    try:
        query_index = input_ids.index(tokens["QUERY"], 2)
        answer_index = input_ids.index(tokens["ANSWER"], query_index + 1)
        eos_index = input_ids.index(tokens["EOS"], answer_index + 1)
    except ValueError as exc:
        raise ValueError("DAG sequence is missing QUERY, ANSWER, or EOS") from exc
    flat_edges = list(input_ids[2:query_index])
    if len(flat_edges) % 2:
        raise ValueError("edge token count must be even")
    if answer_index != query_index + 3:
        raise ValueError("QUERY must be followed by exactly source and target")
    edges = [
        (flat_edges[index], flat_edges[index + 1])
        for index in range(0, len(flat_edges), 2)
    ]
    return {
        "edges": edges,
        "source": input_ids[query_index + 1],
        "target": input_ids[query_index + 2],
        "answer": list(input_ids[answer_index + 1:eos_index]),
    }


def reorder_dag_edges(record: Mapping[str, object], seed: int) -> dict:
    if record["family"] != "dag":
        raise ValueError("edge reordering is only defined for DAG records")
    parsed = parse_dag_sequence(record["input_ids"], int(record["V"]))
    edges = list(parsed["edges"])
    random.Random(seed).shuffle(edges)
    tokens = special_tokens(int(record["V"]))
    flat_edges = [token for edge in edges for token in edge]
    input_ids = [
        tokens["BOS"],
        tokens["GRAPH"],
        *flat_edges,
        tokens["QUERY"],
        parsed["source"],
        parsed["target"],
        tokens["ANSWER"],
        *parsed["answer"],
        tokens["EOS"],
    ]
    copied = json.loads(json.dumps(record))
    copied["input_ids"] = input_ids
    copied["metadata"]["serialized_edges"] = [list(edge) for edge in edges]
    copied["metadata"]["answer_start"] = input_ids.index(tokens["ANSWER"]) + 1
    copied["metadata"]["answer_end"] = copied["metadata"]["answer_start"] + len(
        parsed["answer"]
    )
    return copied


def count_paths(
    edges: Sequence[tuple[int, int]],
    source: int,
    target: int,
    stop_at: int = 2,
) -> int:
    adjacency: dict[int, list[int]] = defaultdict(list)
    for start, end in edges:
        adjacency[start].append(end)
    memo: dict[int, int] = {}

    def visit(node: int) -> int:
        if node == target:
            return 1
        if node in memo:
            return memo[node]
        total = 0
        for successor in adjacency.get(node, []):
            total += visit(successor)
            if total >= stop_at:
                memo[node] = stop_at
                return stop_at
        memo[node] = total
        return total

    return visit(source)


def solve_unique_path(
    edges: Sequence[tuple[int, int]], source: int, target: int
) -> list[int]:
    adjacency: dict[int, list[int]] = defaultdict(list)
    for start, end in edges:
        adjacency[start].append(end)
    solutions: list[list[int]] = []

    def search(node: int, path: list[int]) -> None:
        if len(solutions) > 1:
            return
        if node == target:
            solutions.append(path)
            return
        for successor in adjacency.get(node, []):
            search(successor, [*path, successor])

    search(source, [source])
    if len(solutions) != 1:
        raise ValueError(f"expected one path, found {len(solutions)}")
    return solutions[0]


def validate_record(record: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    input_ids = list(record["input_ids"])
    target_ids = list(record["target_ids"])
    metadata = record["metadata"]
    start = int(metadata["answer_start"])
    end = int(metadata["answer_end"])
    if input_ids[start:end] != target_ids:
        errors.append("serialized answer span differs from target_ids")
    if end - start != int(record["answer_length"]):
        errors.append("answer length mismatch")
    if any(not 0 <= token < int(record["V"]) for token in target_ids):
        errors.append("target token outside effective vocabulary")

    if record["family"] == "random":
        expected = int(record["answer_length"]) * math.log2(int(record["V"]))
        if not math.isclose(float(metadata["H_R_bits"]), expected):
            errors.append("H_R mismatch")
        if any(not 0 <= int(token) < int(record["V"]) for token in metadata["key"]):
            errors.append("key token outside effective vocabulary")
        return errors

    if record["family"] != "dag":
        return [*errors, "unsupported family"]

    layers = [list(map(int, layer)) for layer in metadata["layers"]]
    edges = [tuple(map(int, edge)) for edge in metadata["edges"]]
    path = list(map(int, metadata["path"]))
    L = int(metadata["L"])
    d = int(metadata["d"])
    W = int(metadata["W"])
    layer_of = {
        node: layer_index
        for layer_index, layer in enumerate(layers)
        for node in layer
    }
    if len(layers) != L + 1 or any(len(layer) != W for layer in layers):
        errors.append("layer dimensions mismatch")
    if len(layer_of) != (L + 1) * W:
        errors.append("node names are not unique within graph")
    if any(layer_of.get(v) != layer_of.get(u, -2) + 1 for u, v in edges):
        errors.append("edge violates strict layering")
    outdegree = Counter(source for source, _ in edges)
    if any(outdegree[node] != d for layer in layers[:-1] for node in layer):
        errors.append("nonterminal outdegree mismatch")
    if len(edges) != L * W * d:
        errors.append("edge count mismatch")
    if count_paths(edges, path[0], path[-1]) != 1:
        errors.append("source-target path is not unique")
    else:
        if solve_unique_path(edges, path[0], path[-1]) != path:
            errors.append("stored path differs from solved path")
    path_nodes = set(path)
    reverse_reachable = {path[-1]}
    for source, target in reversed(edges):
        if target in reverse_reachable:
            reverse_reachable.add(source)
    nonpath_nodes = set(layer_of) - path_nodes
    if nonpath_nodes.intersection(reverse_reachable):
        errors.append("nonpath node can reach target")
    if target_ids != path[1:]:
        errors.append("target_ids differ from path successors")
    expected_h_l = L * math.log2(d)
    if not math.isclose(float(metadata["H_L_bits"]), expected_h_l):
        errors.append("H_L mismatch")
    try:
        parsed = parse_dag_sequence(input_ids, int(record["V"]))
        if set(parsed["edges"]) != set(edges):
            errors.append("serialized edge set differs from metadata")
        if parsed["source"] != path[0] or parsed["target"] != path[-1]:
            errors.append("serialized query differs from metadata")
        if parsed["answer"] != target_ids:
            errors.append("serialized answer differs from target_ids")
    except ValueError as exc:
        errors.append(str(exc))
    return errors


def generate_records(
    family: str,
    config: RandomConfig | DagConfig,
    split: str,
    count: int,
    seed: int,
    start_id: int = 0,
) -> Iterator[dict]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if family == "random" and not isinstance(config, RandomConfig):
        raise TypeError("random family requires RandomConfig")
    if family == "dag" and not isinstance(config, DagConfig):
        raise TypeError("dag family requires DagConfig")
    generator = generate_random_record if family == "random" else generate_dag_record
    for sample_id in range(start_id, start_id + count):
        yield generator(config, split, sample_id, seed)


def _insert_unique(
    connection: sqlite3.Connection, table: str, value: str
) -> bool:
    try:
        connection.execute(f"INSERT INTO {table}(value) VALUES (?)", (value,))
        return True
    except sqlite3.IntegrityError:
        return False


def run_admission_checks(
    records: Iterable[Mapping[str, object]],
    sqlite_path: Path,
) -> dict:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    if sqlite_path.exists():
        sqlite_path.unlink()
    connection = sqlite3.connect(sqlite_path)
    for table in ("identities", "random_keys", "random_targets", "graph_hashes"):
        connection.execute(f"CREATE TABLE {table}(value TEXT PRIMARY KEY)")

    records_checked = 0
    validation_error_count = 0
    duplicate_identity_count = 0
    duplicate_key_count = 0
    duplicate_target_count = 0
    duplicate_graph_count = 0
    family_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    error_examples: list[dict] = []
    input_lengths: list[int] = []
    target_lengths: list[int] = []
    theoretical_values: dict[str, set[float]] = {
        "H_R_bits": set(),
        "H_L_bits": set(),
    }
    graph_node_counts: set[int] = set()
    graph_edge_counts: set[int] = set()
    random_token_counts: Counter[int] = Counter()
    random_token_total = 0
    dag_path_token_counts: Counter[int] = Counter()
    dag_path_token_total = 0
    dag_nonpath_token_counts: Counter[int] = Counter()
    dag_nonpath_token_total = 0
    effective_vocab_sizes: set[int] = set()

    try:
        for record in records:
            records_checked += 1
            family = str(record["family"])
            family_counts[family] += 1
            split_counts[str(record["split"])] += 1
            input_lengths.append(len(record["input_ids"]))
            target_lengths.append(len(record["target_ids"]))
            effective_vocab_sizes.add(int(record["V"]))
            errors = validate_record(record)
            validation_error_count += len(errors)
            if errors and len(error_examples) < 20:
                error_examples.append(
                    {
                        "family": family,
                        "split": record["split"],
                        "sample_id": record["sample_id"],
                        "errors": errors,
                    }
                )

            metadata = record["metadata"]
            if family == "random":
                theoretical_values["H_R_bits"].add(float(metadata["H_R_bits"]))
                random_token_counts.update(map(int, record["target_ids"]))
                random_token_total += len(record["target_ids"])
                identity = f"random:{metadata['key_hash']}:{metadata['target_hash']}"
                if not _insert_unique(
                    connection, "random_keys", str(metadata["key_hash"])
                ):
                    duplicate_key_count += 1
                if not _insert_unique(
                    connection, "random_targets", str(metadata["target_hash"])
                ):
                    duplicate_target_count += 1
            elif family == "dag":
                theoretical_values["H_L_bits"].add(float(metadata["H_L_bits"]))
                graph_node_counts.add(int(metadata["node_count"]))
                graph_edge_counts.add(int(metadata["edge_count"]))
                path_nodes = set(map(int, metadata["path"]))
                dag_path_token_counts.update(path_nodes)
                dag_path_token_total += len(path_nodes)
                nonpath_nodes = [
                    int(node)
                    for layer in metadata["layers"]
                    for node in layer
                    if int(node) not in path_nodes
                ]
                dag_nonpath_token_counts.update(nonpath_nodes)
                dag_nonpath_token_total += len(nonpath_nodes)
                identity = f"dag:{metadata['graph_hash']}"
                if not _insert_unique(
                    connection, "graph_hashes", str(metadata["graph_hash"])
                ):
                    duplicate_graph_count += 1
            else:
                identity = f"unsupported:{records_checked}"
            if not _insert_unique(connection, "identities", identity):
                duplicate_identity_count += 1

            if records_checked % 10_000 == 0:
                connection.commit()
        connection.commit()
    finally:
        connection.close()

    def balance_diagnostic(counts: Counter[int], total: int, V: int) -> dict:
        expected = total / V if V else 0.0
        if expected < 20:
            return {
                "assessed": False,
                "passed": None,
                "reason": "expected count per token is below 20",
                "total_tokens": total,
                "expected_per_token": expected,
            }
        max_z = max(
            abs(counts.get(token, 0) - expected) / math.sqrt(expected)
            for token in range(V)
        )
        return {
            "assessed": True,
            "passed": max_z <= 8.0,
            "threshold_max_abs_z": 8.0,
            "max_abs_z": max_z,
            "total_tokens": total,
            "expected_per_token": expected,
        }

    balance: dict[str, dict] = {}
    if len(effective_vocab_sizes) == 1:
        only_V = next(iter(effective_vocab_sizes))
        if random_token_total:
            balance["random_targets"] = balance_diagnostic(
                random_token_counts, random_token_total, only_V
            )
        if dag_path_token_total:
            balance["dag_path_nodes"] = balance_diagnostic(
                dag_path_token_counts, dag_path_token_total, only_V
            )
        if dag_nonpath_token_total:
            balance["dag_nonpath_nodes"] = balance_diagnostic(
                dag_nonpath_token_counts, dag_nonpath_token_total, only_V
            )
    assessed_balance_failures = [
        name
        for name, result in balance.items()
        if result["assessed"] and not result["passed"]
    ]

    all_checks_passed = (
        records_checked > 0
        and validation_error_count == 0
        and duplicate_identity_count == 0
        and duplicate_key_count == 0
        and duplicate_target_count == 0
        and duplicate_graph_count == 0
        and not assessed_balance_failures
    )
    return {
        "all_checks_passed": all_checks_passed,
        "records_checked": records_checked,
        "family_counts": dict(family_counts),
        "split_counts": dict(split_counts),
        "validation_error_count": validation_error_count,
        "duplicate_identity_count": duplicate_identity_count,
        "duplicate_key_count": duplicate_key_count,
        "duplicate_target_count": duplicate_target_count,
        "duplicate_graph_count": duplicate_graph_count,
        "length_statistics": {
            "input_tokens_min": min(input_lengths) if input_lengths else None,
            "input_tokens_max": max(input_lengths) if input_lengths else None,
            "target_tokens_min": min(target_lengths) if target_lengths else None,
            "target_tokens_max": max(target_lengths) if target_lengths else None,
        },
        "theoretical_values": {
            name: sorted(values)
            for name, values in theoretical_values.items()
            if values
        },
        "graph_statistics": {
            "node_counts": sorted(graph_node_counts),
            "edge_counts": sorted(graph_edge_counts),
            "path_count_required": 1,
        },
        "word_frequency_balance": balance,
        "assessed_balance_failures": assessed_balance_failures,
        "error_examples": error_examples,
    }


def write_jsonl_atomic(path: Path, records: Iterable[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    os.replace(temp_path, path)
    return count


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl_gzip_shards(
    records: Iterable[Mapping[str, object]],
    output_dir: Path,
    prefix: str,
    shard_size: int = 100_000,
    compression_level: int = 6,
) -> dict:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if not 0 <= compression_level <= 9:
        raise ValueError("compression_level must be between 0 and 9")
    output_dir.mkdir(parents=True, exist_ok=True)

    files: list[dict] = []
    total_records = 0
    shard_records = 0
    shard_index = -1
    handle = None
    temp_path: Path | None = None
    final_path: Path | None = None

    def close_current_shard() -> None:
        nonlocal handle, temp_path, final_path, shard_records
        if handle is None or temp_path is None or final_path is None:
            return
        handle.close()
        os.replace(temp_path, final_path)
        files.append(
            {
                "name": final_path.name,
                "records": shard_records,
                "bytes": final_path.stat().st_size,
                "sha256": _file_sha256(final_path),
            }
        )
        handle = None
        temp_path = None
        final_path = None
        shard_records = 0

    try:
        for record in records:
            if handle is None or shard_records == shard_size:
                close_current_shard()
                shard_index += 1
                final_path = output_dir / f"{prefix}-{shard_index:05d}.jsonl.gz"
                temp_path = final_path.with_suffix(final_path.suffix + ".tmp")
                handle = gzip.open(
                    temp_path,
                    mode="wt",
                    encoding="utf-8",
                    newline="\n",
                    compresslevel=compression_level,
                )
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            )
            handle.write("\n")
            shard_records += 1
            total_records += 1
        close_current_shard()
    except Exception:
        if handle is not None:
            handle.close()
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise

    return {
        "records": total_records,
        "shard_size": shard_size,
        "compression_level": compression_level,
        "files": files,
    }


def read_jsonl_gzip(path: Path) -> Iterator[dict]:
    with gzip.open(path, mode="rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def random_unit_paths(dataset_root: Path, split: str, units: int) -> list[Path]:
    _validate_split(split)
    if units <= 0:
        raise ValueError("units must be positive")
    split_dir = Path(dataset_root) / split
    paths = [split_dir / f"{index}.jsonl.gz" for index in range(1, units + 1)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing random unit file: {missing[0]}")
    return paths


def config_to_dict(config: RandomConfig | DagConfig) -> dict:
    return asdict(config)
