from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_RUNS = (
    ("5k", Path("phase_c_runs/random_120m_N5k_ddp")),
    ("10k", Path("phase_c_runs/random_120m_N10k_ddp")),
    ("100k", Path("phase_c_runs/random_120m_N100k_ddp")),
)


@dataclass(frozen=True)
class RunSummary:
    label: str
    path: Path
    train_size: int
    validation_size: int | None
    completed_steps: int
    train_nll_bits_per_token: float
    validation_nll_bits_per_token: float
    memory_bits: float
    bits_per_parameter: float
    entropy_ceiling_bits_per_parameter: float
    non_embedding_parameters: int
    random_baseline_bits_per_token: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Phase C random-capacity summaries for the 120M model from "
            "final_metrics.json and train_log.jsonl."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        metavar="LABEL=PATH",
        help=(
            "Run directory to include. May be repeated. Defaults to the three "
            "current 120M runs: 5k, 10k, and 100k."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/image"),
        help="Directory for PNG outputs.",
    )
    parser.add_argument(
        "--summary-name",
        default="random_120m_capacity_summary.png",
        help="Filename for the final-metric summary figure.",
    )
    parser.add_argument(
        "--curves-name",
        default="random_120m_training_curves.png",
        help="Filename for the training-loss curve figure.",
    )
    parser.add_argument(
        "--no-curves",
        action="store_true",
        help="Only write the final-metric summary figure.",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write a CSV table next to the figures.",
    )
    return parser.parse_args()


def parse_run_specs(values: list[str] | None) -> list[tuple[str, Path]]:
    if not values:
        return list(DEFAULT_RUNS)
    specs: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--run must have form LABEL=PATH, got {value!r}")
        label, path = value.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"--run must have non-empty LABEL and PATH, got {value!r}")
        specs.append((label, Path(path)))
    return specs


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
    return rows


def run_arguments(run_config: dict) -> dict:
    arguments = run_config.get("arguments")
    if isinstance(arguments, dict):
        return arguments
    return run_config


def int_or_none(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def load_run(label: str, path: Path) -> RunSummary:
    final_metrics = read_json(path / "final_metrics.json")
    run_config_path = path / "run_config.json"
    run_config = read_json(run_config_path) if run_config_path.exists() else {}
    arguments = run_arguments(run_config)

    train = final_metrics["train"]
    validation = final_metrics["validation"]
    parameters = final_metrics["parameters"]

    train_size = int(arguments.get("train_size", train["samples"]))
    validation_size = int_or_none(arguments.get("validation_size"))
    non_embedding_parameters = int(parameters["non_embedding"])
    dataset_entropy_bits = float(train["dataset_entropy_bits"])
    entropy_ceiling = dataset_entropy_bits / non_embedding_parameters

    V = int(arguments.get("V", 1024))
    random_baseline = math.log2(V)

    return RunSummary(
        label=label,
        path=path,
        train_size=train_size,
        validation_size=validation_size,
        completed_steps=int(final_metrics["completed_steps"]),
        train_nll_bits_per_token=float(train["nll_bits_per_token"]),
        validation_nll_bits_per_token=float(validation["nll_bits_per_token"]),
        memory_bits=float(train["memory_bits"]),
        bits_per_parameter=float(train["bits_per_parameter"]),
        entropy_ceiling_bits_per_parameter=entropy_ceiling,
        non_embedding_parameters=non_embedding_parameters,
        random_baseline_bits_per_token=random_baseline,
    )


def load_curve(label: str, path: Path) -> list[dict]:
    rows = read_jsonl(path / "train_log.jsonl")
    curve: list[dict] = []
    for row in rows:
        if "loss_nats_per_token" not in row or "step" not in row:
            continue
        curve.append(
            {
                "label": label,
                "step": int(row["step"]),
                "epoch": float(row.get("epoch", float("nan"))),
                "loss_bits_per_token": float(row["loss_nats_per_token"])
                / math.log(2.0),
            }
        )
    return curve


def require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it in the active "
            "environment, then rerun this script."
        ) from exc
    return plt


def sort_summaries(summaries: Iterable[RunSummary]) -> list[RunSummary]:
    return sorted(summaries, key=lambda item: item.train_size)


def draw_summary(summaries: list[RunSummary], output_path: Path) -> None:
    plt = require_matplotlib()
    summaries = sort_summaries(summaries)
    labels = [summary.label for summary in summaries]
    train_sizes = [summary.train_size for summary in summaries]
    train_nll = [summary.train_nll_bits_per_token for summary in summaries]
    validation_nll = [summary.validation_nll_bits_per_token for summary in summaries]
    memory_mbits = [summary.memory_bits / 1_000_000.0 for summary in summaries]
    bits_per_parameter = [summary.bits_per_parameter for summary in summaries]
    entropy_ceiling = [
        summary.entropy_ceiling_bits_per_parameter for summary in summaries
    ]
    baseline = summaries[0].random_baseline_bits_per_token

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    fig.suptitle("120M Random-Sequence Memorization Capacity", fontsize=14)

    axes[0].plot(train_sizes, train_nll, marker="o", label="train")
    axes[0].plot(train_sizes, validation_nll, marker="s", label="validation")
    axes[0].axhline(
        baseline,
        color="0.45",
        linestyle="--",
        linewidth=1.2,
        label=f"random baseline ({baseline:.1f})",
    )
    axes[0].set_title("Final NLL")
    axes[0].set_ylabel("bits / token")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(train_sizes, memory_mbits, marker="o", color="#1b7f5f")
    axes[1].set_title("Measured Memory")
    axes[1].set_ylabel("Mbits")

    axes[2].plot(
        train_sizes,
        bits_per_parameter,
        marker="o",
        color="#3659a8",
        label="measured",
    )
    axes[2].plot(
        train_sizes,
        entropy_ceiling,
        marker="^",
        color="0.35",
        linestyle="--",
        label="dataset entropy ceiling",
    )
    axes[2].set_title("Capacity Per Parameter")
    axes[2].set_ylabel("bits / non-embedding parameter")
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("train size")
        axis.set_xticks(train_sizes, labels)
        axis.grid(True, which="major", axis="both", alpha=0.25)
        for x_value, label in zip(train_sizes, labels, strict=True):
            axis.annotate(
                label,
                (x_value, axis.get_ylim()[0]),
                xytext=(0, -18),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def draw_curves(run_specs: list[tuple[str, Path]], output_path: Path) -> bool:
    plt = require_matplotlib()
    curves = [point for label, path in run_specs for point in load_curve(label, path)]
    if not curves:
        return False

    fig, axis = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for label, _ in run_specs:
        points = [point for point in curves if point["label"] == label]
        if not points:
            continue
        points.sort(key=lambda point: point["step"])
        axis.plot(
            [point["step"] for point in points],
            [point["loss_bits_per_token"] for point in points],
            label=label,
            linewidth=1.7,
        )

    axis.set_title("120M Training Loss Curves")
    axis.set_xlabel("optimizer step")
    axis.set_ylabel("train loss, bits / token")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return True


def write_table(summaries: list[RunSummary], output_path: Path) -> None:
    summaries = sort_summaries(summaries)
    header = [
        "label",
        "path",
        "train_size",
        "validation_size",
        "completed_steps",
        "train_nll_bits_per_token",
        "validation_nll_bits_per_token",
        "memory_bits",
        "bits_per_parameter",
        "entropy_ceiling_bits_per_parameter",
        "non_embedding_parameters",
    ]
    lines = [",".join(header)]
    for summary in summaries:
        row = [
            summary.label,
            str(summary.path),
            str(summary.train_size),
            "" if summary.validation_size is None else str(summary.validation_size),
            str(summary.completed_steps),
            f"{summary.train_nll_bits_per_token:.10g}",
            f"{summary.validation_nll_bits_per_token:.10g}",
            f"{summary.memory_bits:.10g}",
            f"{summary.bits_per_parameter:.10g}",
            f"{summary.entropy_ceiling_bits_per_parameter:.10g}",
            str(summary.non_embedding_parameters),
        ]
        lines.append(",".join(row))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_specs = parse_run_specs(args.run)
    summaries = [load_run(label, path) for label, path in run_specs]

    summary_path = args.output_dir / args.summary_name
    draw_summary(summaries, summary_path)

    print(f"wrote {summary_path}")

    if args.write_csv:
        table_path = args.output_dir / "random_120m_capacity_summary.csv"
        write_table(summaries, table_path)
        print(f"wrote {table_path}")

    if not args.no_curves:
        curves_path = args.output_dir / args.curves_name
        if draw_curves(run_specs, curves_path):
            print(f"wrote {curves_path}")
        else:
            print("skipped training curve figure: no train_log.jsonl rows found")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
