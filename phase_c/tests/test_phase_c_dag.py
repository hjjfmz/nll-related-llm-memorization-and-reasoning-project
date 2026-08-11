from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from phase_c.data.core import (
    DagConfig,
    DAG_TASKS,
    generate_dag_record,
    materialize_dag_task,
    special_tokens,
)
from phase_c.data.dag_units import validate_dag_dataset_root, write_dag_units
from phase_c.training import DagTaskCollator, DagRecordDataset, evaluate_dag_model


class DagTaskDataTests(unittest.TestCase):
    def test_canonical_prompt_excludes_path_tokens(self):
        record = generate_dag_record(DagConfig(V=128, L=3, d=2, W=4), "train", 4, 88)
        self.assertEqual(record["input_ids"][-1], special_tokens(128)["ANSWER"])
        self.assertEqual(record["input_ids"].count(special_tokens(128)["ANSWER"]), 1)
        self.assertEqual(record["target_ids"], record["metadata"]["path"][1:-1])

    def test_outcome_and_trace_share_prompt_but_not_targets(self):
        record = generate_dag_record(DagConfig(V=128, L=3, d=2, W=4), "train", 4, 88)
        path = record["metadata"]["path"]
        outcome = materialize_dag_task(record, "outcome")
        trace = materialize_dag_task(record, "trace")

        self.assertEqual(outcome["prompt_ids"], trace["prompt_ids"])
        self.assertEqual(outcome["target_ids"], [path[1]])
        self.assertEqual(trace["target_ids"], path[1:-1])
        self.assertNotIn(path[-1], trace["target_ids"])

    def test_dag_unit_writer_creates_three_disjoint_splits(self):
        config = DagConfig(V=128, L=3, d=2, W=4)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dag_data"
            manifest = write_dag_units(
                output_dir=root,
                config=config,
                train_units=2,
                validation_units=1,
                test_units=1,
                unit_size=10,
                base_seed=1234,
            )

            self.assertEqual(set(manifest["splits"]), {"train", "validation", "test"})
            self.assertTrue((root / "validation" / "1.jsonl.gz").exists())

            from phase_c.training import FileDagRecordDataset

            dataset = FileDagRecordDataset(root, "validation", units=1)
            self.assertEqual(len(dataset), 10)
            self.assertEqual(dataset.split, "validation")
            self.assertEqual(validate_dag_dataset_root(root)["records_checked"], 40)


class ScriptedPathModel(torch.nn.Module):
    def __init__(self, vocab_size: int, answer_token: int, first: int, second: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.answer_token = answer_token
        self.first = first
        self.second = second

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape
        logits = torch.full(
            (batch_size, sequence_length, self.vocab_size), -1000.0, device=input_ids.device
        )
        for row in range(batch_size):
            for position in range(sequence_length):
                previous = int(input_ids[row, position])
                token = self.first if previous == self.answer_token else self.second
                logits[row, position, token] = 0.0
        return logits


class DagEvaluationTests(unittest.TestCase):
    def test_free_run_trace_uses_generated_prefix(self):
        config = DagConfig(V=128, L=3, d=2, W=4)
        record = generate_dag_record(config, "test", 0, 7)
        path = record["metadata"]["path"]
        model = ScriptedPathModel(
            vocab_size=128 + 9,
            answer_token=special_tokens(128)["ANSWER"],
            first=path[1],
            second=path[2],
        )
        metrics = evaluate_dag_model(
            model,
            DagRecordDataset(config, "test", size=1, seed=7),
            DagTaskCollator(special_tokens(128)["PAD"], "trace"),
            task="trace",
            batch_size=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(metrics["free_run_trace_em"], 1.0)
        self.assertEqual(metrics["path_validity_rate"], 1.0)
        self.assertNotIn("lambda", metrics)
        self.assertNotIn("memory_bits", metrics)

    def test_outcome_metrics_use_first_hop_baseline(self):
        config = DagConfig(V=128, L=3, d=2, W=4)
        dataset = DagRecordDataset(config, "test", size=4, seed=7)
        model = ScriptedPathModel(
            vocab_size=128 + 9,
            answer_token=special_tokens(128)["ANSWER"],
            first=dataset[0]["metadata"]["path"][1],
            second=0,
        )
        metrics = evaluate_dag_model(
            model,
            dataset,
            DagTaskCollator(special_tokens(128)["PAD"], "outcome"),
            task="outcome",
            batch_size=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertIn("first_hop_accuracy", metrics)
        self.assertAlmostEqual(metrics["random_choice_accuracy"], 0.5)


class DagTrainingTests(unittest.TestCase):
    def test_dag_resume_rejects_legacy_run_without_task(self):
        from phase_c.experiments.e04_dag_reasoning.config import (
            DagCommandError,
            make_resume_namespace,
            write_run_arguments,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            write_run_arguments(run_dir, {"model": "debug"})
            with self.assertRaises(DagCommandError):
                make_resume_namespace(run_dir)

    def test_dag_train_requires_task_and_validation_units(self):
        from phase_c.experiments.e04_dag_reasoning.train import build_parser

        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        args = parser.parse_args(
            [
                "--task", "outcome", "--dataset-root", "phase_c_dag_data/x",
                "--train-units", "10", "--validation-units", "2", "--test-units", "5",
            ]
        )
        self.assertEqual(args.task, "outcome")
        self.assertEqual(args.validation_units, 2)

    def test_dag_train_cpu_smoke_one_step(self):
        from phase_c.experiments.e04_dag_reasoning.train import build_parser, run

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "run"
            args = build_parser().parse_args(
                [
                    "--task", "trace", "--model", "debug", "--V", "64", "--L", "3",
                    "--d", "2", "--train-size", "20", "--validation-size", "10",
                    "--test-size", "10", "--device", "cpu", "--dtype", "float32",
                    "--micro-batch-size", "1", "--gradient-accumulation", "1",
                    "--max-steps", "1", "--eval-size", "10", "--eval-batch-size", "2",
                    "--eval-interval", "0", "--output-dir", str(output_dir),
                ]
            )
            metrics = run(args)
            self.assertEqual(metrics["completed_steps"], 1)
            self.assertIn("validation", metrics)
            self.assertIn("free_run_trace_em", metrics["test"])


if __name__ == "__main__":
    unittest.main()
