from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from phase_c.data.core import DagConfig, special_tokens, total_vocab_size
from phase_c.data.dag_units import write_dag_units
from phase_c.models import MODEL_PRESETS, DecoderOnlyTransformer
from phase_c.training import (
    AnswerOnlyCollator,
    DagRecordDataset,
    FileDagRecordDataset,
    evaluate_dag_model,
)


class DagUnitDatasetTests(unittest.TestCase):
    def test_write_dag_units_and_file_dataset_prefix_reading(self):
        config = DagConfig(V=128, L=3, d=2, W=4)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dag_data"
            write_dag_units(
                output_dir=root,
                config=config,
                train_units=2,
                test_units=1,
                unit_size=10,
                base_seed=1234,
            )
            self.assertTrue((root / "dataset_manifest.json").exists())
            self.assertTrue((root / "train" / "1.jsonl.gz").exists())
            self.assertTrue((root / "train" / "2.jsonl.gz").exists())
            self.assertTrue((root / "test" / "1.jsonl.gz").exists())

            train_1 = FileDagRecordDataset(root, "train", 1)
            train_2 = FileDagRecordDataset(root, "train", 2)
            self.assertEqual(len(train_1), 10)
            self.assertEqual(len(train_2), 20)
            self.assertEqual(train_2.config, config)

    def test_file_dag_dataset_derives_config_from_record(self):
        config = DagConfig(V=64, L=2, d=4, W=6)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dag_data"
            write_dag_units(root, config, train_units=1, test_units=1, unit_size=5, base_seed=7)
            dataset = FileDagRecordDataset(root, "train", 1)
            self.assertEqual(dataset.config.V, 64)
            self.assertEqual(dataset.config.L, 2)
            self.assertEqual(dataset.config.d, 4)
            self.assertEqual(dataset.config.W, 6)
            record = dataset[0]
            self.assertEqual(len(record["target_ids"]), 2)
            self.assertEqual(record["metadata"]["H_L_bits"], 4.0)


class DagEvaluationTests(unittest.TestCase):
    def _model_and_collator(self, V: int):
        model_config = MODEL_PRESETS["debug"]
        model = DecoderOnlyTransformer(model_config, total_vocab_size(V)).eval()
        collator = AnswerOnlyCollator(special_tokens(V)["PAD"])
        return model, collator

    def test_evaluate_dag_model_metrics_are_defined(self):
        config = DagConfig(V=64, L=4, d=2, W=4)
        model, collator = self._model_and_collator(config.V)
        dataset = DagRecordDataset(config, "test", size=16, seed=20260601)
        metrics = evaluate_dag_model(
            model,
            dataset,
            collator,
            batch_size=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(metrics["samples"], 16)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["nll_bits_per_token"])))
        self.assertGreater(metrics["lambda"], 0.0)
        self.assertTrue(0.0 <= metrics["em"] <= 1.0)
        self.assertTrue(0.0 <= metrics["path_validity_rate"] <= 1.0)
        # Correct synthetic data: the deterministic solver always recovers the path.
        self.assertEqual(metrics["solver_em"], 1.0)
        self.assertEqual(len(metrics["stepwise_marginal_accuracy"]), config.L)
        self.assertEqual(len(metrics["stepwise_conditional_accuracy"]), config.L)
        # Random baseline checks.
        self.assertAlmostEqual(metrics["random_choice_step_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["random_choice_em_baseline"], 0.5**4)

    def test_edge_reorder_eval_runs_and_returns_same_shape(self):
        config = DagConfig(V=64, L=3, d=2, W=4)
        model, collator = self._model_and_collator(config.V)
        dataset = DagRecordDataset(config, "test", size=12, seed=20260602)
        normal = evaluate_dag_model(
            model, dataset, collator, 4, torch.device("cpu"), torch.float32
        )
        reordered = evaluate_dag_model(
            model,
            dataset,
            collator,
            4,
            torch.device("cpu"),
            torch.float32,
            reorder_edges_seed=7,
        )
        self.assertEqual(reordered["samples"], normal["samples"])
        self.assertIn("em", reordered)


class DagTrainingTests(unittest.TestCase):
    def test_dag_train_parser_accepts_depth_and_units_arguments(self):
        from phase_c.experiments.e04_dag_reasoning.train import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "--model",
                "debug",
                "--V",
                "64",
                "--L",
                "3",
                "--d",
                "2",
                "--W",
                "5",
                "--dataset-root",
                "phase_c_dag_data/x",
                "--train-units",
                "10",
                "--test-units",
                "5",
                "--max-steps",
                "1",
            ]
        )
        self.assertEqual(args.L, 3)
        self.assertEqual(args.d, 2)
        self.assertEqual(args.W, 5)
        self.assertEqual(args.train_units, 10)
        self.assertEqual(args.test_units, 5)

    def test_dag_train_cpu_smoke_one_step(self):
        from phase_c.experiments.e04_dag_reasoning.train import build_parser, run

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "run"
            args = build_parser().parse_args(
                [
                    "--model",
                    "debug",
                    "--V",
                    "64",
                    "--L",
                    "3",
                    "--d",
                    "2",
                    "--train-size",
                    "20",
                    "--test-size",
                    "10",
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                    "--micro-batch-size",
                    "1",
                    "--gradient-accumulation",
                    "1",
                    "--max-steps",
                    "1",
                    "--eval-size",
                    "10",
                    "--eval-batch-size",
                    "2",
                    "--eval-interval",
                    "0",
                    "--output-dir",
                    str(output_dir),
                ]
            )
            metrics = run(args)
            self.assertIsNotNone(metrics)
            self.assertEqual(metrics["completed_steps"], 1)
            self.assertEqual(metrics["train"]["samples"], 10)
            self.assertEqual(metrics["test"]["samples"], 10)
            self.assertIn("lambda", metrics["test"])
            self.assertIn("em", metrics["test"])
            self.assertTrue((output_dir / "final_metrics.json").exists())
            self.assertTrue((output_dir / "run_config.json").exists())
            self.assertTrue((output_dir / "train_log.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
