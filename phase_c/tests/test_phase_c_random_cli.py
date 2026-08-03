import tempfile
import json
import subprocess
import sys
import unittest
from pathlib import Path


class RandomCliConfigTests(unittest.TestCase):
    def test_resume_command_reuses_saved_arguments_without_training_overrides(self):
        from phase_c.experiments.e03_random_capacity.config import (
            RandomCommandError,
            load_saved_run_arguments,
            make_resume_namespace,
            write_run_arguments,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            write_run_arguments(
                run_dir,
                {
                    "model": "L4_H128",
                    "dataset_root": "phase_c_random_data/V2048_S64_seed20260715",
                    "train_units": 10,
                    "test_units": 5,
                    "epochs": 300,
                    "max_steps": 9000,
                    "learning_rate": 3e-4,
                    "minimum_learning_rate": 3e-5,
                    "micro_batch_size": 1,
                    "gradient_accumulation": 32,
                    "output_dir": str(run_dir),
                },
            )

            args = make_resume_namespace(run_dir)
            saved_minimum_lr = load_saved_run_arguments(run_dir)[
                "minimum_learning_rate"
            ]

        self.assertEqual(args.model, "L4_H128")
        self.assertEqual(args.epochs, 300)
        self.assertEqual(args.run_mode, "resume")
        self.assertEqual(args.output_dir, run_dir)
        self.assertEqual(args.resume, run_dir / "checkpoint_latest.pt")
        self.assertEqual(saved_minimum_lr, 3e-5)
        with self.assertRaises(RandomCommandError):
            make_resume_namespace(run_dir, overrides={"epochs": 400})

    def test_extend_command_adds_epochs_and_uses_constant_min_lr(self):
        from phase_c.experiments.e03_random_capacity.config import (
            make_extend_namespace,
            write_run_arguments,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            write_run_arguments(
                run_dir,
                {
                    "model": "L4_H128",
                    "dataset_root": "phase_c_random_data/V2048_S64_seed20260715",
                    "train_units": 10,
                    "test_units": 5,
                    "epochs": 300,
                    "max_steps": 9000,
                    "learning_rate": 3e-4,
                    "minimum_learning_rate": 3e-5,
                    "micro_batch_size": 1,
                    "gradient_accumulation": 32,
                    "output_dir": str(run_dir),
                },
            )

            args = make_extend_namespace(
                run_dir,
                extra_epochs=100,
                lr_policy="constant-min",
                output_dir=run_dir / "extend_001",
            )

        self.assertEqual(args.run_mode, "extend")
        self.assertEqual(args.epochs, 400)
        self.assertIsNone(args.max_steps)
        self.assertEqual(args.learning_rate, 3e-5)
        self.assertEqual(args.minimum_learning_rate, 3e-5)
        self.assertEqual(args.warmup_steps, 0)
        self.assertEqual(args.resume, run_dir / "checkpoint_latest.pt")
        self.assertEqual(args.output_dir, run_dir / "extend_001")

    def test_extend_command_adds_steps_for_a_paper_style_run(self):
        from phase_c.experiments.e03_random_capacity.config import (
            make_extend_namespace,
            write_run_arguments,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            write_run_arguments(
                run_dir,
                {
                    "model": "L4_H128",
                    "dataset_root": "phase_c_random_data/V2048_S64_seed20260715",
                    "train_units": 10,
                    "test_units": 5,
                    "epochs": None,
                    "max_steps": 1_000_000,
                    "resolved_max_steps": 1_000_000,
                    "learning_rate": 3e-4,
                    "minimum_learning_rate": 3e-4,
                    "output_dir": str(run_dir),
                },
            )

            args = make_extend_namespace(run_dir, extra_steps=250_000)

        self.assertEqual(args.max_steps, 1_250_000)
        self.assertIsNone(args.epochs)
        self.assertEqual(args.learning_rate, 3e-4)

    def test_eval_command_loads_checkpoint_without_training_steps(self):
        from phase_c.experiments.e03_random_capacity.config import (
            make_eval_namespace,
            write_run_arguments,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            write_run_arguments(
                run_dir,
                {
                    "model": "debug",
                    "V": 32,
                    "S": 6,
                    "train_size": 20,
                    "validation_size": 10,
                    "epochs": 20,
                    "output_dir": str(run_dir),
                },
            )

            args = make_eval_namespace(run_dir, output_dir=run_dir / "eval")

        self.assertEqual(args.run_mode, "eval")
        self.assertTrue(args.eval_only)
        self.assertEqual(args.max_steps, 0)
        self.assertEqual(args.resume, run_dir / "checkpoint_latest.pt")
        self.assertEqual(args.output_dir, run_dir / "eval")


class ModelPackageTests(unittest.TestCase):
    def test_model_package_exposes_presets_transformer_and_counts(self):
        from phase_c.models.config import ModelConfig
        from phase_c.models.counting import count_parameters
        from phase_c.models.presets import MODEL_PRESETS
        from phase_c.models.transformer import DecoderOnlyTransformer

        config = MODEL_PRESETS["L4_H128"]
        model = DecoderOnlyTransformer(config, vocab_size=1027)
        counts = count_parameters(model)

        self.assertIsInstance(config, ModelConfig)
        self.assertEqual(config.n_layers, 4)
        self.assertEqual(config.hidden_size, 128)
        self.assertEqual(counts["non_embedding"], 793344)


class RandomCliExecutionTests(unittest.TestCase):
    def test_paper_grid_script_covers_three_models_four_dataset_sizes_and_cleanup(self):
        root = Path(__file__).resolve().parents[2]
        script_path = root / "scripts" / "run_e03_random_capacity_grid.sh"

        self.assertTrue(script_path.exists())
        script = script_path.read_text(encoding="utf-8")
        for run_spec in (
            "L2_H128:1:725632",
            "L2_H128:3:725632",
            "L2_H128:5:725632",
            "L2_H128:10:725632",
            "L4_H128:1:1122176",
            "L4_H128:4:1122176",
            "L4_H128:7:1122176",
            "L4_H128:13:1122176",
            "L8_H256:5:6976256",
            "L8_H256:20:6976256",
            "L8_H256:40:6976256",
            "L8_H256:80:6976256",
        ):
            self.assertIn(run_spec, script)
        self.assertIn("trap cleanup INT TERM", script)
        self.assertIn("--eval-size 0", script)
        self.assertIn('TEST_UNITS="${TEST_UNITS:-5}"', script)
        self.assertIn('MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-32}"', script)
        self.assertIn('GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-32}"', script)
        self.assertIn('MAX_STEPS="${MAX_STEPS:-250000}"', script)

    def test_reporting_reads_current_test_metrics_and_total_parameter_capacity(self):
        from phase_c.experiments.e03_random_capacity.reporting import load_run

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "run_config.json").write_text(
                json.dumps({"arguments": {"V": 2048, "train_units": 2}}),
                encoding="utf-8",
            )
            (run_dir / "final_metrics.json").write_text(
                json.dumps(
                    {
                        "completed_steps": 1_000_000,
                        "train": {
                            "samples": 2_000,
                            "nll_bits_per_token": 2.0,
                            "memory_bits": 100.0,
                            "bits_per_parameter": 0.01,
                            "dataset_entropy_bits": 200.0,
                        },
                        "test": {"samples": 20, "nll_bits_per_token": 11.0},
                        "parameters": {"total": 10_000, "non_embedding": 8_000},
                    }
                ),
                encoding="utf-8",
            )

            summary = load_run("paper", run_dir)

        self.assertEqual(summary.test_nll_bits_per_token, 11.0)
        self.assertEqual(summary.total_parameters, 10_000)
        self.assertEqual(summary.entropy_ceiling_bits_per_parameter, 0.02)

    def test_random_unit_writer_is_available_from_data_package(self):
        from phase_c.data.random_units import write_random_units

        self.assertIsNotNone(write_random_units)

    def test_module_cli_inspect_prints_parameter_counts(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "phase_c.cli",
                "random",
                "inspect",
                "--model",
                "L4_H128",
                "--V",
                "1024",
            ],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["model"]["name"], "L4_H128")
        self.assertEqual(payload["parameters"]["non_embedding"], 793344)

    def test_module_cli_random_gen_data_writes_units(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "phase_c.cli",
                    "random",
                    "gen-data",
                    "--V",
                    "32",
                    "--S",
                    "6",
                    "--unit-size",
                    "2",
                    "--train-units",
                    "1",
                    "--test-units",
                    "1",
                    "--base-seed",
                    "1234",
                    "--output-dir",
                    temp_dir,
                ],
                cwd=Path(__file__).resolve().parents[2],
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((Path(temp_dir) / "train" / "1.jsonl.gz").exists())
            self.assertTrue((Path(temp_dir) / "test" / "1.jsonl.gz").exists())

    def test_eval_namespace_generates_metrics_without_appending_train_log(self):
        from phase_c.experiments.e03_random_capacity.config import make_eval_namespace
        from phase_c.experiments.e03_random_capacity.train import build_parser, run

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "base"
            train_args = build_parser().parse_args(
                [
                    "--model",
                    "debug",
                    "--V",
                    "32",
                    "--S",
                    "6",
                    "--train-size",
                    "4",
                    "--validation-size",
                    "2",
                    "--max-steps",
                    "1",
                    "--eval-size",
                    "2",
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                    "--output-dir",
                    str(run_dir),
                ]
            )
            run(train_args)

            eval_dir = Path(temp_dir) / "eval"
            eval_args = make_eval_namespace(run_dir, output_dir=eval_dir)
            eval_args.device = "cpu"
            eval_args.dtype = "float32"
            eval_args.eval_size = 2
            metrics = run(eval_args)

            self.assertEqual(metrics["completed_steps"], 1)
            self.assertTrue((eval_dir / "final_metrics.json").exists())
            self.assertFalse((eval_dir / "train_log.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
