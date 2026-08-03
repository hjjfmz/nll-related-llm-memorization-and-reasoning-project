import tempfile
import json
import subprocess
import sys
import unittest
from pathlib import Path


class RandomGenerationTests(unittest.TestCase):
    def test_random_record_is_a_paper_style_independent_token_sequence(self):
        from phase_c.data.core import RandomConfig, generate_random_record, special_tokens

        config = RandomConfig(V=32, S=6)
        train = generate_random_record(config, "train", sample_id=9, seed=101)

        tokens = special_tokens(32)
        self.assertEqual(train["input_ids"][0], tokens["BOS"])
        self.assertEqual(train["input_ids"][-1], tokens["EOS"])
        self.assertEqual(train["input_ids"][1:-1], train["target_ids"])
        self.assertNotIn("key", train["metadata"])
        start = train["metadata"]["answer_start"]
        end = train["metadata"]["answer_end"]
        self.assertEqual(train["input_ids"][start:end], train["target_ids"])
        self.assertEqual((start, end), (1, 7))
        self.assertEqual(train["metadata"]["H_R_bits"], 30.0)

    def test_random_defaults_match_the_paper_protocol(self):
        from phase_c.data.core import RandomConfig

        self.assertEqual(RandomConfig(), RandomConfig(V=2048, S=64))

    def test_random_generation_is_deterministic(self):
        from phase_c.data.core import RandomConfig, generate_random_record

        config = RandomConfig(V=64, S=12)
        first = generate_random_record(config, "test", 123, 999)
        second = generate_random_record(config, "test", 123, 999)
        self.assertEqual(first, second)


class DagGenerationTests(unittest.TestCase):
    def test_dag_satisfies_layer_path_and_outdegree_invariants(self):
        from phase_c.data.core import (
            DagConfig,
            count_paths,
            generate_dag_record,
            validate_record,
        )

        config = DagConfig(V=1024, L=4, d=3, W=5)
        record = generate_dag_record(config, "train", sample_id=4, seed=88)
        metadata = record["metadata"]
        edges = [tuple(edge) for edge in metadata["edges"]]
        layer_of = {
            node: layer_index
            for layer_index, layer in enumerate(metadata["layers"])
            for node in layer
        }

        self.assertEqual(len(edges), config.L * config.W * config.d)
        self.assertTrue(all(layer_of[v] == layer_of[u] + 1 for u, v in edges))
        self.assertEqual(
            count_paths(edges, metadata["source"], metadata["target"]), 1
        )
        self.assertEqual(record["target_ids"], metadata["path"][1:])

        outdegree = {}
        for source, _ in edges:
            outdegree[source] = outdegree.get(source, 0) + 1
        for layer in metadata["layers"][:-1]:
            self.assertTrue(all(outdegree[node] == config.d for node in layer))
        self.assertEqual(validate_record(record), [])

    def test_edge_reordering_keeps_the_same_answer(self):
        from phase_c.data.core import (
            DagConfig,
            generate_dag_record,
            parse_dag_sequence,
            reorder_dag_edges,
        )

        record = generate_dag_record(
            DagConfig(V=128, L=3, d=2, W=4),
            "test",
            sample_id=12,
            seed=44,
        )
        reordered = reorder_dag_edges(record, seed=55)
        original_parsed = parse_dag_sequence(record["input_ids"], V=128)
        reordered_parsed = parse_dag_sequence(reordered["input_ids"], V=128)

        self.assertEqual(original_parsed["source"], reordered_parsed["source"])
        self.assertEqual(original_parsed["target"], reordered_parsed["target"])
        self.assertEqual(original_parsed["answer"], reordered_parsed["answer"])
        self.assertEqual(
            set(original_parsed["edges"]), set(reordered_parsed["edges"])
        )


class AdmissionTests(unittest.TestCase):
    def test_streaming_admission_passes_valid_records(self):
        from phase_c.data.core import (
            DagConfig,
            RandomConfig,
            generate_dag_record,
            generate_random_record,
            run_admission_checks,
        )

        records = []
        random_config = RandomConfig(V=64, S=8)
        dag_config = DagConfig(V=128, L=3, d=2, W=4)
        for split_index, split in enumerate(("train", "validation", "test")):
            for sample_id in range(20):
                records.append(
                    generate_random_record(
                        random_config, split, sample_id, 100 + split_index
                    )
                )
                records.append(
                    generate_dag_record(
                        dag_config, split, sample_id, 200 + split_index
                    )
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            report = run_admission_checks(
                records, sqlite_path=Path(temp_dir) / "admission.sqlite"
            )

        self.assertTrue(report["all_checks_passed"])
        self.assertEqual(report["records_checked"], 120)
        self.assertEqual(report["duplicate_identity_count"], 0)

    def test_streaming_admission_detects_duplicate_record(self):
        from phase_c.data.core import (
            RandomConfig,
            generate_random_record,
            run_admission_checks,
        )

        record = generate_random_record(RandomConfig(V=32, S=6), "train", 1, 9)
        with tempfile.TemporaryDirectory() as temp_dir:
            report = run_admission_checks(
                [record, record], sqlite_path=Path(temp_dir) / "admission.sqlite"
            )

        self.assertFalse(report["all_checks_passed"])
        self.assertEqual(report["duplicate_identity_count"], 1)

    def test_gzip_shards_round_trip_without_loading_all_records(self):
        from phase_c.data.core import (
            RandomConfig,
            generate_records,
            read_jsonl_gzip,
            write_jsonl_gzip_shards,
        )

        records = generate_records(
            "random", RandomConfig(V=32, S=6), "train", 7, seed=4
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = write_jsonl_gzip_shards(
                records,
                output_dir=Path(temp_dir),
                prefix="random_train",
                shard_size=3,
            )
            rows = [
                row
                for file_info in manifest["files"]
                for row in read_jsonl_gzip(Path(temp_dir) / file_info["name"])
            ]

        self.assertEqual(manifest["records"], 7)
        self.assertEqual([item["records"] for item in manifest["files"]], [3, 3, 1])
        self.assertEqual([row["sample_id"] for row in rows], list(range(7)))


class RandomUnitDatasetTests(unittest.TestCase):
    def test_random_unit_paths_select_prefix_units(self):
        from phase_c.data.core import random_unit_paths

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train = root / "train"
            train.mkdir()
            for index in range(1, 6):
                (train / f"{index}.jsonl.gz").write_bytes(b"placeholder")

            paths = random_unit_paths(root, "train", units=3)

        self.assertEqual(
            [path.name for path in paths],
            ["1.jsonl.gz", "2.jsonl.gz", "3.jsonl.gz"],
        )

    def test_random_unit_paths_reject_missing_unit(self):
        from phase_c.data.core import random_unit_paths

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train = root / "train"
            train.mkdir()
            (train / "1.jsonl.gz").write_bytes(b"placeholder")

            with self.assertRaises(FileNotFoundError):
                random_unit_paths(root, "train", units=2)


class CliTests(unittest.TestCase):
    def test_preview_cli_prints_parseable_dag_record(self):
        command = [
            sys.executable,
            "-m",
                "phase_c.data.cli",
            "preview",
            "--family",
            "dag",
            "--V",
            "128",
            "--L",
            "2",
            "--d",
            "2",
            "--W",
            "4",
            "--sample-id",
            "3",
            "--seed",
            "7",
        ]
        result = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)
        self.assertEqual(record["family"], "dag")
        self.assertEqual(record["answer_length"], 2)

    def test_random_units_cli_writes_simple_train_and_test_folders(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            command = [
                sys.executable,
                "-m",
                "phase_c.data.cli",
                "random-units",
                "--V",
                "32",
                "--S",
                "6",
                "--unit-size",
                "3",
                "--train-units",
                "2",
                "--test-units",
                "1",
                "--base-seed",
                "1234",
                "--output-dir",
                temp_dir,
            ]
            result = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[2],
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            root = Path(temp_dir)
            self.assertTrue((root / "train" / "1.jsonl.gz").exists())
            self.assertTrue((root / "train" / "2.jsonl.gz").exists())
            self.assertTrue((root / "test" / "1.jsonl.gz").exists())
            self.assertTrue((root / "dataset_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
