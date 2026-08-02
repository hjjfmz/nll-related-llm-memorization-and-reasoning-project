from pathlib import Path
import unittest


class PackageStructureTests(unittest.TestCase):
    def test_root_directory_has_no_python_code_files(self):
        root = Path(__file__).resolve().parents[2]
        self.assertEqual([path.name for path in root.glob("*.py")], [])

    def test_phase_c_runtime_imports_do_not_depend_on_legacy_root_modules(self):
        root = Path(__file__).resolve().parents[2]
        forbidden = (
            "phase_c_data",
            "phase_c_model",
            "phase_c_training",
            "train_phase_c_random",
            "generate_phase_c_data",
        )
        offenders = []
        for path in (root / "phase_c").rglob("*.py"):
            if "tests" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            for name in forbidden:
                if f"from {name}" in text or f"import {name}" in text:
                    offenders.append(f"{path.relative_to(root)} imports {name}")
        self.assertEqual(offenders, [])

    def test_new_package_entrypoints_are_importable(self):
        from phase_c.data.cli import build_parser as build_data_parser
        from phase_c.experiments.e03_random_capacity.train import build_parser, run

        self.assertIsNotNone(build_data_parser)
        self.assertIsNotNone(build_parser)
        self.assertIsNotNone(run)


if __name__ == "__main__":
    unittest.main()
