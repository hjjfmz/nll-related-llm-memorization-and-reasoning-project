import math
import tempfile
import unittest
from pathlib import Path

import torch


class ModelAndTrainingTests(unittest.TestCase):
    def test_sample_stream_shards_ids_by_rank_without_overlap(self):
        from phase_c_training import SampleStream

        stream = SampleStream(12, 7)
        rank0 = stream.next_ids_for_rank(count=2, rank=0, world_size=2)
        rank1 = stream.next_ids_for_rank(count=2, rank=1, world_size=2)
        self.assertEqual(len(rank0), 2)
        self.assertEqual(len(rank1), 2)
        self.assertTrue(set(rank0).isdisjoint(rank1))
        self.assertEqual(stream.position, 8)

        stream = SampleStream(5, 11)
        first_rank0 = stream.next_ids_for_rank(count=1, rank=0, world_size=2)
        first_rank1 = stream.next_ids_for_rank(count=1, rank=1, world_size=2)
        second_rank0 = stream.next_ids_for_rank(count=1, rank=0, world_size=2)
        second_rank1 = stream.next_ids_for_rank(count=1, rank=1, world_size=2)
        self.assertTrue(set(first_rank0).isdisjoint(first_rank1))
        self.assertTrue(set(second_rank0).isdisjoint(second_rank1))

    def test_sample_stream_rejects_invalid_rank_configuration(self):
        from phase_c_training import SampleStream

        stream = SampleStream(10, 5)
        with self.assertRaises(ValueError):
            stream.next_ids_for_rank(count=1, rank=-1, world_size=2)
        with self.assertRaises(ValueError):
            stream.next_ids_for_rank(count=1, rank=2, world_size=2)
        with self.assertRaises(ValueError):
            stream.next_ids_for_rank(count=1, rank=0, world_size=0)

    def test_120m_parameter_scale_and_tied_weights(self):
        from phase_c_data import total_vocab_size
        from phase_c_model import MODEL_PRESETS, DecoderOnlyTransformer, count_parameters

        model = DecoderOnlyTransformer(
            MODEL_PRESETS["120m"], total_vocab_size(1024)
        )
        counts = count_parameters(model)
        self.assertGreaterEqual(counts["non_embedding"], 110_000_000)
        self.assertLessEqual(counts["non_embedding"], 125_000_000)
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)

    def test_answer_only_labels_and_capacity_formula(self):
        from phase_c_data import RandomConfig, generate_random_record, special_tokens
        from phase_c_training import AnswerOnlyCollator, random_capacity_metrics

        record = generate_random_record(RandomConfig(V=32, S=6, q=4), "train", 0, 9)
        batch = AnswerOnlyCollator(special_tokens(32)["PAD"])([record])
        positions = torch.nonzero(batch["labels"][0] != -100).flatten().tolist()
        start = record["metadata"]["answer_start"]
        end = record["metadata"]["answer_end"]
        self.assertEqual(positions, list(range(start - 1, end - 1)))

        metrics = random_capacity_metrics(
            total_nll_nats=100 * math.log(2),
            supervised_tokens=20,
            num_samples=5,
            H_R_bits_per_sample=32,
            non_embedding_parameters=10,
        )
        self.assertAlmostEqual(metrics["memory_bits"], 60)
        self.assertAlmostEqual(metrics["bits_per_parameter"], 6)

    def test_checkpoint_round_trip(self):
        from phase_c_model import DecoderOnlyTransformer, ModelConfig
        from phase_c_training import SampleStream, load_checkpoint, save_checkpoint

        config = ModelConfig("test", 1, 16, 4, context_length=16)
        model = DecoderOnlyTransformer(config, 20)
        optimizer = torch.optim.AdamW(model.parameters())
        stream = SampleStream(20, 3)
        stream.next_ids(5)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, 7, stream, {"name": "test"})
            restored = DecoderOnlyTransformer(config, 20)
            restored_optimizer = torch.optim.AdamW(restored.parameters())
            restored_stream = SampleStream(20, 99)
            state = load_checkpoint(
                path, restored, restored_optimizer, restored_stream, "cpu"
            )
        self.assertEqual(state["step"], 7)
        self.assertEqual(restored_stream.state_dict(), stream.state_dict())

    def test_load_checkpoint_accepts_list_rng_state(self):
        from phase_c_model import DecoderOnlyTransformer, ModelConfig
        from phase_c_training import SampleStream, load_checkpoint, save_checkpoint

        config = ModelConfig("test", 1, 16, 4, context_length=16)
        model = DecoderOnlyTransformer(config, 20)
        optimizer = torch.optim.AdamW(model.parameters())
        stream = SampleStream(20, 3)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, 7, stream, {"name": "test"})
            state = torch.load(path, map_location="cpu", weights_only=False)
            state["torch_rng_state"] = state["torch_rng_state"].tolist()
            torch.save(state, path)

            restored = DecoderOnlyTransformer(config, 20)
            restored_optimizer = torch.optim.AdamW(restored.parameters())
            restored_stream = SampleStream(20, 99)
            loaded = load_checkpoint(
                path, restored, restored_optimizer, restored_stream, "cpu"
            )

        self.assertEqual(loaded["step"], 7)
        self.assertEqual(restored_stream.state_dict(), stream.state_dict())


if __name__ == "__main__":
    unittest.main()
