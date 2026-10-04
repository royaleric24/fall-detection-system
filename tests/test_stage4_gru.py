"""High-value GRU padding, gradients, metrics, checkpoint and access tests."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.datasets.fall_sequence_dataset import collate_sequences
from ml.models.gru import FallGRU, load_checkpoint
from ml.preprocessing.contract import REPO
from ml.training.train_gru import binary_metrics, evaluate, load_datasets, set_seed


class Stage4Tests(unittest.TestCase):
    def setUp(self):
        set_seed(42)
        self.model = FallGRU()
        self.features = torch.randn(3, 9, 132)
        self.lengths = torch.tensor([4, 9, 2])  # Unsorted, true last observed indices.

    def test_logits_shape_and_backward(self):
        logits = self.model(self.features, self.lengths)
        self.assertEqual(tuple(logits.shape), (3,))
        loss = nn.BCEWithLogitsLoss()(logits, torch.tensor([1., 0., 1.]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in self.model.parameters()))
        self.assertGreater(float(self.model.gru.weight_ih_l0.grad.abs().sum()), 0)

    def test_padding_suffix_invariant_and_matches_individual_sequences(self):
        self.model.eval()
        changed = self.features.clone()
        for i, length in enumerate(self.lengths.tolist()):
            changed[i, length:] = 9999
        with torch.no_grad():
            expected = self.model(self.features, self.lengths)
            torch.testing.assert_close(expected, self.model(changed, self.lengths), rtol=0, atol=0)
            for i, length in enumerate(self.lengths.tolist()):
                single = self.model(self.features[i:i+1, :length], torch.tensor([length]))
                torch.testing.assert_close(single[0], expected[i], rtol=1e-5, atol=1e-6)

    def test_checkpoint_round_trip_same_logits(self):
        self.model.eval()
        checkpoint = dict(schema_version="fall_gru_v1", state_dict=self.model.state_dict(),
                          model_config=self.model.config, classification_threshold=0.5,
                          preprocessing=dict(normalize_pose=True, max_forward_fill_frames=5, epsilon=1e-6))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.pt"
            torch.save(checkpoint, path)
            loaded, metadata = load_checkpoint(path)
            with torch.no_grad():
                torch.testing.assert_close(loaded(self.features, self.lengths),
                                            self.model(self.features, self.lengths), rtol=0, atol=0)
            self.assertEqual(metadata["classification_threshold"], 0.5)
            self.assertFalse(loaded.training)

    def test_metrics_known_confusion_and_undefined_values(self):
        metrics = binary_metrics([0, 0, 1, 1, 1], [0, 1, 1, 1, 0])
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 2]])
        self.assertEqual(metrics["accuracy"], 3/5)
        self.assertEqual(metrics["precision"], 2/3)
        self.assertEqual(metrics["recall"], 2/3)
        self.assertEqual(metrics["f1"], 2/3)
        self.assertEqual(metrics["positive_class"], "fall")
        collapsed = binary_metrics([0, 1], [0, 0])
        self.assertEqual((collapsed["precision"], collapsed["recall"], collapsed["f1"]), (0, 0, 0))

    def test_evaluation_threshold_boundary_and_records(self):
        class FixedLogits(nn.Module):
            def forward(self, features, lengths):
                return torch.tensor([0.0, -1.0, 1.0])
        samples = [dict(features=self.features[i, :length], sequence_length=length,
                        label=label, mask=torch.ones(length, 33, dtype=torch.bool),
                        observed_pose=torch.ones(length, dtype=torch.bool), clip_id=str(i), subject_id=1)
                   for i, (length, label) in enumerate(zip(self.lengths.tolist(), [1, 0, 1]))]
        metrics, records = evaluate(FixedLogits(), DataLoader(samples, batch_size=3,
                                                             collate_fn=collate_sequences), 0.5)
        self.assertEqual([r["prediction"] for r in records], [1, 0, 1])
        self.assertEqual(records[0]["probability"], 0.5)
        self.assertEqual(metrics["f1"], 1)

    def test_training_path_actual_stage33_batch_and_no_test_access(self):
        original = Path.open
        opened = []
        def guarded(path, *args, **kwargs):
            self.assertNotIn("Subject.6", path.parts)
            self.assertNotIn("Subject.7", path.parts)
            self.assertNotEqual(path.suffix, ".avi")
            opened.append(path)
            return original(path, *args, **kwargs)
        config = json.loads((REPO / "configs/stage4_gru.json").read_text())
        with patch.object(Path, "open", guarded):
            train, validation = load_datasets(config)
        self.assertEqual((len(train), len(validation)), (59, 20))
        self.assertFalse({s.subject_id for s in train.sources} & {s.subject_id for s in validation.sources})
        self.assertFalse({6, 7} & {s.subject_id for ds in (train, validation) for s in ds.sources})
        self.assertEqual(len([p for p in opened if p.suffix == ".npz"]), 80)
        batch = next(iter(DataLoader(train, batch_size=8, collate_fn=collate_sequences)))
        logits = self.model(batch["features"], batch["lengths"])
        self.assertEqual(tuple(logits.shape), (8,))
        self.assertTrue(torch.isfinite(logits).all())

    def test_invalid_lengths_rejected(self):
        for lengths in (torch.tensor([0, 9, 2]), torch.tensor([10, 9, 2]),
                        self.lengths.float()):
            with self.assertRaises(ValueError):
                self.model(self.features, lengths)


if __name__ == "__main__":
    unittest.main()
