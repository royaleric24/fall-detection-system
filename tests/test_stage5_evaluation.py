"""Small tests for controlled normalization ablation and causal dropout evaluation."""

import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from torch.utils.data import DataLoader

from ml.datasets.fall_sequence_dataset import FallSequenceDataset, collate_sequences, source_fingerprint
from ml.evaluation.stage5 import (BASELINE_DIR, DROPOUT_SEED, check_clean_match, dropout_samples,
                                  frame_dropout, validate_ablation_config)
from ml.models.gru import load_checkpoint
from ml.preprocessing.pose_sequence import PreprocessingConfig
from ml.preprocessing.contract import ContractError, REPO, select_sources
from ml.training.train_gru import evaluate, load_datasets, set_seed


class Stage5Tests(unittest.TestCase):
    def test_dropout_deterministic_nested_and_does_not_mutate(self):
        arrays = dict(pose_detected=np.ones(100, bool),
                      landmarks=np.ones((100, 33, 4), np.float32),
                      frame_index=np.arange(100, dtype=np.int32),
                      timestamp_ms=np.arange(100, dtype=np.int64) * 50)
        original = {k: v.copy() for k, v in arrays.items()}
        a, indices = frame_dropout(arrays, "synthetic", 0.1, 42)
        b, again = frame_dropout(arrays, "synthetic", 0.1, 42)
        _, more = frame_dropout(arrays, "synthetic", 0.2, 42)
        np.testing.assert_array_equal(indices, again)
        self.assertEqual(len(indices), 10)
        self.assertTrue(set(indices) <= set(more))
        self.assertTrue(np.isnan(a["landmarks"][indices]).all())
        self.assertFalse(a["pose_detected"][indices].any())
        for key in arrays:
            np.testing.assert_array_equal(a[key], b[key])
            np.testing.assert_array_equal(arrays[key], original[key])
        clean, zero = frame_dropout(arrays, "synthetic", 0, 42)
        self.assertEqual(len(zero), 0)
        for key in arrays:
            np.testing.assert_array_equal(clean[key], arrays[key])

    def test_ablation_changes_only_normalization(self):
        baseline = json.loads((REPO / "configs/stage4_gru.json").read_text())
        ablation = json.loads((REPO / "configs/stage5_no_normalization.json").read_text())
        validate_ablation_config(baseline, ablation)
        altered = copy.deepcopy(ablation)
        altered["hidden_size"] = 32
        with self.assertRaises(ValueError):
            validate_ablation_config(baseline, altered)
        with self.assertRaises(ValueError):
            validate_ablation_config(baseline, baseline)
        with self.assertRaises(ValueError):
            load_datasets(ablation)  # Stage 4 default still rejects ablation.
        # Verify the actual dataset factory receives normalize_pose=False twice.
        with patch("ml.training.train_gru.FallSequenceDataset") as factory:
            factory.return_value.__len__.return_value = 1
            factory.return_value.input_sources = ()
            load_datasets(ablation, normalization_ablation=True)
            self.assertEqual([call.args[0] for call in factory.call_args_list], ["train", "validation"])
            self.assertTrue(all(call.kwargs["config"].normalize_pose is False
                                for call in factory.call_args_list))

    def test_heldout_or_train_rejected_before_pose_io(self):
        source = select_sources(("validation",))[0]
        for subject, split in ((6, "test"), (7, "validation"), (1, "train")):
            fake = SimpleNamespace(input_sources=(replace(source, subject_id=subject, split=split),))
            with patch.object(Path, "open", side_effect=AssertionError("Unexpected I/O")):
                with self.assertRaises((ContractError, ValueError)):
                    dropout_samples(fake, 0.1, 42)

    def test_zero_dropout_exact_baseline_and_raw_files_unchanged(self):
        set_seed(42)
        original_open = Path.open
        def guarded(path, *args, **kwargs):
            self.assertNotIn("Subject.6", path.parts)
            self.assertNotIn("Subject.7", path.parts)
            self.assertNotEqual(path.suffix, ".avi")
            if path.suffix == ".npz":
                self.assertEqual(args[0] if args else kwargs.get("mode", "r"), "rb")
            return original_open(path, *args, **kwargs)
        with patch.object(Path, "open", guarded):
            baseline = json.loads((BASELINE_DIR / "summary.json").read_text())
            dataset = FallSequenceDataset("validation")
            before = source_fingerprint(dataset.input_sources)
            samples, stats = dropout_samples(dataset, 0, DROPOUT_SEED)
            model, _ = load_checkpoint(REPO / baseline["checkpoint"]["path"])
            metrics, records = evaluate(model, DataLoader(samples, batch_size=8,
                                                         collate_fn=collate_sequences), 0.5)
            check_clean_match(metrics, records, baseline, BASELINE_DIR / "validation_predictions.csv")
            self.assertEqual(source_fingerprint(dataset.input_sources), before)
        self.assertEqual(len(samples), 20)
        self.assertEqual(stats["newly_missing_frames"], 0)
        self.assertEqual(metrics["positive_class"], "fall")

    def test_dropout_keeps_labels_lengths_and_uses_causal_fill(self):
        # Controlled frame indices: one real pose followed by 6 induced failures.
        dataset = SimpleNamespace(
            input_sources=select_sources(("validation",))[:1],
            config=PreprocessingConfig())
        dataset.sources = dataset.input_sources
        source = dataset.sources[0]
        dataset.metadata = {source.source_video: {"source_fps": "20", "expected_frame_count": "7"}}
        from tests.test_stage33_pose_sequence import pose
        arrays = dict(landmarks=pose(7), pose_detected=np.ones(7, bool))
        corrupted = {k: v.copy() for k, v in arrays.items()}
        corrupted["pose_detected"][1:] = False
        corrupted["landmarks"][1:] = np.nan
        with patch("ml.evaluation.stage5.load_validated", return_value=arrays), \
             patch("ml.evaluation.stage5.frame_dropout", return_value=(corrupted, np.arange(1, 7))):
            samples, _ = dropout_samples(dataset, 0.2, 42)
        self.assertEqual(samples[0]["sequence_length"], 7)
        self.assertEqual(samples[0]["label"], int(source.video_label == "fall"))
        self.assertTrue((samples[0]["features"][1:6] == samples[0]["features"][0]).all())
        self.assertTrue((samples[0]["features"][6] == 0).all())


if __name__ == "__main__":
    unittest.main()
