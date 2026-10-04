"""Synthetic transforms/access tests plus two explicitly authorized clip smoke tests."""

import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ml.datasets.fall_sequence_dataset import (FallSequenceDataset, binary_label,
                                               collate_sequences, validate_sources)
from ml.preprocessing.contract import ContractError, select_sources
from ml.preprocessing.pose_sequence import (FEATURE_DIM, PreprocessingConfig,
                                            normalize_pose_sequence)


def pose(frames=3):
    raw = np.zeros((frames, 33, 4), dtype=np.float32)
    raw[:, :, :2] = [0.5, 0.5]
    raw[:, :, 3] = 0.2  # Low visibility remains observed.
    raw[:, 23, :2] = [0.4, 0.6]
    raw[:, 24, :2] = [0.6, 0.6]
    raw[:, 11, :2] = [0.4, 0.3]
    raw[:, 12, :2] = [0.6, 0.3]
    return raw


def normalize(raw, detected=None):
    return normalize_pose_sequence(raw, np.ones(len(raw), bool) if detected is None else detected,
                                   image_aspect=1.5)


def sample(frames):
    features, mask = normalize(pose(frames))
    return dict(features=features, mask=mask, sequence_length=frames, label=1,
                clip_id=f"synthetic-{frames}", subject_id=1)


class Stage33Tests(unittest.TestCase):
    def test_hip_center_and_torso_scale(self):
        raw = pose()
        features, mask = normalize(raw)
        joints = features.reshape(3, 33, 4)
        np.testing.assert_allclose(joints[:, [23, 24], :2].mean(axis=1), 0, atol=1e-6)
        np.testing.assert_allclose(joints[:, [11, 12], :2].mean(axis=1),
                                   np.tile([0, -1], (3, 1)), atol=1e-6)
        np.testing.assert_allclose(joints[:, 24, 0], 0.5, atol=1e-6)
        np.testing.assert_array_equal(joints[:, :, 2], raw[:, :, 3])
        self.assertTrue(mask.all())

    def test_translation_scale_invariance_and_determinism(self):
        raw = pose()
        before = raw.copy()
        expected, _ = normalize(raw)
        changed = raw.copy()
        changed[:, :, :2] = changed[:, :, :2] * 2 + 0.1
        np.testing.assert_allclose(normalize(changed)[0], expected, atol=1e-6)
        np.testing.assert_array_equal(normalize(raw)[0], expected)
        np.testing.assert_array_equal(raw, before)

    def test_missing_frames_not_interpolated(self):
        raw = pose(5)
        raw[[0, 2, 4]] = np.nan
        detected = np.array([False, True, False, True, False])
        features, mask = normalize(raw, detected)
        self.assertTrue((features[[0, 2, 4]] == 0).all())
        self.assertFalse(mask[[0, 2, 4]].any())
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(features.shape, (5, FEATURE_DIM))

    def test_single_hip_fallback_and_no_hip(self):
        raw = pose()
        raw[:, 23] = np.nan
        features, mask = normalize(raw)
        self.assertTrue(mask[:, 24].all())
        self.assertFalse(mask[:, 23].any())
        np.testing.assert_array_equal(features.reshape(3, 33, 4)[:, 24, :2], 0)
        raw[:, 24] = np.nan
        self.assertFalse(normalize(raw)[1].any())

    def test_single_shoulder_and_no_shoulder(self):
        raw = pose()
        raw[:, 11] = np.nan
        self.assertTrue(normalize(raw)[1][:, 12].all())
        raw[:, 12] = np.nan
        self.assertFalse(normalize(raw)[1].any())

    def test_zero_tiny_and_invalid_scale(self):
        for delta in (0, 0.0001):
            raw = pose()
            raw[:, [11, 12], 1] = 0.6 + delta
            features, mask = normalize(raw)
            self.assertFalse(mask.any())
            self.assertTrue((features == 0).all())
        for floor in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                PreprocessingConfig(floor)

    def test_partial_invalid_joint_and_no_pose(self):
        raw = pose()
        raw[:, 0, 0] = np.inf
        features, mask = normalize(raw)
        self.assertFalse(mask[:, 0].any())
        self.assertTrue(np.isfinite(features).all())
        self.assertFalse(normalize(raw, np.zeros(3, bool))[1].any())

    def test_padding_lengths_and_feature_masks(self):
        batch = collate_sequences([sample(2), sample(5)])
        self.assertEqual(batch["features"].shape, (2, 5, FEATURE_DIM))
        np.testing.assert_array_equal(batch["lengths"], [2, 5])
        self.assertTrue((batch["features"][0, 2:] == 0).all())
        self.assertFalse(batch["mask"][0, 2:].any())
        self.assertEqual(batch["features"].dtype, np.float32)
        with self.assertRaises(ValueError):
            collate_sequences([])
        bad = sample(2)
        bad["sequence_length"] = 3
        with self.assertRaises(ValueError):
            collate_sequences([bad])

    def test_labels(self):
        self.assertEqual(binary_label("fall"), 1)
        self.assertEqual(binary_label("non_fall"), 0)
        with self.assertRaises(ValueError):
            binary_label("grounded")

    def test_split_integrity(self):
        sources = select_sources(("train", "validation"))
        validate_sources(sources)
        self.assertEqual(len(sources), 80)
        self.assertFalse({s.subject_id for s in sources if s.split == "train"} &
                         {s.subject_id for s in sources if s.split == "validation"})
        self.assertFalse({6, 7} & {s.subject_id for s in sources})
        with self.assertRaises(ValueError):
            validate_sources([sources[0], sources[0]])
        with self.assertRaises(ContractError):
            validate_sources([replace(sources[0], split="validation")])

    def test_test_excluded_before_io_including_disguised_subjects(self):
        source = select_sources()[0]
        with patch.object(Path, "open", side_effect=AssertionError("Forbidden I/O")):
            with self.assertRaises(ContractError):
                FallSequenceDataset("test")
            for subject in (6, 7):
                with self.assertRaises(ContractError):
                    validate_sources([replace(source, subject_id=subject)])

    def test_authorized_loader_smoke_guards_every_file_open(self):
        original = Path.open
        opened = []

        def guarded(path, *args, **kwargs):
            self.assertNotIn("Subject.6", path.parts)
            self.assertNotIn("Subject.7", path.parts)
            self.assertNotEqual(path.suffix, ".avi")
            opened.append(path)
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", guarded):
            samples = [FallSequenceDataset(split)[0] for split in ("train", "validation")]
        batch = collate_sequences(samples)
        self.assertTrue(np.isfinite(batch["features"]).all())
        self.assertEqual(len([p for p in opened if p.suffix == ".npz"]), 80)
        self.assertEqual([s["label"] for s in samples], [1, 1])

    def test_all_invalid_clip_excluded_deterministically(self):
        def fake_load(dataset, source):
            result = sample(2)
            if source == dataset.input_sources[0]:
                result["features"][:] = 0
                result["mask"][:] = False
            result["missing_pose_frames"] = 2 if not result["mask"].any() else 0
            return result
        with patch.object(FallSequenceDataset, "_load_sample", fake_load):
            dataset = FallSequenceDataset()
        self.assertEqual(len(dataset), 59)
        self.assertEqual(len(dataset.excluded), 1)
        self.assertEqual(dataset.excluded[0]["reason"], "no_valid_normalized_pose")
        modified = dataset[0]
        modified["features"][:] = 100
        self.assertFalse((dataset[0]["features"] == 100).all())

    def test_bad_transform_inputs(self):
        with self.assertRaises(ValueError):
            normalize_pose_sequence(pose(), np.ones(3, bool), image_aspect=0)
        with self.assertRaises(ValueError):
            normalize(pose(0))
        with self.assertRaises(ValueError):
            normalize(pose(), np.ones(3, np.int32))


if __name__ == "__main__":
    unittest.main()
