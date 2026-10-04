"""Focused Stage 3.3 transform, causal gap, split and real DataLoader checks."""

import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from ml.datasets.fall_sequence_dataset import (FallSequenceDataset, binary_label,
                                               collate_sequences, validate_sources)
from ml.preprocessing.contract import ContractError, select_sources
from ml.preprocessing.pose_sequence import (FEATURE_DIM, PreprocessingConfig,
                                            handle_missing_pose, normalize_pose_sequence)


def pose(frames=3):
    raw = np.zeros((frames, 33, 4), dtype=np.float32)
    raw[:, :, :3] = [0.5, 0.5, 0.1]
    raw[:, :, 3] = 0.2  # Low visibility remains observed.
    raw[:, 23, :3] = [0.4, 0.6, 0.1]
    raw[:, 24, :3] = [0.6, 0.6, 0.1]
    raw[:, 11, :3] = [0.4, 0.3, 0.1]
    raw[:, 12, :3] = [0.6, 0.3, 0.1]
    raw[:, 0, 2] = 0.4
    return raw


def normalize(raw, detected=None, config=PreprocessingConfig()):
    return normalize_pose_sequence(raw, np.ones(len(raw), bool) if detected is None else detected,
                                   config=config)


def sample(frames):
    features, mask = normalize(pose(frames))
    return dict(features=torch.from_numpy(features), mask=torch.from_numpy(mask),
                observed_pose=torch.ones(frames, dtype=torch.bool), sequence_length=frames,
                label=1, clip_id=f"synthetic-{frames}", subject_id=1,
                missing_pose_frames=0, forward_filled_frames=0)


class Stage33Tests(unittest.TestCase):
    def test_xyz_hip_center_torso_scale_visibility(self):
        raw = pose()
        features, mask = normalize(raw)
        joints = features.reshape(3, 33, 4)
        np.testing.assert_allclose(joints[:, [23, 24], :3].mean(axis=1), 0, atol=1e-6)
        np.testing.assert_allclose(joints[:, [11, 12], :3].mean(axis=1),
                                   np.tile([0, -1, 0], (3, 1)), atol=1e-6)
        np.testing.assert_allclose(joints[:, 24, 0], 1/3, atol=1e-6)
        np.testing.assert_allclose(joints[:, 0, 2], 1, atol=1e-6)
        np.testing.assert_array_equal(joints[:, :, 3], raw[:, :, 3])
        self.assertTrue(mask.all())
        self.assertEqual(features.shape, (3, 132))

    def test_translation_scale_invariance_determinism_and_source_immutable(self):
        raw = pose()
        before = raw.copy()
        expected, _ = normalize(raw)
        changed = raw.copy()
        changed[:, :, :3] = changed[:, :, :3] * 2 + 0.1
        np.testing.assert_allclose(normalize(changed)[0], expected, atol=1e-6)
        np.testing.assert_array_equal(normalize(raw)[0], expected)
        np.testing.assert_array_equal(raw, before)

    def test_causal_fill_leading_gap_five_frame_limit_and_restart(self):
        raw = pose(11)
        detected = np.array([False, False, True, False, False, False,
                             False, False, False, True, False])
        raw[~detected] = np.nan
        raw[9, :, 0] += 0.1
        before = raw.copy()
        filled, available, imputed = handle_missing_pose(raw, detected)
        np.testing.assert_array_equal(available,
                                      [False, False, True, True, True, True, True, True, False, True, True])
        np.testing.assert_array_equal(imputed,
                                      [False, False, False, True, True, True, True, True, False, False, True])
        for t in range(3, 8):
            np.testing.assert_array_equal(filled[t], raw[2])
        self.assertTrue((filled[[0, 1, 8]] == 0).all())
        np.testing.assert_array_equal(filled[10], raw[9])
        np.testing.assert_array_equal(raw, before)
        # Prefix results cannot depend on future frames, for handling and transform.
        prefix, _, _ = handle_missing_pose(raw[:9], detected[:9])
        np.testing.assert_array_equal(prefix, filled[:9])
        np.testing.assert_array_equal(normalize(raw[:9], detected[:9])[0],
                                      normalize(raw, detected)[0][:9])
        self.assertEqual(normalize(raw, detected)[0].shape[0], 11)

    def test_ablation_preserves_xyz_visibility_and_same_missing_policy(self):
        raw = pose(9)
        detected = np.ones(9, bool)
        detected[1:8] = False
        raw[~detected] = np.nan
        a, ma = normalize(raw, detected, PreprocessingConfig(normalize_pose=False))
        b, mb = normalize(raw, detected)
        np.testing.assert_array_equal(ma, mb)
        np.testing.assert_array_equal(a.reshape(9, 33, 4)[0], raw[0])
        for t in range(1, 6):
            np.testing.assert_array_equal(a[t], a[0])
            np.testing.assert_array_equal(b[t], b[0])
        self.assertTrue((a[6:8] == 0).all())
        np.testing.assert_array_equal(a.reshape(9, 33, 4)[:, :, 3],
                                      b.reshape(9, 33, 4)[:, :, 3])

    def test_reference_fallback_and_partial_invalid_joint(self):
        raw = pose()
        raw[:, 23] = np.nan
        raw[:, 0, 0] = np.inf
        features, mask = normalize(raw)
        self.assertTrue(mask[:, 24].all())
        self.assertFalse(mask[:, [23, 0]].any())
        np.testing.assert_array_equal(features.reshape(3, 33, 4)[:, 24, :3], 0)
        raw[:, 24] = np.nan
        self.assertFalse(normalize(raw)[1].any())
        raw = pose()
        raw[:, 11] = np.nan
        self.assertTrue(normalize(raw)[1][:, 12].all())
        raw[:, 12] = np.nan
        self.assertFalse(normalize(raw)[1].any())

    def test_epsilon_zero_scale_visibility_sanitation_and_all_missing(self):
        raw = pose()
        raw[:, [11, 12], :3] = [0.5, 0.6, 0.1]
        raw[:, 0, 3] = 2
        features, mask = normalize(raw)
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(mask.all())
        self.assertEqual(float(features.reshape(3, 33, 4)[0, 0, 3]), 1)
        np.testing.assert_allclose(features.reshape(3, 33, 4)[:, 24, 0], 0.1/1e-6, rtol=1e-6)
        raw[:] = np.nan
        features, mask = normalize(raw, np.zeros(3, bool))
        self.assertFalse(mask.any())
        self.assertTrue((features == 0).all())

    def test_dataloader_padding_lengths_and_pack_interface(self):
        loader = DataLoader([sample(2), sample(5)], batch_size=2, collate_fn=collate_sequences)
        batch = next(iter(loader))
        self.assertEqual(tuple(batch["features"].shape), (2, 5, FEATURE_DIM))
        torch.testing.assert_close(batch["lengths"], torch.tensor([2, 5]))
        self.assertTrue((batch["features"][0, 2:] == 0).all())
        self.assertFalse(batch["mask"][0, 2:].any())
        self.assertEqual(batch["features"].dtype, torch.float32)
        self.assertEqual(batch["labels"].dtype, torch.int64)
        packed = pack_padded_sequence(batch["features"], batch["lengths"],
                                     batch_first=True, enforce_sorted=False)
        self.assertEqual(tuple(packed.data.shape), (7, FEATURE_DIM))
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

    def test_authorized_loader_preserves_length_and_guards_file_access(self):
        original = Path.open
        opened = []
        def guarded(path, *args, **kwargs):
            self.assertNotIn("Subject.6", path.parts)
            self.assertNotIn("Subject.7", path.parts)
            self.assertNotEqual(path.suffix, ".avi")
            opened.append(path)
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guarded):
            datasets = [FallSequenceDataset(split) for split in ("train", "validation")]
        for ds in datasets:
            self.assertIsInstance(ds, Dataset)
            self.assertEqual(len(ds.manifest), len(ds))
            for s, row in zip(ds.samples, ds.manifest):
                self.assertEqual(s["sequence_length"], int(ds.metadata[row["sequence_id"]]["expected_frame_count"]))
                self.assertEqual(s["features"].shape[0], s["sequence_length"])
                self.assertTrue(torch.isfinite(s["features"]).all())
        self.assertEqual([len(ds) for ds in datasets], [59, 20])
        self.assertEqual(len([p for p in opened if p.suffix == ".npz"]), 80)
        batch = next(iter(DataLoader(datasets[0], batch_size=4, collate_fn=collate_sequences)))
        self.assertEqual(batch["features"].shape[0], 4)

    def test_all_missing_exclusion_independent_of_normalization_and_cache_safe(self):
        def fake_load(dataset, source):
            result = sample(2)
            if source == dataset.input_sources[0]:
                result["features"][:] = 0
                result["mask"][:] = False
                result["observed_pose"][:] = False
                result["missing_pose_frames"] = 2
            return result
        with patch.object(FallSequenceDataset, "_load_sample", fake_load):
            a = FallSequenceDataset(config=PreprocessingConfig(normalize_pose=False))
            b = FallSequenceDataset()
        self.assertEqual(len(a), 59)
        self.assertEqual(a.excluded, b.excluded)
        self.assertEqual(a.excluded[0]["reason"], "all_frames_pose_missing")
        modified = a[0]
        modified["features"][:] = 100
        self.assertFalse((a[0]["features"] == 100).all())

    def test_bad_inputs(self):
        for floor in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                PreprocessingConfig(epsilon=floor)
        for gap in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                PreprocessingConfig(max_forward_fill_frames=gap)
        with self.assertRaises(ValueError):
            normalize(pose(0))
        with self.assertRaises(ValueError):
            normalize(pose(), np.ones(3, np.int32))


if __name__ == "__main__":
    unittest.main()
