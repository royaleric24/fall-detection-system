"""Focused streaming equivalence, MQTT schema and actual-checkpoint buffer checks."""

import copy
import json
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from backend.app.inference import InferenceEngine, load_selected_model
from common.mqtt import decode_pose, encode, validate_pose
from ml.datasets.extract_pose_video import SOURCE_ROOT, source_metadata
from ml.datasets.pose_raw import load_validated
from ml.preprocessing.contract import select_sources
from ml.preprocessing.pose_sequence import PreprocessingConfig, StreamingPreprocessor, normalize_pose_sequence


def control(kind="sequence_start", source="edge-01", sequence="synthetic"):
    return dict(schema_version=1, message_type=kind, source_id=source, sequence_id=sequence)


def frame(index, source="edge-01", sequence="synthetic"):
    return dict(control("frame", source, sequence), frame_index=index, timestamp_ms=index*50,
                timestamp_kind="clip_relative", fps=20., features=[.25]*132, pose_detected=True)


class DummyModel:
    def __call__(self, features, lengths):
        assert features.shape == (1, int(lengths[0]), 132)
        return torch.tensor([0.0])


class Stage6Tests(unittest.TestCase):
    def test_stream_matches_offline_leading_short_long_gaps(self):
        from tests.test_stage33_pose_sequence import pose
        raw = pose(20)
        detected = np.array([False]*2 + [True]*2 + [False]*8 + [True]*3 + [False]*5)
        raw[~detected] = np.nan
        original = raw.copy()
        config = PreprocessingConfig(normalize_pose=False)
        expected, masks = normalize_pose_sequence(raw, detected, config=config)
        stream = StreamingPreprocessor(config)
        actual = [stream.step(lm, bool(present)) for lm, present in zip(raw, detected)]
        np.testing.assert_array_equal(np.stack([r[0] for r in actual]), expected)
        np.testing.assert_array_equal(np.stack([r[1] for r in actual]), masks)
        np.testing.assert_array_equal(raw, original)
        self.assertTrue((expected[9:12] == 0).all())

    def test_stream_matches_authorized_raw_npz_and_does_not_modify(self):
        source = select_sources()[0]
        before = source.source_npz.read_bytes()
        arrays = load_validated(source.source_npz, 20., 125)
        config = PreprocessingConfig(normalize_pose=False)
        expected, _ = normalize_pose_sequence(arrays['landmarks'], arrays['pose_detected'], config=config)
        stream = StreamingPreprocessor(config)
        actual = np.stack([stream.step(lm, bool(d))[0] for lm,d in zip(arrays['landmarks'], arrays['pose_detected'])])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        self.assertEqual(source.source_npz.read_bytes(), before)
        self.assertNotIn(source.subject_id, (6,7))

    def test_mqtt_roundtrip(self):
        message = frame(123)
        self.assertEqual(decode_pose(encode(message).encode(), 'fall/pose/edge-01'), message)
        self.assertEqual(len(message['features']), 132)

    def test_malformed_rejected_and_engine_still_works(self):
        engine = InferenceEngine(DummyModel())
        engine.handle(control())
        bad = []
        for features in ([0.]*131, [float('nan')]*132, [float('inf')]*132, [True]*132, [1e100]*132):
            bad.append(dict(frame(0), features=features))
        bad.extend([dict(frame(0), timestamp_ms=-1),dict(frame(0), fps=0),dict(frame(0), schema_version=True)])
        for message in bad:
            with self.assertRaises(ValueError):
                validate_pose(message)
        with self.assertRaises(ValueError):
            decode_pose(b'{"broken":', 'fall/pose/edge-01')
        with self.assertRaises(ValueError):
            decode_pose(encode(frame(0)).encode(), 'fall/pose/other')
        engine.handle(validate_pose(frame(0)))
        self.assertEqual(engine.frames_received,1)

    def test_minimum_stride_maximum_and_prediction_fields(self):
        engine = InferenceEngine(DummyModel())
        engine.handle(control())
        results=[]
        for i in range(250):
            prediction = engine.handle(frame(i))
            self.assertLessEqual(len(engine.sources['edge-01'].features),200)
            if i<85:
                self.assertIsNone(prediction)
            if prediction:
                results.append(prediction)
        self.assertEqual([r['frame_index'] for r in results], list(range(85,250,10)))
        self.assertEqual(results[0]['buffer_length'],86)
        self.assertEqual(results[-1]['buffer_length'],200)
        self.assertEqual(results[0]['source_id'],'edge-01')
        self.assertEqual(results[0]['timestamp_ms'],4250)
        self.assertEqual(results[0]['fall_probability'],.5)
        self.assertEqual(results[0]['predicted_label'],'fall')

    def test_source_sequence_isolation_duplicates_and_gaps(self):
        engine=InferenceEngine(DummyModel())
        engine.handle(control())
        engine.handle(control(source='edge-02'))
        for i in range(20):
            engine.handle(frame(i))
        engine.handle(control())  # Duplicate QoS1 start preserves existing frames.
        self.assertEqual(len(engine.sources['edge-01'].features),20)
        engine.handle(frame(19))
        self.assertEqual(engine.ignored,1)
        self.assertEqual(len(engine.sources['edge-02'].features),0)
        engine.handle(frame(21))
        self.assertEqual(engine.gap_resets,1)
        self.assertEqual(len(engine.sources['edge-01'].features),1)
        engine.handle(control(sequence='next'))
        self.assertEqual(len(engine.sources['edge-01'].features),0)
        engine.handle(frame(22))  # Old sequence cannot contaminate new one.
        self.assertEqual(engine.ignored,2)
        engine.handle(control('sequence_end', sequence='next'))
        self.assertNotIn('edge-01',engine.sources)

    def test_selected_checkpoint_and_real_finite_inference(self):
        model, config=load_selected_model()
        self.assertFalse(config['preprocessing']['normalize_pose'])
        engine=InferenceEngine(model)
        engine.handle(control())
        prediction=None
        for i in range(86):
            prediction=engine.handle(frame(i))
        self.assertTrue(math.isfinite(prediction['fall_probability']))
        self.assertTrue(0<=prediction['fall_probability']<=1)
        self.assertEqual(prediction['frame_index'],85)

    def test_checkpoint_missing_or_wrong_fails_without_substitution(self):
        with patch.dict('os.environ',CHECKPOINT_PATH='/private/tmp/does-not-exist-stage6.pt'):
            with self.assertRaises(FileNotFoundError):
                load_selected_model()
        from ml.preprocessing.contract import REPO
        with patch.dict('os.environ',CHECKPOINT_PATH=str(REPO/'ml/checkpoints/stage4_gru_best.pt')):
            with self.assertRaisesRegex(ValueError,'SHA256'):
                load_selected_model()

    def test_sealed_avi_rejected_before_file_open(self):
        with patch.object(Path,'open',side_effect=AssertionError('Unexpected access')):
            for subject in (6,7):
                with self.assertRaisesRegex(ValueError,'Held-out'):
                    source_metadata(SOURCE_ROOT / f'Subject.{subject}' / 'Fall forward' / 'anything.avi')


if __name__=='__main__':
    unittest.main()
