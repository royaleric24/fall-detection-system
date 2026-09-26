"""Stage 3.2c tests use reviewed Train metadata and synthetic images only."""

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from ml.preprocessing import pilot_png_pose as p


def fixture():
    inventory = p.read_rows(p.SOURCE/"annotation_native_sequence_inventory.csv")
    sources = tuple(p.contract.PoseSource("CAUCAFall", "V5", p.contract.RUN_ID,
        r["source_video_identity"], Path("/unused.npz"), int(r["subject_id"]), r["activity"],
        "train", "fall" if r["activity"].startswith("Fall") else "non_fall") for r in inventory)
    return inventory, sources


def arrays(count=2, detected=None):
    detected = np.array(detected if detected is not None else [True] * count, dtype=np.bool_)
    values = np.ones((count,33,4), np.float32)
    values[~detected] = np.nan
    return dict(annotation_ordinal=np.arange(count,dtype=np.int32),
                pose_detected=detected, landmarks=values)


class Stage32cTests(unittest.TestCase):
    def test_pilot_selection_deterministic_under_input_reordering(self):
        inventory, sources = fixture()
        first = p.select_pilot(inventory, sources)
        second = p.select_pilot(list(reversed(inventory)), tuple(reversed(sources)))
        self.assertEqual(first, second)

    def test_exactly_ten_full_sequences(self):
        inventory, sources = fixture(); selected = p.select_pilot(inventory, sources)
        self.assertEqual(len(selected),10)
        self.assertEqual(len({r["source_video_identity"] for r in selected}),10)
        self.assertTrue(all(int(r["png_count"]) > 0 and r["png_count"]==r["txt_count"] for r in selected))

    def test_all_eight_required_stress_sequences_in_task_order(self):
        inventory, sources = fixture(); selected = p.select_pilot(inventory, sources)
        self.assertEqual(tuple(r["source_video_identity"] for r in selected[:8]),p.REQUIRED)

    def test_clean_controls_are_lexicographic_and_metadata_only(self):
        inventory, sources = fixture()
        with patch.object(p,"infer_image",side_effect=AssertionError("MediaPipe must not run")):
            selected = p.select_pilot(inventory,sources)
        self.assertEqual([r["source_video_identity"] for r in selected[-2:]],
                         ["Subject.1/Fall backwards/FallBackwardsS1.avi", "Subject.1/Hop/HopS1.avi"])
        self.assertEqual([r["video_label"] for r in selected[-2:]], ["fall","non_fall"])
        self.assertTrue(all(r["explicit_alias_pairs"]==r["exact_duplicate_groups"]==0 for r in selected[-2:]))

    def test_missing_stress_case_fails_closed(self):
        inventory,sources=fixture()
        inventory=[r for r in inventory if r["source_video_identity"]!=p.REQUIRED[0]]
        with self.assertRaises(ValueError):p.select_pilot(inventory,sources)

    def test_validation_rejected_before_source_io(self):
        for subject in (10,5):
            with patch.object(p.previous,"checked_path") as opener:
                with self.assertRaises(p.contract.ContractError):p.require_train(subject,"validation")
                opener.assert_not_called()
            with self.assertRaises(p.contract.ContractError):p.require_train(subject,"train")

    def test_test_rejected_before_source_io(self):
        for subject in (6,7):
            with patch.object(p.previous,"checked_path") as opener:
                with self.assertRaises(p.contract.ContractError):p.require_train(subject,"test")
                opener.assert_not_called()
            with self.assertRaises(p.contract.ContractError):p.require_train(subject,"train")

    def test_valid_pilot_schema_has_only_three_fields(self):
        x=arrays();p.validate_pilot_arrays(x,2)
        self.assertEqual(set(x),p.POSE_KEYS)

    def test_timestamp_field_is_forbidden(self):
        x=arrays();x["timestamp_ms"]=np.array([0,50],np.int64)
        with self.assertRaises(ValueError):p.validate_pilot_arrays(x,2)

    def test_inferred_fps_field_is_forbidden(self):
        x=arrays();x["fps"]=np.array([20],np.float32)
        with self.assertRaises(ValueError):p.validate_pilot_arrays(x,2)

    def test_annotation_ordinals_start_at_zero_and_are_contiguous(self):
        x=arrays(3);p.validate_pilot_arrays(x,3)
        x["annotation_ordinal"]=np.array([1,2,3],np.int32)
        with self.assertRaises(ValueError):p.validate_pilot_arrays(x,3)

    def test_detected_pose_values_must_be_finite(self):
        x=arrays();x["landmarks"][0,0,0]=np.nan
        with self.assertRaises(ValueError):p.validate_pilot_arrays(x,2)

    def test_missing_pose_values_must_be_all_nan(self):
        x=arrays(2,[True,False]);p.validate_pilot_arrays(x,2)
        x["landmarks"][1,0,0]=0
        with self.assertRaises(ValueError):p.validate_pilot_arrays(x,2)

    def test_zero_skeleton_cannot_encode_missing_pose(self):
        x=arrays(1,[False]);x["landmarks"][:]=0
        with self.assertRaises(ValueError):p.validate_pilot_arrays(x,1)

    def test_class_label_cannot_enter_npz(self):
        x=arrays();x["annotation_class_id"]=np.array([0,1],np.int32)
        with self.assertRaises(ValueError):p.validate_pilot_arrays(x,2)

    def test_npz_roundtrip_excludes_label_and_time(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temp:
            path=Path(temp)/"pilot.npz"
            p.write_pilot(path,arrays(2,[True,False]))
            loaded=p.load_pilot(path,2)
            self.assertEqual(set(loaded),p.POSE_KEYS)
            self.assertTrue(np.isnan(loaded["landmarks"][1]).all())

    def test_npz_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temp:
            path=Path(temp)/"pilot.npz"
            p.write_pilot(path,arrays())
            with self.assertRaises(ValueError):p.write_pilot(path,arrays())

    def test_one_to_one_png_annotation_ordinal_identity(self):
        rows=[dict(annotation_ordinal=i,png_filename=f"f{i}.png",annotation_filename=f"f{i}.txt") for i in range(3)]
        p.check_sequence_identity(rows,3)
        for damaged in (rows[:-1],[rows[0],rows[0],rows[2]],
                        [rows[0],dict(rows[1],annotation_filename="f0.txt"),rows[2]]):
            with self.assertRaises(ValueError):p.check_sequence_identity(damaged,3)

    def test_exact_duplicate_images_keep_separate_ordinals(self):
        x=arrays(2,[False,False]);p.validate_pilot_arrays(x,2)
        self.assertEqual(x["annotation_ordinal"].tolist(),[0,1])
        self.assertEqual(len(x["landmarks"]),2)

    def test_four_explicit_aliases_accepted(self):
        for source,png,txt in p.EXPLICIT_ALIASES:
            p.check_pairing(dict(source_video_identity=source,png_filename=png,
                                 annotation_filename=txt,annotation_pairing_status="verified_explicit_sequence_alias"))

    def test_unknown_alias_fails_closed(self):
        row=dict(source_video_identity=p.REQUIRED[0],png_filename="f001.png",
                 annotation_filename="f001a.txt",annotation_pairing_status="verified_explicit_sequence_alias")
        with self.assertRaises(ValueError):p.check_pairing(row)
        row["annotation_pairing_status"]="exact_stem"
        with self.assertRaises(ValueError):p.check_pairing(row)

    def test_exact_stem_pairing_allowed(self):
        p.check_pairing(dict(source_video_identity=p.REQUIRED[0],png_filename="f001.png",
                             annotation_filename="f001.txt",annotation_pairing_status="exact_stem"))

    def test_inference_api_cannot_accept_annotation_class(self):
        params=set(inspect.signature(p.infer_image).parameters)
        self.assertEqual(params,{"content","landmarker","cv2_module","mp"})
        self.assertFalse(any("label" in x or "class" in x for x in params))

    def test_synthetic_image_decodes_bgr_and_converts_to_rgb_without_time(self):
        bgr=np.zeros((3,4,3),np.uint8);bgr[:]=[10,20,30]
        content=cv2.imencode(".png",bgr)[1].tobytes()
        captured=[]
        mp=SimpleNamespace(ImageFormat=SimpleNamespace(SRGB="srgb"),Image=lambda **kwargs:captured.append(kwargs) or kwargs)
        landmarker=SimpleNamespace(detect=lambda image:SimpleNamespace(pose_landmarks=[]))
        present,values,shape=p.infer_image(content,landmarker,cv2,mp)
        self.assertFalse(present);self.assertTrue(np.isnan(values).all())
        self.assertEqual(shape,(3,4))
        self.assertEqual(captured[0]["data"][0,0].tolist(),[30,20,10])
        self.assertNotIn("timestamp_ms",captured[0])

    def test_corrupt_image_is_not_silent_missing_pose(self):
        with self.assertRaises(p.ImageDecodeError):
            p.infer_image(b"not a PNG",None,cv2,None)

    def test_repeatability_subset_includes_both_states_and_duplicate_pair(self):
        identity=[dict(source_video_identity=p.REQUIRED[0],annotation_ordinal=i,
            png_filename=f"f{i}.png",png_sha256=str(i),annotation_class_id=int(i>=1),
            annotation_semantic_label="grounded_fall_state" if i>=1 else "nofall") for i in range(3)]
        groups=[dict(source_video_identity=p.REQUIRED[0],identity_kind="decoded_pixel_sha256",ordinals="[1,2]")]
        chosen=p.select_repeatability(identity,groups)
        self.assertEqual({r["annotation_ordinal"] for r in chosen},{0,1,2})
        self.assertEqual({r["annotation_class_id"] for r in chosen},{0,1})

    def test_exact_observation_comparison_and_max_difference(self):
        base=(True,np.ones((33,4),np.float32))
        self.assertTrue(p.compare_observations(base,base)["exact_equal"])
        changed=base[1].copy();changed[0,0]=1.5
        diff=p.compare_observations(base,(True,changed))
        self.assertFalse(diff["exact_equal"])
        self.assertEqual(diff["max_abs_landmark_difference"],.5)

    def test_missing_structure_comparison(self):
        missing=(False,np.full((33,4),np.nan,np.float32))
        self.assertTrue(p.compare_observations(missing,missing)["exact_equal"])
        self.assertFalse(p.compare_observations(missing,(True,np.ones((33,4),np.float32)))["exact_equal"])

    def test_stage2_raw_pose_path_is_prohibited(self):
        path=p.REPO/"data/interim/caucafall_v5/pose_raw_v1_runs"/p.contract.RUN_ID/"pilot.npz"
        with self.assertRaises(p.contract.ContractError):p.contract.validate_output_path(path)


if __name__=="__main__":
    unittest.main()
