"""Stage 3.2f synthetic metadata/frame/controller tests; no pilot annotation."""

import copy
import csv
import hashlib
import json
import random
import tempfile
import threading
import unittest
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from unittest.mock import patch

from ml.preprocessing import annotation_pilot as pilot
from ml.preprocessing.annotation_tool import (
    AnnotationSession, DraftStore, annotator_payload,
)
from ml.preprocessing.annotation_web import annotator_state, create_server, frame_png_bytes
from ml.preprocessing.manual_annotation_contract import AnnotationContractError, validate_record


def grid():
    videos = []
    for subject in pilot.TRAIN_SUBJECTS:
        for label, activities in (("fall", pilot.FALL), ("non_fall", pilot.NON_FALL)):
            for activity in activities:
                videos.append(pilot.TrainVideo(subject, activity, label,
                    f"Subject.{subject}/{activity}/synthetic.avi", "0" * 64, 9, 20, 16, 16))
    return videos


def session(root):
    private = dict(neutral_clip_id="clip-0123456789abcdef",
                   source_video="Subject.1/Fall backwards/synthetic.avi",
                   source_avi_sha256="0" * 64, subject_id=1, split="train",
                   expected_frame_count=9, source_fps=20)
    public = dict(neutral_clip_id=private["neutral_clip_id"], frame_count=9, duration_ms=450)
    frames = pilot.IndexedFrameSource([f"synthetic-{i}" for i in range(9)])
    return AnnotationSession(private, public, frames, DraftStore(root),
                             "synthetic-pseudonym", "synthetic-session", "synthetic-run")


def observed(session_obj, name, indices):
    session_obj.set_status(name, "observed")
    for coordinate, index in zip(("earliest_plausible_frame", "preferred_frame", "latest_plausible_frame"), indices):
        session_obj.frames.at(index)
        session_obj.mark(name, coordinate)


class Stage32fToolTests(unittest.TestCase):
    def test_frozen_real_pilot_artifacts_validate_without_reopening_avi(self):
        with patch.object(pilot, "iter_indexed_frames", side_effect=AssertionError("No video reads")):
            self.assertEqual(pilot.validate_artifacts(), {
                "selected_clips": 12, "fall_clips": 6, "non_fall_clips": 6,
                "decoded_frames": 2163, "frame_count_mismatches": 0,
            })

    def test_selection_constraints_and_activity_coverage(self):
        chosen = pilot.select_pilot(grid())
        self.assertEqual(len(chosen), 12)
        self.assertEqual(Counter(v.label for v in chosen), {"fall": 6, "non_fall": 6})
        self.assertEqual(Counter((v.subject_id, v.label) for v in chosen),
                         {(s, label): 1 for s in pilot.TRAIN_SUBJECTS for label in ("fall", "non_fall")})
        self.assertEqual({v.activity for v in chosen if v.label == "fall"}, set(pilot.FALL))
        self.assertEqual({v.activity for v in chosen if v.label == "non_fall"}, set(pilot.NON_FALL))
        self.assertFalse({v.subject_id for v in chosen} & {5, 6, 7, 10})

    def test_selection_independent_of_input_order(self):
        candidates = grid()
        expected = pilot.select_pilot(candidates)
        random.Random(42).shuffle(candidates)
        self.assertEqual(pilot.select_pilot(candidates), expected)
        self.assertEqual(pilot.build_manifests(expected), pilot.build_manifests(pilot.select_pilot(candidates)))
        self.assertEqual(set(pilot.TrainVideo.__dataclass_fields__), {
            "subject_id", "activity", "label", "source_video", "source_sha256",
            "expected_frame_count", "source_fps", "width", "height",
        })

    def test_selection_ignores_pose_missingness_and_model_metadata_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "metadata.csv"
            contract = root / "contract.json"
            fields = ("subject_id", "original_activity", "binary_label", "split",
                      "source_relative_video_path", "source_sha256", "source_fps",
                      "width", "height", "expected_frame_count", "status",
                      "pose_detected_count", "missingness", "model_score")

            def write_rows(poisons, reverse=False):
                videos = list(reversed(grid())) if reverse else grid()
                with manifest.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    for index, video in enumerate(videos):
                        writer.writerow(dict(subject_id=video.subject_id, original_activity=video.activity,
                            binary_label=video.label, split="train", source_relative_video_path=video.source_video,
                            source_sha256=video.source_sha256, source_fps="20.0", width=16, height=16,
                            expected_frame_count=9, status="complete", pose_detected_count=poisons[index],
                            missingness=poisons[index], model_score=poisons[index]))
                contract.write_text(json.dumps({"source": {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}}))

            write_rows(list(range(60)))
            expected = pilot.select_pilot(pilot.load_train_metadata(manifest, contract))
            write_rows(list(reversed(range(60))), reverse=True)
            self.assertEqual(pilot.select_pilot(pilot.load_train_metadata(manifest, contract)), expected)

    def test_missing_activity_constraint_fails_without_substitution(self):
        candidates = [v for v in grid() if not (v.subject_id == 2 and v.activity == "Fall forward")]
        with self.assertRaisesRegex(pilot.PilotError, "Incomplete or duplicate"):
            pilot.select_pilot(candidates)

    def test_freeze_precedes_any_avi_decode_and_public_manifest_is_blinded(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(pilot, "load_train_metadata", return_value=grid()), \
                patch.object(pilot, "iter_indexed_frames", side_effect=AssertionError("Decoded before freeze")):
            out = pilot.freeze_pilot(Path(tmp) / "pilot")
            private, public, order = pilot.load_frozen(out)
            self.assertEqual(len(private), len(public))
            self.assertEqual([row["neutral_clip_id"] for row in public], order)
            self.assertEqual(set(public[0]), {"neutral_clip_id", "frame_count", "duration_ms"})
            self.assertTrue(all(row["split"] == "train" for row in private))
            self.assertTrue(all(row["neutral_clip_id"].startswith("clip-") for row in private))
            self.assertEqual(len(set(order)), 12)
            labels = [next(row["dataset_label"] for row in private if row["neutral_clip_id"] == clip_id) for clip_id in order]
            self.assertFalse(all(a != b for a, b in zip(labels, labels[1:])))
            self.assertEqual(json.loads((out / "freeze_record.json").read_text())["decode_verification_at_freeze"], "NOT_RUN")
            with self.assertRaises(FileExistsError):
                pilot.freeze_pilot(out)
            (out / "presentation_order.json").write_text("[]")
            with self.assertRaisesRegex(pilot.PilotError, "changed"):
                pilot.load_frozen(out)

    def test_indexed_navigation_is_exact_and_clamped(self):
        source = pilot.IndexedFrameSource([f"frame-{i}" for i in range(4)])
        self.assertEqual(source.at(0), (0, "frame-0"))
        self.assertEqual(source.previous(), (0, "frame-0"))
        self.assertEqual(source.next(), (1, "frame-1"))
        self.assertEqual(source.at(3), (3, "frame-3"))
        self.assertEqual(source.next(), (3, "frame-3"))
        self.assertEqual(source.previous(), (2, "frame-2"))
        self.assertEqual(source.at(0), (0, "frame-0"))
        with self.assertRaises(pilot.PilotError):
            source.at(4)

    def test_synthetic_codec_sequential_decode_and_cache(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (32, 24))
            self.assertTrue(writer.isOpened(), "MJPG writer unavailable; codec integration cannot be verified")
            for value in (20, 100, 200):
                writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
            writer.release()
            pairs = list(pilot.iter_indexed_frames(path, 3, 20, 32, 24))
            self.assertEqual([index for index, _ in pairs], [0, 1, 2])
            means = [round(float(frame.mean())) for _, frame in pairs]
            self.assertTrue(all(abs(actual - expected) < 8 for actual, expected in zip(means, (20, 100, 200))), means)
            source = pilot.IndexedFrameSource.open_avi(path, 3, 20, 32, 24)
            self.assertEqual(source.at(2)[0], 2)
            self.assertEqual(source.previous()[0], 1)
            self.assertTrue(frame_png_bytes(source.at(0)[1]).startswith(b"\x89PNG"))
            with self.assertRaisesRegex(pilot.PilotError, "metadata differs"):
                list(pilot.iter_indexed_frames(path, 4, 20, 32, 24))

    def test_public_payload_rejects_private_fields(self):
        payload = dict(neutral_clip_id="clip-0123456789abcdef", frame_count=9, duration_ms=450)
        self.assertEqual(annotator_payload(payload), payload)
        for key in ("subject_id", "activity", "source_video", "source_filename", "dataset_label",
                    "png_label", "pose_raw_v1", "pose_detected", "mediapipe_missingness",
                    "model_probability", "other_annotator_record"):
            with self.subTest(key=key), self.assertRaisesRegex(pilot.PilotError, "private"):
                annotator_payload(dict(payload, **{key: "secret"}))
        with tempfile.TemporaryDirectory() as tmp:
            controller = session(Path(tmp))
            self.assertEqual(set(controller.public), set(payload))
            self.assertNotIn("Subject.1", json.dumps(controller.public))
            view = annotator_state(controller)
            self.assertNotIn("Subject.1", json.dumps(view))
            self.assertFalse(set(view) & {"source_video", "subject_id", "activity", "dataset_label"})

    def test_state_vocabulary_coordinates_and_no_fall_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = session(Path(tmp))
            with self.assertRaises(AnnotationContractError):
                controller.set_event_presence("dataset_fall")
            controller.set_event_presence("no_fall_observed")
            with self.assertRaises(AnnotationContractError):
                controller.set_status("grounded_start", "observed")
            validate_record(controller.record)
            controller.set_event_presence("fall_observed")
            with self.assertRaises(AnnotationContractError):
                controller.set_status("grounded_start", "not_applicable")
            with self.assertRaises(AnnotationContractError):
                controller.set_status("grounded_start", "invalid")
            observed(controller, "fall_transition_start", (1, 2, 3))
            observed(controller, "grounded_start", (4, 5, 6))
            controller.set_reason_flags("event", ["motion_blur"])
            controller.set_reason_flags("grounded_start", ["ambiguous_grounded_state"])
            controller.set_note("synthetic note")
            validate_record(controller.record)
            self.assertEqual(controller.record["boundaries"]["grounded_start"]["preferred_frame"], 5)
            self.assertEqual(controller.record["event_presence_reason_flags"], ["motion_blur"])
            self.assertEqual(controller.record["annotator_note"], "synthetic note")

    def test_invalid_triplet_order_and_event_order_fail_without_autocorrection(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = session(Path(tmp))
            controller.set_event_presence("fall_observed")
            observed(controller, "fall_transition_start", (1, 2, 3))
            observed(controller, "grounded_start", (4, 5, 6))
            before = copy.deepcopy(controller.record)
            controller.record["boundaries"]["grounded_start"]["earliest_plausible_frame"] = 6
            with self.assertRaisesRegex(AnnotationContractError, "earliest <= preferred"):
                controller.finalize()
            controller.record = copy.deepcopy(before)
            controller.record["boundaries"]["grounded_start"]["latest_plausible_frame"] = 4
            with self.assertRaisesRegex(AnnotationContractError, "earliest <= preferred"):
                controller.finalize()
            controller.record = copy.deepcopy(before)
            controller.record["boundaries"]["fall_transition_start"]["preferred_frame"] = 7
            controller.record["boundaries"]["fall_transition_start"]["latest_plausible_frame"] = 8
            with self.assertRaisesRegex(AnnotationContractError, "event order"):
                controller.finalize()
            self.assertEqual(controller.record["boundaries"]["fall_transition_start"]["preferred_frame"], 7)

    def test_overlap_draft_resume_and_finalized_exclusive_create(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = session(Path(tmp))
            controller.set_event_presence("fall_observed")
            observed(controller, "fall_transition_start", (1, 2, 7))
            observed(controller, "grounded_start", (3, 5, 6))
            controller.set_reason_flags("event", ["occlusion"])
            controller.set_note("synthetic draft")
            draft_path = controller.save_draft()
            self.assertTrue(draft_path.exists())
            recovered = controller.store.load_draft(controller.record["annotation_record_id"])
            resumed = AnnotationSession(
                dict(neutral_clip_id="clip-0123456789abcdef", source_video="Subject.1/Fall backwards/synthetic.avi",
                     source_avi_sha256="0" * 64, subject_id=1, split="train", expected_frame_count=9),
                controller.public, controller.frames, controller.store,
                "synthetic-pseudonym", "synthetic-session", "synthetic-run", recovered)
            self.assertEqual(resumed.record["annotator_note"], "synthetic draft")
            final_path = resumed.finalize()
            self.assertTrue(final_path.exists())
            self.assertFalse(draft_path.exists())
            wrapper = json.loads(final_path.read_text())
            self.assertEqual(wrapper["state"], "FINALIZED")
            validate_record(wrapper["record"])
            with self.assertRaises(AnnotationContractError):
                resumed.finalize()
            with self.assertRaises(AnnotationContractError):
                resumed.save_draft()
            with self.assertRaises(AnnotationContractError):
                controller.store.load_draft(controller.record["annotation_record_id"])

    def test_loopback_http_view_and_actions_keep_private_provenance_hidden(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            controller = session(Path(tmp))
            yy, xx = np.indices((17, 19))
            patterns = [np.stack(((xx * 11 + yy * 7 + i * 13) % 256,
                                  (xx * 3 + yy * 17 + i * 29) % 256,
                                  (xx * 23 + yy * 5 + i * 41) % 256), axis=-1).astype(np.uint8)
                        for i in range(9)]
            controller.frames = pilot.IndexedFrameSource(patterns)
            server, token = create_server(controller, 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"

            def fetch_frame(index):
                query = urlencode({"clip": controller.public["neutral_clip_id"], "index": index})
                with urlopen(base + "/frame?" + query) as response:
                    self.assertEqual(response.headers["Content-Type"], "image/png")
                    self.assertEqual(response.headers["Cache-Control"], "no-store")
                    self.assertEqual(response.headers["X-Frame-Index"], str(index))
                    self.assertEqual(response.headers["X-Neutral-Clip-Id"], controller.public["neutral_clip_id"])
                    self.assertNotIn("Subject.1", str(response.headers))
                    encoded = response.read()
                self.assertTrue(encoded.startswith(b"\x89PNG\r\n\x1a\n"))
                return cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            def post(action):
                request = Request(base + "/action", data=json.dumps(action).encode(),
                                  headers={"Content-Type": "application/json", "X-Tool-Token": token})
                result = json.loads(urlopen(request).read())
                self.assertNotIn("Subject.1", json.dumps(result))
                return result["state"]

            try:
                page = urlopen(base + "/").read().decode()
                self.assertNotIn("Subject.1", page)
                self.assertNotIn("Fall backwards", page)
                self.assertNotIn("synthetic.avi", page)
                self.assertIn("frame_index", page)
                self.assertIn("await ready.decode()", page)
                self.assertIn("epoch !== renderEpoch", page)
                self.assertIn("replaceWith(ready)", page)
                with urlopen(base + "/state") as response:
                    self.assertEqual(response.headers["Cache-Control"], "no-store")
                    payload = json.loads(response.read())
                self.assertEqual(payload["neutral_clip_id"], controller.public["neutral_clip_id"])
                self.assertNotIn("source_video", payload)
                for index in (0, 1, 4, 7, 8):
                    with self.subTest(index=index):
                        before = controller.frames.index
                        self.assertTrue(np.array_equal(fetch_frame(index), patterns[index]))
                        self.assertEqual(controller.frames.index, before)

                with patch.object(controller.frames, "peek", side_effect=lambda index: (index, patterns[min(index + 1, 8)])):
                    with self.assertRaises(AssertionError):
                        self.assertTrue(np.array_equal(fetch_frame(4), patterns[4]))

                for action, expected in (({"verb": "step", "amount": 1}, 1),
                                         ({"verb": "step", "amount": -1}, 0),
                                         ({"verb": "jump", "index": 4}, 4),
                                         ({"verb": "jump", "index": 8}, 8),
                                         ({"verb": "step", "amount": 1}, 8),
                                         ({"verb": "jump", "index": 0}, 0)):
                    state = post(action)
                    self.assertEqual(state["frame_index"], expected)
                    self.assertEqual(state["time_ms"], expected * 50)
                    self.assertTrue(np.array_equal(fetch_frame(state["frame_index"]), patterns[expected]))

                stale_state = json.loads(urlopen(base + "/state").read())
                post({"verb": "step", "amount": 1})
                self.assertEqual(stale_state["frame_index"], 0)
                self.assertEqual(controller.frames.index, 1)
                self.assertTrue(np.array_equal(fetch_frame(stale_state["frame_index"]), patterns[0]))
                for _ in range(20):
                    post({"verb": "step", "amount": 1})
                    current = json.loads(urlopen(base + "/state").read())
                    self.assertTrue(np.array_equal(fetch_frame(current["frame_index"]), patterns[current["frame_index"]]))
                self.assertEqual(controller.frames.index, 8)

                for invalid in ("/frame", "/frame?index=0", "/frame?clip=wrong&index=0",
                                "/frame?clip=" + controller.public["neutral_clip_id"] + "&index=9"):
                    with self.subTest(invalid=invalid), self.assertRaises(HTTPError):
                        urlopen(base + invalid)
                with self.assertRaises(HTTPError):
                    urlopen(Request(base + "/action", data=b'{"verb":"step","amount":1}',
                                    headers={"Content-Type": "application/json"}))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
