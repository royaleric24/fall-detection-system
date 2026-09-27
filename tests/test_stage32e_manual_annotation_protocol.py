"""Synthetic in-memory Stage 3.2e contract fixtures; no media or pose reads."""

import copy
import json
import unittest
from pathlib import Path

from ml.preprocessing.manual_annotation_contract import (
    AnnotationContractError,
    validate_adjudication_link,
    validate_record,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/stage32e_manual_annotation_protocol.json"
STRATEGY = ROOT / "configs/stage32d_temporal_supervision_strategy.json"


def observed(frame):
    return {
        "status": "observed", "reason_flags": [],
        "earliest_plausible_frame": frame - 2,
        "preferred_frame": frame,
        "latest_plausible_frame": frame + 2,
        "preferred_timestamp_ms": frame * 50,
    }


def synthetic_record():
    """Return only an in-memory schema fixture, never a dataset annotation."""
    return {
        "annotation_record_id": "synthetic-record-A",
        "protocol_version": "stage32e_v1",
        "source": {
            "source_video": "synthetic/clip.avi",
            "neutral_clip_id": "clip-neutral-A",
            "subject_id": 8,
            "split": "train",
            "source_avi_sha256": None,
            "frame_count": 100,
            "source_fps": 20,
            "source_fps_provenance": "frozen_Stage_2_AVI_timeline",
        },
        "annotator": {"pseudonymous_id": "synthetic-annotator-A", "session_id": "synthetic-session"},
        "event_presence": "fall_observed",
        "event_presence_reason_flags": [],
        "boundaries": {
            "fall_transition_start": observed(10),
            "grounded_start": observed(30),
            "recovery_start": observed(60),
        },
        "creation": {
            "created_at_utc": "2026-01-01T00:00:00Z",
            "annotation_run_id": "synthetic-run",
            "tool_version": "schema-test-only",
        },
    }


class ManualAnnotationProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        cls.strategy = json.loads(STRATEGY.read_text(encoding="utf-8"))

    def assert_invalid(self, change, match):
        record = synthetic_record()
        change(record)
        with self.assertRaisesRegex(AnnotationContractError, match):
            validate_record(record)

    def test_parent_and_strategy_are_frozen(self):
        p = self.protocol
        self.assertEqual((p["stage"], p["parent_commit"]),
                         ("3.2e", "5c420adc145767ca83a065da7e9bdbe0261bcc0d"))
        self.assertEqual(p["strategy_contract"], "configs/stage32d_temporal_supervision_strategy.json")
        self.assertEqual(self.strategy["selected_primary_route"], "B")

    def test_resolution_is_explicitly_scoped_to_manual_protocol(self):
        resolution = self.protocol["decision_resolution"]
        self.assertEqual(resolution["precedence_rule"],
                         "latest_explicit_resolution_wins_only_within_declared_scope")
        self.assertEqual(resolution["authoritative_from_stage"], "3.2e")
        self.assertEqual(resolution["stage32d_route_and_source_decisions"], "UNCHANGED")
        self.assertEqual(set(resolution["resolved_subdecisions"]), {
            "manual_event_presence_vocabulary", "minimal_event_boundary_ontology",
            "event_boundary_semantics_and_statuses", "boundary_uncertainty_representation",
            "annotation_record_schema", "annotation_reason_vocabulary",
            "blinding_requirements", "inter_annotator_agreement_methodology",
            "adjudication_record_separation",
        })
        for field in resolution["resolved_subdecisions"].values():
            value = self.protocol
            for part in field.split("."):
                value = value[part]
            self.assertTrue(value)
        self.assertEqual(set(resolution["remaining_unresolved_subdecisions"]),
                         set(self.protocol["deferred"]))

    def test_event_presence_vocabulary_and_manual_source(self):
        p = self.protocol["event_presence"]
        self.assertEqual(p["vocabulary"], ["fall_observed", "no_fall_observed", "uncertain"])
        self.assertEqual(p["source"], "manual_visual_decision")
        validate_record(synthetic_record())
        self.assert_invalid(lambda r: r.update(event_presence="dataset_fall"), "event_presence")
        self.assertEqual(self.protocol["evidence"]["activity_filename_or_dataset_label_as_event_presence"],
                         "PROHIBITED")

    def test_no_fall_cannot_receive_fall_boundaries(self):
        self.assert_invalid(lambda r: r.update(event_presence="no_fall_observed"), "no_fall_observed")
        record = synthetic_record()
        record["event_presence"] = "no_fall_observed"
        record["boundaries"] = {name: {"status": "not_applicable", "reason_flags": []}
                                for name in record["boundaries"]}
        validate_record(record)

    def test_uncertain_requires_reason_and_explicit_boundary_statuses(self):
        self.assert_invalid(lambda r: r.update(event_presence="uncertain"), "requires a reason")
        record = synthetic_record()
        record["event_presence"] = "uncertain"
        record["event_presence_reason_flags"] = ["insufficient_clip_context"]
        record["boundaries"] = {
            "fall_transition_start": {"status": "unjudgeable", "reason_flags": ["occlusion"]},
            "grounded_start": {"status": "unjudgeable", "reason_flags": ["occlusion"]},
            "recovery_start": {"status": "not_applicable", "reason_flags": []},
        }
        validate_record(record)
        record["boundaries"]["fall_transition_start"] = observed(10)
        with self.assertRaisesRegex(AnnotationContractError, "uncertain event"):
            validate_record(record)

    def test_observed_boundary_requires_three_frame_coordinates(self):
        self.assert_invalid(lambda r: r["boundaries"]["grounded_start"].pop("earliest_plausible_frame"),
                            "requires earliest/preferred/latest")
        self.assert_invalid(lambda r: r["boundaries"]["grounded_start"].update(preferred_frame=True),
                            "source AVI frame indices")
        self.assert_invalid(lambda r: r["boundaries"]["grounded_start"].update(latest_plausible_frame=100),
                            "source AVI frame indices")

    def test_status_and_reason_vocabularies_are_controlled(self):
        self.assertEqual(set(self.protocol["boundaries"]["boundary_statuses"]), {
            "observed", "left_censored", "right_censored", "not_observed",
            "unjudgeable", "not_applicable",
        })
        self.assertEqual(set(self.protocol["reason_flags"]), {
            "occlusion", "subject_out_of_frame", "motion_blur",
            "ambiguous_transition_start", "ambiguous_grounded_state",
            "ambiguous_recovery", "insufficient_clip_context", "other",
        })
        self.assert_invalid(lambda r: r["boundaries"]["grounded_start"].update(status=[]), "invalid status")
        self.assert_invalid(lambda r: r["boundaries"]["grounded_start"].update(reason_flags=["model_score"]),
                            "unknown reason flag")

    def test_uncertainty_triplet_order_is_checked(self):
        self.assert_invalid(lambda r: r["boundaries"]["grounded_start"].update(earliest_plausible_frame=31),
                            "earliest <= preferred <= latest")
        self.assert_invalid(lambda r: r["boundaries"]["grounded_start"].update(latest_plausible_frame=29),
                            "earliest <= preferred <= latest")

    def test_censored_and_unobserved_boundaries_have_no_fabricated_coordinates(self):
        for status in ("left_censored", "right_censored", "not_observed", "unjudgeable", "not_applicable"):
            with self.subTest(status=status):
                self.assert_invalid(
                    lambda r, s=status: r["boundaries"]["recovery_start"].update(status=s),
                    "cannot fabricate unknown coordinates",
                )
        self.assert_invalid(lambda r: r["boundaries"]["fall_transition_start"].update(
            status="not_observed"), "cannot fabricate unknown coordinates")

    def test_optional_recovery_may_be_absent(self):
        record = synthetic_record()
        record["boundaries"]["recovery_start"] = {"status": "not_observed", "reason_flags": []}
        validate_record(record)
        record["boundaries"]["fall_transition_start"] = {"status": "not_observed", "reason_flags": []}
        with self.assertRaisesRegex(AnnotationContractError, "only for optional recovery"):
            validate_record(record)

    def test_unjudgeable_requires_reason(self):
        record = synthetic_record()
        record["boundaries"]["grounded_start"] = {"status": "unjudgeable", "reason_flags": []}
        with self.assertRaisesRegex(AnnotationContractError, "requires a reason"):
            validate_record(record)
        record["boundaries"]["grounded_start"]["reason_flags"] = ["occlusion"]
        validate_record(record)

    def test_observed_preferred_transition_precedes_grounded(self):
        self.assert_invalid(lambda r: r["boundaries"].update(
            fall_transition_start=observed(35)), "event order violated")

    def test_observed_recovery_follows_grounded(self):
        self.assert_invalid(lambda r: r["boundaries"].update(
            recovery_start=observed(25)), "event order violated")

    def test_overlapping_uncertainty_intervals_are_allowed(self):
        record = synthetic_record()
        record["boundaries"]["fall_transition_start"].update(latest_plausible_frame=35)
        record["boundaries"]["grounded_start"].update(earliest_plausible_frame=8)
        validate_record(record)
        self.assertTrue(self.protocol["boundaries"]["uncertainty_intervals_may_overlap"])

    def test_frame_index_authoritative_timestamp_derived(self):
        p = self.protocol["record_schema"]
        self.assertEqual(p["timestamp_formula"], "round(preferred_frame * 1000 / 20)")
        self.assertEqual(p["timestamp_entry_by_annotator"], "PROHIBITED")
        self.assertEqual(self.strategy["canonical_temporal_source"]["authoritative_manual_boundary_coordinate"],
                         "frame_index")
        self.assert_invalid(lambda r: r["boundaries"]["grounded_start"].update(preferred_timestamp_ms=1499),
                            "machine-derived")
        self.assert_invalid(lambda r: r.update(timestamp_ms=1500), "unknown fields")
        record = synthetic_record()
        del record["boundaries"]["grounded_start"]["preferred_timestamp_ms"]
        validate_record(record)

    def test_pose_and_model_evidence_prohibited(self):
        evidence = self.protocol["evidence"]
        self.assertEqual(evidence["manual_boundary_source"], "source_AVI_visual_pixels_only")
        self.assertEqual(evidence["canonical_downstream_pose_source"],
                         "Stage_2_pose_raw_v1_not_annotation_evidence")
        self.assertEqual(set(evidence["prohibited_boundary_inputs"]), {
            "pose_raw_v1", "pose_detected", "MediaPipe_landmarks_missingness_or_features",
            "pose_raw_v1_skeleton_visualizations", "model_predictions_or_confidence",
            "model_errors_or_performance", "Validation_or_Test_performance",
            "future_alert_outputs", "same_video_official_PNG_TXT_frame_labels",
        })
        self.assert_invalid(lambda r: r.update(model_prediction=0.8), "unknown fields")

    def test_png_semantic_reference_is_not_per_video_boundary_evidence(self):
        evidence = self.protocol["evidence"]
        self.assertEqual(evidence["official_semantic_reference_for_protocol_design"],
                         "PERMITTED_already_reviewed_grounded_fall_state_meaning_only")
        self.assertEqual(evidence["official_PNG_TXT_to_AVI_frame_coordinate_transfer"], "PROHIBITED")
        self.assertIn("same_video_official_PNG_TXT_frame_labels", evidence["prohibited_boundary_inputs"])

    def test_provenance_and_blinded_display_are_separate(self):
        schema = self.protocol["record_schema"]
        self.assertIn("source_video", schema["source_required"])
        self.assertIn("neutral_clip_id", schema["source_required"])
        self.assertIn("source_avi_sha256", schema["source_required"])
        self.assertEqual(schema["stored_provenance_vs_annotator_display"], "SEPARATE")
        display = self.protocol["future_tool_blinding"]
        self.assertIn("revealing_filename", display["hide"])
        self.assertIn("pose_or_skeleton_data", display["hide"])
        self.assertIn("original_AVI_visual_frames", display["may_display"])
        self.assertIn("neutral_clip_id", display["may_display"])
        self.assert_invalid(lambda r: r["source"].update(source_avi_sha256="not-a-hash"),
                            "SHA-256")
        self.assert_invalid(lambda r: r["source"].update(source_fps=23), "20 FPS")

    def test_adjudication_has_separate_identity_and_preserves_originals(self):
        originals = [synthetic_record(), synthetic_record()]
        originals[1]["annotation_record_id"] = "synthetic-record-B"
        before = copy.deepcopy(originals)
        link = {
            "adjudication_record_id": "synthetic-adjudication",
            "protocol_version": "stage32e_v1",
            "original_annotation_record_ids": ["synthetic-record-A", "synthetic-record-B"],
            "adjudicator_pseudonymous_id": "synthetic-adjudicator",
            "created_at_utc": "2026-01-02T00:00:00Z",
        }
        validate_adjudication_link(link, ("synthetic-record-A", "synthetic-record-B"))
        self.assertEqual(originals, before)
        link["adjudication_record_id"] = "synthetic-record-A"
        with self.assertRaisesRegex(AnnotationContractError, "separate identity"):
            validate_adjudication_link(link, ("synthetic-record-A", "synthetic-record-B"))
        self.assertEqual(self.protocol["inter_annotator_protocol"]["original_records_after_adjudication"],
                         "IMMUTABLE")

    def test_agreement_is_event_level_with_descriptive_boundary_metrics(self):
        agreement = self.protocol["agreement"]
        self.assertEqual(agreement["unit"], "clip_event_or_jointly_observed_boundary_not_independent_frames")
        self.assertEqual(agreement["event_presence"], ["raw_agreement", "Cohens_kappa_where_defined"])
        self.assertIn("abs(preferred_frame_A-preferred_frame_B)",
                      agreement["boundary_preferred_frame_absolute_disagreement"])
        self.assertIn("1000/20", agreement["boundary_preferred_time_disagreement_ms"])
        self.assertIn("<=", agreement["uncertainty_interval_overlap"])
        self.assertEqual(agreement["uncertainty_interval_gap_frames"],
                         "max(0,earliest_A-latest_B,earliest_B-latest_A)")
        self.assertEqual(agreement["required_agreement_threshold"], "UNDECIDED")

    def test_future_pilot_train_only_and_both_event_types(self):
        pilot = self.protocol["inter_annotator_protocol"]
        self.assertEqual(pilot["pilot_split"], "train_only")
        self.assertEqual(pilot["annotators_minimum"], 2)
        self.assertTrue(pilot["same_frozen_subset_and_protocol_version"])
        self.assertTrue(pilot["independent_before_adjudication"])
        self.assertTrue({"fall_source_videos", "non_fall_source_videos", "hard_negative_ADL"}
                        <= set(pilot["pilot_must_include"]))
        self.assertEqual(set(pilot["pilot_selection_basis_excludes"]),
                         {"model_predictions", "pose_missingness"})
        self.assert_invalid(lambda r: r["source"].update(split="validation", subject_id=10),
                            "Train subject")
        self.assert_invalid(lambda r: r["source"].update(split="test", subject_id=6),
                            "Train subject")

    def test_validation_and_test_unavailable_for_protocol_development(self):
        pilot = self.protocol["inter_annotator_protocol"]
        self.assertEqual(pilot["validation_protocol_development"], "PROHIBITED")
        self.assertEqual(pilot["test_protocol_development"], "PROHIBITED")
        self.assertEqual(pilot["test_subjects"], [6, 7])
        self.assertEqual(pilot["test_status"], "SEALED")

    def test_final_window_labels_and_thresholds_remain_undecided(self):
        deferred = self.protocol["deferred"]
        self.assertEqual(set(deferred.values()), {"UNDECIDED"})
        self.assertIn("final_window_label_rule", deferred)
        self.assertIn("window_overlap_threshold", deferred)
        self.assertIn("pilot_sample_size_and_selection", deferred)
        self.assert_invalid(lambda r: r.update(window_label=1), "unknown fields")

    def test_stage32e_cannot_generate_annotations_or_model_ready_data(self):
        self.assertEqual(set(self.protocol["execution"].values()), {False})
        self.assertEqual(self.protocol["review_status"], "pending_external_gate_review")
        self.assertFalse((ROOT / "artifacts/annotations/stage32e").exists())


if __name__ == "__main__":
    unittest.main()
