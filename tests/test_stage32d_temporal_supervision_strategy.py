"""Stage 3.2d contract tests; metadata-only, no media or pose-array reads."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/stage32d_temporal_supervision_strategy.json"
STAGE3 = ROOT / "configs/stage3_contract.json"


class TemporalSupervisionStrategyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.strategy = json.loads(CONTRACT.read_text(encoding="utf-8"))
        cls.frozen = json.loads(STAGE3.read_text(encoding="utf-8"))

    def test_versioned_parent_and_review_status(self):
        s = self.strategy
        self.assertEqual((s["contract_version"], s["stage"]), (1, "3.2d"))
        self.assertEqual(s["parent_commit"], "d99478c68f2bc1bcaa736ca610affc5a8135eee6")
        self.assertEqual(s["review_status"], "pending_external_gate_review")

    def test_canonical_timeline_is_stage2_decoded_avi(self):
        self.assertEqual(self.strategy["canonical_temporal_source"]["identity"], "Stage 2 decoded AVI timeline")
        self.assertEqual(self.strategy["canonical_temporal_source"]["dataset"], "CAUCAFall V5 local distributed AVI")

    def test_canonical_pose_is_frozen_stage2_pose_raw_v1(self):
        p = self.strategy["canonical_pose_source"]
        self.assertEqual((p["identity"],p["schema_version"]), ("Stage 2 pose_raw_v1","pose_raw_v1"))
        self.assertEqual(p["run_id"],self.frozen["source"]["run_id"])
        self.assertEqual(p["pose_root"],self.frozen["source"]["pose_root"])

    def test_local_avi_rate_is_20_with_source_provenance(self):
        t=self.strategy["canonical_temporal_source"]
        self.assertEqual((t["source_fps"],t["source_fps_unit"]),(20,"frames_per_second"))
        self.assertIn("artifacts/dataset_inspection/inventory.csv",t["source_fps_provenance"])
        self.assertIn("docs/pose_extraction_contract.md",t["source_fps_provenance"])

    def test_frame_index_is_authoritative_and_timestamp_is_derived(self):
        t=self.strategy["canonical_temporal_source"]
        self.assertEqual((t["authoritative_manual_boundary_coordinate"],t["frame_index_origin"]),("frame_index",0))
        self.assertEqual(t["derived_coordinate"],"timestamp_ms")
        self.assertEqual(t["timestamp_method"],"round(frame_index * 1000 / source_fps)")
        self.assertEqual(t["rounding"],"Python round, ties to even")
        self.assertEqual(t["timestamp_scope"],"clip_relative_not_unix_epoch")
        self.assertEqual(t["timestamp_entry_policy"],"derived_from_frozen_AVI_timeline_not_hand_entered")

    def test_manual_annotation_source_and_required_lineage(self):
        a=self.strategy["future_temporal_annotation"]
        self.assertEqual(a["source"],"manual_review_of_original_AVI_visual_evidence")
        self.assertEqual(a["manual_boundary_coordinate"],"frame_index")
        self.assertTrue({"source_video","subject_id","activity","frame_index","timestamp_ms",
                         "annotation_protocol_version","annotator_id_or_pseudonym","annotation_provenance"}
                        <= set(a["required_record_fields"]))

    def test_historical_stage3_decision_and_scoped_downstream_precedence(self):
        resolution = self.strategy["decision_resolution"]
        historical = resolution["historical_parent_decision"]
        self.assertEqual(historical, {
            "contract": "configs/stage3_contract.json",
            "field": "decisions.temporal_supervision_protocol",
            "value_at_stage_3_0_gate": "UNDECIDED",
        })
        self.assertEqual(self.frozen["decisions"]["temporal_supervision_protocol"], "UNDECIDED")
        self.assertEqual(resolution["precedence_rule"], "latest_explicit_resolution_wins_only_within_declared_scope")
        self.assertEqual(resolution["authoritative_from_stage"], "3.2d")
        self.assertEqual(resolution["broader_temporal_supervision_protocol"], "NOT_FULLY_RESOLVED")

    def test_declared_resolution_scope_points_to_selected_values(self):
        scope = self.strategy["decision_resolution"]["resolved_subdecisions"]
        self.assertEqual(set(scope), {
            "primary_temporal_supervision_strategy",
            "canonical_temporal_source",
            "canonical_pose_source",
            "authoritative_temporal_coordinate_system",
            "official_png_txt_annotation_role",
            "png_to_avi_frame_label_join",
            "video_level_weak_supervision_role",
        })
        expected = {
            "primary_temporal_supervision_strategy": "B",
            "canonical_temporal_source": "Stage 2 decoded AVI timeline",
            "canonical_pose_source": "Stage 2 pose_raw_v1",
            "authoritative_temporal_coordinate_system": "frame_index",
            "official_png_txt_annotation_role": "semantic_and_annotation_native_reference_only",
            "png_to_avi_frame_label_join": "PROHIBITED",
            "video_level_weak_supervision_role": "fallback_only",
        }
        for subdecision, entry in scope.items():
            with self.subTest(subdecision=subdecision):
                self.assertEqual(entry["status"], "RESOLVED")
                value = self.strategy
                for part in entry["contract_field"].split("."):
                    value = value[part]
                self.assertEqual(value, expected[subdecision])

    def test_manual_protocol_details_remain_unresolved(self):
        remaining = self.strategy["decision_resolution"]["remaining_unresolved_subdecisions"]
        self.assertEqual(set(remaining), {
            "manual_annotation_ontology",
            "state_definitions",
            "event_boundary_definitions",
            "uncertainty_ambiguity_representation",
            "annotation_schema_details",
            "annotator_agreement_threshold",
            "adjudication_protocol_details",
            "final_window_label_rule",
        })
        self.assertEqual(set(remaining.values()), {"UNDECIDED"})
        self.assertEqual(self.strategy["future_temporal_annotation"]["ontology"], "UNDECIDED")
        self.assertEqual(self.strategy["future_temporal_annotation"]["operational_boundary_definitions"], "UNDECIDED")
        self.assertEqual(self.strategy["unresolved_decisions"]["window_supervision_rule"], "UNDECIDED")

    def test_manual_avi_pixels_are_separate_from_downstream_pose(self):
        q = self.strategy["annotation_quality_requirements"]
        self.assertEqual(q["visual_evidence_source"], "original_source_AVI_visual_pixels_only")
        self.assertEqual(q["canonical_downstream_pose_source"], "Stage_2_pose_raw_v1_not_annotation_evidence")
        self.assertEqual(self.strategy["canonical_pose_source"]["schema_version"], "pose_raw_v1")
        self.assertEqual(self.strategy["future_temporal_annotation"]["manual_boundary_coordinate"], "frame_index")

    def test_pose_mediapi_and_model_signals_cannot_set_manual_boundaries(self):
        prohibited = self.strategy["annotation_quality_requirements"]["prohibited_manual_boundary_inputs"]
        self.assertEqual(set(prohibited), {
            "pose_raw_v1_landmarks", "pose_detected_state", "MediaPipe_missingness_patterns",
            "any_MediaPipe_derived_feature", "pose_raw_v1_skeleton_visualization",
            "model_predictions", "model_probabilities_or_confidence", "model_errors",
            "model_performance", "validation_performance", "future_alert_system_outputs",
            "same_video_official_PNG_TXT_frame_labels",
        })
        self.assertEqual(set(prohibited.values()), {"PROHIBITED"})

    def test_png_semantic_reference_is_separate_from_per_video_boundaries(self):
        official = self.strategy["official_png_annotations"]
        self.assertEqual(official["protocol_design_semantic_reference"],
                         "PERMITTED_from_official_documentation_and_already_reviewed_evidence")
        self.assertEqual(official["same_video_png_txt_frame_labels_for_manual_avi_boundaries"], "PROHIBITED")
        self.assertEqual(self.strategy["annotation_quality_requirements"]
                         ["prohibited_manual_boundary_inputs"]
                         ["same_video_official_PNG_TXT_frame_labels"], "PROHIBITED")
        self.assertEqual(official["physical_timebase"], "UNRESOLVED")

    def test_png_physical_timebase_remains_unresolved(self):
        self.assertEqual(self.strategy["official_png_annotations"]["physical_timebase"],"UNRESOLVED")
        self.assertIn("PNG_ordinal_converted_to_seconds_using_unverified_FPS",self.strategy["prohibited_mappings"])

    def test_png_to_avi_ordinal_and_filename_index_joins_prohibited(self):
        p=self.strategy["prohibited_mappings"]
        self.assertIn("AVI_frame_index_equals_PNG_ordinal",p)
        self.assertIn("AVI_frame_index_equals_parsed_PNG_filename_index",p)
        self.assertEqual(self.strategy["official_png_annotations"]["png_to_avi_frame_join"],"PROHIBITED")

    def test_similarity_alignment_and_hidden_png_boundary_transfer_prohibited(self):
        p=self.strategy["prohibited_mappings"]
        self.assertIn("AVI_PNG_similarity_based_reconstruction",p)
        self.assertIn("PNG_class_transition_used_as_hidden_AVI_boundary",p)
        self.assertEqual(self.strategy["official_png_annotations"]["avi_frame_level_ground_truth_role"],"PROHIBITED")

    def test_official_grounded_state_is_not_active_fall(self):
        semantic=self.strategy["semantic_distinctions"]
        self.assertEqual((semantic["official_png_class_0"],semantic["official_png_class_1"]),
                         ("nofall","grounded_fall_state"))
        self.assertEqual(semantic["automatic_class1_to_active_fall"],"PROHIBITED")
        self.assertIn("distinct_from_grounded",semantic["fall_transition_or_falling_motion"])

    def test_whole_fall_video_cannot_be_all_positive(self):
        self.assertEqual(self.strategy["semantic_distinctions"]["automatic_fall_video_to_positive_frames_or_windows"],"PROHIBITED")
        self.assertEqual(self.strategy["video_level_weak_supervision"]["automatic_whole_fall_video_positive_labeling"],"PROHIBITED")

    def test_route_b_selected_primary(self):
        self.assertEqual(self.strategy["selected_primary_route"],"B")
        self.assertEqual(self.strategy["routes"]["B"]["status"],"SELECTED_PRIMARY")
        self.assertIn("AVI timeline",self.strategy["routes"]["B"]["description"])

    def test_route_a_deferred_not_declared_invalid(self):
        a=self.strategy["routes"]["A"]
        self.assertEqual(a["status"],"DEFERRED_NOT_PRIMARY")
        self.assertEqual(a["future_research_status"],"not_ruled_out")
        self.assertIn("timebase is unresolved",a["reason"])

    def test_route_c_fallback_only(self):
        self.assertEqual(self.strategy["routes"]["C"]["status"],"FALLBACK")
        self.assertEqual(self.strategy["video_level_weak_supervision"]["role"],"fallback_only")

    def test_manual_ontology_and_exact_boundary_rules_undecided(self):
        a=self.strategy["future_temporal_annotation"]
        u=self.strategy["unresolved_decisions"]
        self.assertEqual(a["ontology"],"UNDECIDED")
        self.assertEqual(a["operational_boundary_definitions"],"UNDECIDED")
        self.assertEqual(u["manual_AVI_annotation_ontology"],"UNDECIDED")
        self.assertEqual(u["manual_boundary_operational_definitions"],"UNDECIDED")

    def test_window_label_rule_remains_undecided(self):
        u=self.strategy["unresolved_decisions"]
        self.assertEqual(u["window_supervision_rule"],"UNDECIDED")
        self.assertEqual(u["positive_window_overlap_threshold"],"UNDECIDED")
        self.assertEqual(self.frozen["decisions"]["temporal_supervision_protocol"],"UNDECIDED")

    def test_train_and_validation_ordered_access_policy(self):
        p=self.strategy["split_policy"]
        self.assertEqual(p["train"]["subjects"],[8,4,3,9,1,2])
        self.assertEqual(p["validation"]["subjects"],[10,5])
        self.assertEqual(p["validation"]["before_initial_Train_protocol"],"PROHIBITED")
        self.assertIn("only_after_protocol_frozen",p["validation"]["later_use"])

    def test_test_subjects_remain_sealed(self):
        test=self.strategy["split_policy"]["test"]
        self.assertEqual(test["subjects"],[6,7])
        self.assertEqual(test["current_status"],"SEALED")
        self.assertEqual(test["current_annotation"],"PROHIBITED")
        self.assertEqual(test["current_video_pose_png_annotation_inspection"],"PROHIBITED")
        self.assertEqual(test["future_annotator_blinding"],"blind_to_model_outputs")

    def test_png_pilot_remains_nonprimary_and_unpromoted(self):
        pilot=self.strategy["png_native_pose_pilot"]
        self.assertEqual(pilot["source_commit"],self.strategy["parent_commit"])
        self.assertEqual(pilot["role"],"feasibility_evidence_non_primary_experimental_source")
        self.assertEqual(pilot["promotion_to_primary_training_source"],"NOT_AUTHORIZED")
        self.assertEqual(pilot["full_Train_PNG_extraction"],"NOT_AUTHORIZED")

    def test_quality_requirements_preserve_original_sources_and_records(self):
        q=self.strategy["annotation_quality_requirements"]
        self.assertEqual(q["visual_evidence_source"],"original_source_AVI_visual_pixels_only")
        self.assertEqual(q["model_predictions_as_annotation_input"],"PROHIBITED")
        self.assertEqual(q["future_Test_performance_as_annotation_input"],"PROHIBITED")
        self.assertEqual(q["source_video_edit_or_reencode"],"PROHIBITED")
        self.assertEqual(q["independent_multi_annotator_pilot"],"frozen_subset_of_Train_fall_videos_before_full_Train_annotation")
        self.assertEqual(q["agreement_unit"],"video_or_event_not_independent_frames")
        self.assertEqual(q["adjudication"],"preserve_original_annotator_records")
        self.assertEqual(q["required_agreement_threshold"],"UNDECIDED")

    def test_next_stage_is_not_started_and_lists_required_design(self):
        next_stage=self.strategy["next_stage"]
        self.assertEqual((next_stage["identity"],next_stage["status"]),("3.2e","NOT_STARTED"))
        self.assertEqual(len(next_stage["required_design_topics"]),11)
        self.assertIn("Gate_criteria",next_stage["required_design_topics"])
        self.assertIn("annotation_schema",next_stage["required_design_topics"])

    def test_stage32d_cannot_emit_annotations_windows_or_model_ready_data(self):
        e=self.strategy["execution"]
        self.assertEqual(e["implementation_status"],"strategy_record_only")
        self.assertIs(e["annotation_enabled"],False)
        self.assertIs(e["derived_data_build_enabled"],False)
        self.assertIs(e["window_construction_enabled"],False)
        self.assertIs(e["model_training_enabled"],False)
        self.assertEqual(self.strategy["future_temporal_annotation"]["status"],"NOT_IMPLEMENTED")
        self.assertFalse((ROOT/"artifacts/preprocessing/stage32d_temporal_supervision_strategy_v1").exists())

    def test_frozen_stage3_contract_decisions_remain_unmodified(self):
        self.assertTrue(self.frozen["decisions"])
        self.assertEqual(set(self.frozen["decisions"].values()),{"UNDECIDED"})


if __name__=="__main__":
    unittest.main()
