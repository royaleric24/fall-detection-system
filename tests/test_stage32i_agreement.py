"""Synthetic-only Stage 3.2i tests. Genuine data reads fail immediately."""
from __future__ import annotations

import copy
import csv
import io
import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ml.preprocessing import stage32i_agreement as a
from ml.preprocessing.annotation_execution import FinalizedBlob, PilotSpec, raw_freeze_record, check_completion
from ml.preprocessing.annotation_tool import TOOL_VERSION
from ml.preprocessing.manual_annotation_contract import AnnotationContractError


def boundary(e=10, p=15, l=20, status="observed"):
    result = {"status": status, "reason_flags": ["other"] if status == "unjudgeable" else []}
    if status == "observed":
        result.update(zip(a.COORDINATES, (e, p, l)))
    return result


def fixture_spec():
    order = tuple(f"clip-SYNTHETIC-{i:02}" for i in range(12))
    rows = {key: {"neutral_clip_id": key, "source_video": f"SYNTHETIC_ONLY/{key}.avi",
                  "subject_id": 1, "split": "train", "source_avi_sha256": "0" * 64,
                  "expected_frame_count": 100, "source_fps": 20,
                  "source_fps_provenance": "frozen_Stage_2_AVI_timeline"} for key in order}
    return PilotSpec(rows, order, "1" * 64, "2" * 64)


def record(owner="A01", index=0, event="fall_observed", offset=0):
    spec = fixture_spec()
    boundaries = {name: boundary(frame-2+offset, frame+offset, frame+2+offset)
                  for name, frame in zip(a.BOUNDARIES, (10, 35, 70))}
    if event == "no_fall_observed":
        boundaries = {name: boundary(status="not_applicable") for name in a.BOUNDARIES}
    if event == "uncertain":
        boundaries = {name: boundary(status="unjudgeable" if i < 2 else "not_applicable")
                      for i, name in enumerate(a.BOUNDARIES)}
    return {"annotation_record_id": f"SYNTHETIC-{owner}-{index:02}", "protocol_version": "stage32e_v1",
            "source": spec.source_fields(spec.presentation_order[index]),
            "annotator": {"pseudonymous_id": owner, "session_id": f"{a.GENUINE_RUN}-{owner}-{index:032x}"},
            "creation": {"created_at_utc": "2026-01-01T00:00:00Z", "annotation_run_id": spec.version,
                         "tool_version": TOOL_VERSION},
            "annotator_note": "SYNTHETIC TEST ONLY - NOT GENUINE ANNOTATION",
            "event_presence": event, "event_presence_reason_flags": ["other"] if event == "uncertain" else [],
            "boundaries": boundaries}


def pair(index=0, first="fall_observed", second="fall_observed", offset=0):
    r1, r2 = record("A01", index, first), record("A02", index, second, offset)
    return a.AnnotationPair(r1["source"]["neutral_clip_id"], r1, r2,
                            f"A01/finalized/{r1['annotation_record_id']}.finalized.json",
                            f"A02/finalized/{r2['annotation_record_id']}.finalized.json")


def full_files():
    files = {owner: [] for owner in a.ANNOTATORS}
    for owner in a.ANNOTATORS:
        for i in range(12):
            value = record(owner, i, a.EVENTS[i % 3], int(owner == "A02"))
            path = f"{owner}/finalized/{value['annotation_record_id']}.finalized.json"
            files[owner].append(FinalizedBlob(path, json.dumps({"state": "FINALIZED", "record": value}).encode()))
    return files


class SyntheticOnlyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="stage32i-SYNTHETIC-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        original_open = Path.open
        original_iterdir = Path.iterdir
        real_data = a.REPO / "data"
        def deny_real(path, *args, **kwargs):
            if path.resolve().is_relative_to(real_data):
                raise AssertionError("Genuine data access prohibited in synthetic tests")
            return original_open(path, *args, **kwargs)
        def deny_walk(path):
            if path.resolve().is_relative_to(real_data):
                raise AssertionError("Genuine data traversal prohibited in synthetic tests")
            return original_iterdir(path)
        p = patch.object(Path, "open", deny_real)
        p.start()
        self.addCleanup(p.stop)
        p = patch.object(Path, "iterdir", deny_walk)
        p.start()
        self.addCleanup(p.stop)


class EventTests(SyntheticOnlyTests):
    def test_perfect_three_category_nominal_agreement(self):
        result = a.event_metrics([(x, x) for x in a.EVENTS])
        self.assertEqual(result["confusion_matrix"], [[1,0,0],[0,1,0],[0,0,1]])
        self.assertEqual(result["exact_agreement_count"], 3)
        self.assertEqual(result["cohen_kappa"]["value"], 1)
        self.assertEqual(result["gwet_ac1"]["value"], 1)

    def test_complete_disagreement(self):
        result = a.event_metrics(list(zip(a.EVENTS, (*a.EVENTS[1:], a.EVENTS[0]))))
        self.assertEqual(result["exact_agreement_rate"], 0)
        self.assertEqual(result["cohen_kappa"]["value"], -.5)
        self.assertEqual(result["gwet_ac1"]["value"], -.5)

    def test_uncertain_has_no_partial_credit(self):
        result = a.event_metrics([("uncertain", "fall_observed"), ("uncertain", "uncertain")])
        self.assertEqual(result["exact_agreement_rate"], .5)
        self.assertEqual(result["confusion_matrix"][2], [1,0,1])

    def test_asymmetric_marginals_and_manual_ac1(self):
        # Synthetic table: [[2,1,0],[0,1,0],[1,0,1]], N=6.
        # P_o=2/3; P_e(kappa)=13/36 -> kappa=11/23.
        # Pooled p=(1/2,1/4,1/4); P_e(AC1)=5/16 -> AC1=17/33.
        result = a.event_metrics([("fall_observed","fall_observed")]*2 + [
            ("fall_observed","no_fall_observed"), ("no_fall_observed","no_fall_observed"),
            ("uncertain","fall_observed"), ("uncertain","uncertain")])
        self.assertAlmostEqual(result["cohen_kappa"]["value"], 11/23)
        self.assertAlmostEqual(result["gwet_ac1"]["chance_agreement"], 5/16)
        self.assertAlmostEqual(result["gwet_ac1"]["value"], 17/33)
        self.assertEqual(result["annotator_marginals"]["A01"]["uncertain"]["count"], 2)
        self.assertEqual(result["annotator_marginals"]["A02"]["uncertain"]["count"], 1)
        self.assertEqual(result["annotator_marginals"]["A01"]["uncertain"]["denominator"], 6)

    def test_degenerate_kappa_is_na_with_reason(self):
        result = a.event_metrics([("fall_observed","fall_observed")]*4)
        self.assertIsNone(result["cohen_kappa"]["value"])
        self.assertEqual(result["cohen_kappa"]["reason"], "undefined_due_to_degenerate_marginals")
        self.assertEqual(result["gwet_ac1"]["value"], 1)

    def test_fixed_k_three_with_unused_category(self):
        result = a.event_metrics([("fall_observed","fall_observed"), ("fall_observed","no_fall_observed")])
        self.assertEqual(result["categories"], list(a.EVENTS))
        self.assertEqual(result["confusion_matrix"], [[1,1,0],[0,0,0],[0,0,0]])
        # pi=(3/4,1/4,0), chance=3/16, AC1=5/13 (K must remain 3).
        self.assertAlmostEqual(result["gwet_ac1"]["value"], 5/13)
        self.assertEqual(result["gwet_ac1"]["K"], 3)

    def test_unknown_event_fails(self):
        with self.assertRaises(a.AgreementError):
            a.event_metrics([("new_category", "uncertain")])

    def test_empty_event_denominator(self):
        result = a.event_metrics([])
        self.assertEqual(result["denominator"], 0)
        self.assertIsNone(result["exact_agreement_rate"])
        self.assertIsNone(result["cohen_kappa"]["value"])
        self.assertEqual(result["gwet_ac1"]["reason"], "zero_denominator")


class TemporalTests(SyntheticOnlyTests):
    def test_all_statuses_and_availability_classes(self):
        pairs = [("observed","observed"),("left_censored","left_censored"),
                 ("observed","right_censored"),("not_observed","unjudgeable"),
                 ("not_applicable","not_applicable")]
        result = a.status_metrics(pairs)
        self.assertEqual(result["categories"], list(a.STATUSES))
        self.assertEqual([result[k] for k in a.AVAILABILITY], [1,2,1,1])
        self.assertEqual(sum(map(sum,result["confusion_matrix"])),5)
        self.assertEqual(result["status_exact_agreement_count"],3)
        self.assertEqual(result["confusion_matrix"][3][4],1)

    def test_noncomparable_reasons_and_all_metrics_na(self):
        cases = [("left_censored","observed","A01_NONOBSERVED_ONLY"),
                 ("observed","right_censored","A02_NONOBSERVED_ONLY"),
                 ("left_censored","left_censored","BOTH_NONOBSERVED_SAME_STATUS"),
                 ("not_observed","not_applicable","BOTH_NONOBSERVED_DIFFERENT_STATUS")]
        for first,second,reason in cases:
            with self.subTest(reason=reason):
                result=a.temporal_metrics(boundary(status=first),boundary(status=second))
                self.assertFalse(result["temporal_comparable"])
                self.assertEqual(result["non_comparable_reason"],reason)
                self.assertTrue(all(result[k] is None for k in a.TEMPORAL_FIELDS))

    def test_positive_negative_zero_sign_and_ms(self):
        for p,d in ((18,3),(12,-3),(15,0)):
            with self.subTest(p=p):
                result=a.temporal_metrics(boundary(),boundary(p=p))
                self.assertEqual(result["preferred_signed_difference_frames"],d)
                self.assertEqual(result["preferred_signed_difference_ms"],d*50)
                self.assertEqual(result["preferred_absolute_difference_ms"],abs(d)*50)
                self.assertEqual(result["preferred_exact_match"],d==0)

    def test_exact_interval_equality(self):
        result=a.temporal_metrics(boundary(),boundary())
        self.assertEqual(result["A01_interval_width_frames"],11)
        self.assertEqual(result["A01_interval_width_ms"],550)
        self.assertEqual(result["intersection_width_frames"],11)
        self.assertEqual(result["union_width_frames"],11)
        self.assertEqual(result["interval_iou"],1)
        self.assertTrue(result["interval_exact_match"])
        self.assertTrue(result["mutual_preferred_containment"])
        self.assertIsNone(result[a.GAP])

    def test_partial_overlap_inclusive_off_by_one(self):
        result=a.temporal_metrics(boundary(10,15,20),boundary(20,23,25))
        self.assertEqual(result["intersection_width_frames"],1)
        self.assertEqual(result["union_width_frames"],16)
        self.assertEqual(result["interval_iou"],1/16)
        self.assertEqual(result["overlap_coefficient"],1/6)
        self.assertIsNone(result[a.GAP])

    def test_full_containment(self):
        result=a.temporal_metrics(boundary(10,15,20),boundary(12,15,17))
        self.assertEqual(result["interval_iou"],6/11)
        self.assertEqual(result["overlap_coefficient"],1)
        self.assertFalse(result["interval_exact_match"])

    def test_single_frame_interval(self):
        result=a.temporal_metrics(boundary(0,0,0),boundary(0,0,0),frame_count=1)
        self.assertEqual(result["union_width_frames"],1)
        self.assertEqual(result["A01_interval_width_ms"],50)
        self.assertEqual(result["interval_iou"],1)

    def test_adjacent_and_separated_blank_gaps(self):
        for start,gap in ((21,0),(25,4)):
            with self.subTest(start=start):
                for x,y in ((boundary(10,15,20),boundary(start,start,30)),
                            (boundary(start,start,30),boundary(10,15,20))):
                    result=a.temporal_metrics(x,y)
                    self.assertEqual(result[a.GAP],gap)
                    self.assertEqual(result["interval_iou"],0)
                    self.assertEqual(result["overlap_coefficient"],0)
                    self.assertFalse(result["interval_overlap_any"])

    def test_one_direction_and_inclusive_containment(self):
        result=a.temporal_metrics(boundary(10,20,20),boundary(20,25,30))
        self.assertTrue(result["A01_preferred_in_A02_interval"])
        self.assertFalse(result["A02_preferred_in_A01_interval"])
        self.assertFalse(result["mutual_preferred_containment"])

    def test_invalid_order_types_bounds_and_status(self):
        bad=[boundary(11,10,20),boundary(10,21,20),boundary(-1,2,3),boundary(0,True,3),
             boundary(1,2.0,3),boundary(1,2,100),boundary(status="other")]
        for value in bad:
            with self.subTest(value=value),self.assertRaises(a.AgreementError):
                a.temporal_metrics(value,boundary(),frame_count=100)

    def test_nonobserved_fields_must_be_absent_even_null(self):
        for field in (*a.COORDINATES,"preferred_timestamp_ms"):
            for value in (None,0):
                b=boundary(status="left_censored")
                b[field]=value
                with self.subTest(field=field,value=value),self.assertRaises(a.AgreementError):
                    a.temporal_metrics(b,boundary())

    def test_type7_quartiles(self):
        self.assertEqual(a.quantile_type7([0,1,4,9],.25),.75)
        self.assertEqual(a.quantile_type7([0,1,4,9],.75),5.25)
        self.assertEqual(a.quantile_type7([7],.25),7)
        self.assertIsNone(a.quantile_type7([],.5))

    def test_zero_comparable_summary(self):
        temporal,interval=a.boundary_summaries([])
        self.assertEqual(temporal["preferred_exact_match_count"],0)
        self.assertIsNone(temporal["preferred_exact_match_rate"])
        self.assertIsNone(temporal["median_absolute_difference_frames"])
        self.assertEqual(interval["blank_gap_denominator"],0)
        self.assertIsNone(interval[f"median_{a.GAP}_among_disjoint"])
        self.assertIsNone(interval["any_overlap_rate"])

    def test_summary_denominators_ignore_noncomparable_not_zero_gap(self):
        rows=[]
        for other in (boundary(21,21,30),boundary(25,25,30),boundary(),boundary(status="left_censored")):
            rows.append({"boundary_type":"grounded_start",**a.temporal_metrics(boundary(),other)})
        temporal,interval=a.boundary_summaries(rows)
        self.assertEqual(temporal["preferred_exact_match_denominator"],3)
        self.assertEqual(interval["interval_denominator"],3)
        self.assertEqual(interval["containment_denominator"],3)
        self.assertEqual(interval["blank_gap_denominator"],2)
        self.assertEqual(interval[f"median_{a.GAP}_among_disjoint"],2)
        self.assertEqual(interval[f"max_{a.GAP}_among_disjoint"],4)
        self.assertEqual(interval["any_overlap_rate"],1/3)

    def test_preferred_summary_signed_and_absolute_distributions(self):
        rows=[{"boundary_type":"grounded_start",**a.temporal_metrics(boundary(0,15,50),boundary(0,p,50))}
              for p in (6,11,16,15)]
        temporal,interval=a.boundary_summaries(rows)
        expected={"preferred_exact_match_count":1,"preferred_exact_match_rate":.25,
                  "median_signed_difference_frames":-2,"median_signed_difference_ms":-100,
                  "median_absolute_difference_frames":2.5,"median_absolute_difference_ms":125,
                  "Q1_absolute_difference_frames":.75,"Q3_absolute_difference_frames":5.25,
                  "IQR_absolute_difference_frames":4.5,"min_signed_difference_frames":-9,
                  "max_signed_difference_frames":1,"max_absolute_difference_frames":9}
        for key,value in expected.items():
            self.assertEqual(temporal[key],value,key)
        self.assertEqual(interval["blank_gap_denominator"],0)
        self.assertIsNone(interval[f"median_{a.GAP}_among_disjoint"])
        self.assertIsNone(interval[f"max_{a.GAP}_among_disjoint"])

    def test_boundary_pooling_rejected(self):
        rows=[{"boundary_type":name,**a.temporal_metrics(boundary(),boundary())} for name in a.BOUNDARIES]
        with self.assertRaises(a.AgreementError):
            a.boundary_summaries(rows)


class StructureTests(SyntheticOnlyTests):
    def test_twelve_pairs_sorted_by_identity_not_filename_or_input_order(self):
        files=full_files()
        for values in files.values():
            values.reverse()
        pairs=a.construct_pairs(fixture_spec(),files)
        self.assertEqual([p.annotation_unit_id for p in pairs],list(fixture_spec().presentation_order))
        self.assertEqual(len(pairs),12)

    def test_missing_and_duplicate_pair_rejected(self):
        for kind in ("missing","duplicate"):
            files=full_files()
            if kind=="missing":
                files["A02"].pop()
            else:
                files["A01"][1]=files["A01"][0]
            with self.subTest(kind=kind),self.assertRaises(a.AgreementError):
                a.construct_pairs(fixture_spec(),files)

    def test_nonfinalized_unknown_vocabulary_and_source_rejected(self):
        for mutation in (
            lambda w:w.update(state="DRAFT"),
            lambda w:w["record"].update(event_presence="unknown"),
            lambda w:w["record"]["boundaries"]["grounded_start"].update(status="unknown"),
            lambda w:w["record"]["source"].update(subject_id=6,split="test"),
            lambda w:w["record"]["source"].update(source_fps=30),
        ):
            files=full_files()
            b=files["A01"][0]
            wrapper=json.loads(b.content)
            mutation(wrapper)
            files["A01"][0]=FinalizedBlob(b.relative_path,json.dumps(wrapper).encode())
            with self.assertRaises(a.AgreementError):
                a.construct_pairs(fixture_spec(),files)

    def test_duplicate_direct_pair_and_bad_owner_rejected(self):
        with self.assertRaises(a.AgreementError):
            a.analyze_pairs([pair(),pair()])
        p=pair()
        p.a02["annotator"]["pseudonymous_id"]="A01"
        with self.assertRaises(a.AgreementError):
            a.analyze_pairs([p])

    def test_analysis_preserves_inputs_and_independent_boundary_denominators(self):
        pairs=[pair(0),pair(1,second="uncertain"),pair(2,first="no_fall_observed",second="no_fall_observed")]
        pairs[0].a02["boundaries"]["recovery_start"]=boundary(status="not_observed")
        before=copy.deepcopy(pairs)
        result=a.analyze_pairs(pairs)
        self.assertEqual(pairs,before)
        self.assertEqual(result["event"]["denominator"],3)
        self.assertEqual(result["temporal"]["grounded_start"]["n_temporally_comparable"],1)
        self.assertEqual(result["temporal"]["recovery_start"]["n_temporally_comparable"],0)
        self.assertEqual(len(result["boundary_rows"]),9)
        self.assertNotIn("overall_agreement_score",result)

    def test_denominator_audit_rejects_tampering(self):
        for scope,key in (("temporal","preferred_exact_match_denominator"),("interval","containment_denominator"),
                          ("interval","blank_gap_denominator"),("status","denominator")):
            result=a.analyze_pairs([pair()])
            result[scope]["grounded_start"][key]+=1
            with self.subTest(scope=scope,key=key),self.assertRaises(a.AgreementError):
                a.audit_denominators(result,1)

    def test_objective_discrepancy_flags_only(self):
        result=a.analyze_pairs([pair(0,offset=5),pair(1,second="uncertain")])
        actual={flag for row in result["discrepancies"] for flag in row["discrepancy_flags"].split("|")}
        self.assertEqual(actual,{"EVENT_PRESENCE_MISMATCH","BOUNDARY_STATUS_MISMATCH","PREFERRED_FRAME_MISMATCH","INTERVAL_DISJOINT"})
        self.assertTrue(all(a.GAP in row for row in result["discrepancies"]))


class OutputTests(SyntheticOnlyTests):
    def manifest(self,n=2):
        return {"artifact_kind":"synthetic_test_only","expected_pair_count":n,"observed_pair_count":n,
                "analysis_code_commit":"b"*40,"source_annotation_runtime_commit":a.SOURCE_RUNTIME_COMMIT,
                "completion_gate":"HOLD"}

    def test_all_required_outputs_and_csv_na_zero(self):
        result=a.analyze_pairs([pair(0),pair(1,first="uncertain",second="uncertain")])
        output=self.root/'SYNTHETIC_ONLY'
        a.write_artifacts(output,result,self.manifest(),{"input_integrity":"SYNTHETIC_TEST_ONLY"})
        expected={"stage32i_analysis_manifest.json","stage32i_input_validation.json","paired_annotations.csv",
                  "event_presence_confusion_matrix.csv","event_presence_metrics.json","boundary_status_pairs.csv",
                  "boundary_status_metrics.csv","boundary_temporal_comparable.csv","boundary_temporal_summary.csv",
                  "boundary_interval_summary.csv","discrepancy_inventory.csv","stage32i_report.md",
                  *(f"boundary_status_confusion_{name}.csv" for name in a.BOUNDARIES)}
        self.assertEqual({p.name for p in output.iterdir()},expected)
        with (output/'boundary_status_pairs.csv').open() as stream:
            rows=list(csv.DictReader(stream))
        self.assertEqual(rows[0]["preferred_signed_difference_frames"],"0")
        self.assertEqual(rows[3]["preferred_signed_difference_frames"],"NA")
        self.assertEqual(rows[3]["temporal_comparable"],"false")
        self.assertEqual(rows[0][a.GAP],"NA")
        self.assertIn("SYNTHETIC TEST ONLY",(output/'stage32i_report.md').read_text())
        with self.assertRaises(a.AgreementError):
            a.write_artifacts(output,result,self.manifest(),{})

    def test_empty_comparable_csv_has_headers_and_empty_discrepancy(self):
        output=self.root/'SYNTHETIC_EMPTY'
        result=a.analyze_pairs([pair(0,first="no_fall_observed",second="no_fall_observed")])
        a.write_artifacts(output,result,self.manifest(1),{})
        with (output/'boundary_temporal_comparable.csv').open() as stream:
            reader=csv.DictReader(stream)
            self.assertIn(a.GAP,reader.fieldnames)
            self.assertEqual(list(reader),[])
        self.assertEqual(len((output/'discrepancy_inventory.csv').read_text().splitlines()),1)

    def test_serialization_is_byte_deterministic(self):
        pairs=[pair(1,offset=1),pair(0)]
        for name,values in (("one",pairs),("two",list(reversed(pairs)))):
            a.write_artifacts(self.root/name,a.analyze_pairs(values),self.manifest(),{})
        for path in (self.root/'one').iterdir():
            self.assertEqual(path.read_bytes(),(self.root/'two'/path.name).read_bytes())

    def test_json_null_for_undefined_kappa(self):
        output=self.root/'SYNTHETIC_NA'
        a.write_artifacts(output,a.analyze_pairs([pair()]),self.manifest(1),{})
        value=json.loads((output/'event_presence_metrics.json').read_text())
        self.assertIsNone(value["cohen_kappa"]["value"])
        self.assertEqual(value["exact_agreement_rate"],1)

    def test_output_symlink_rejected_before_write(self):
        link=self.root/'link'
        link.symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(a.AgreementError):
            a.write_artifacts(link/'new',a.analyze_pairs([pair()]),self.manifest(1),{})
        self.assertFalse((self.root/'new').exists())

    def test_empty_pure_sample_serializes_without_fake_rates(self):
        result=a.analyze_pairs([])
        a.write_artifacts(self.root/'SYNTHETIC_ZERO',result,self.manifest(0),{})
        self.assertIsNone(result["event"]["exact_agreement_rate"])
        self.assertIsNone(result["interval"]["recovery_start"]["any_overlap_rate"])


    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Optional Matplotlib is not installed")
    def test_synthetic_png_outputs_and_zero_comparable_boundary(self):
        for label,pairs,expected in (("observed",[pair(0),pair(1,offset=5)],10),
                                     ("none",[pair(0,first="uncertain",second="uncertain")],4)):
            output=self.root/label
            output.mkdir()
            version=a.write_plots(output,a.analyze_pairs(pairs),synthetic=True)
            self.assertTrue(version)
            images=list(output.glob('*.png'))
            self.assertEqual(len(images),expected)
            for image in images:
                self.assertTrue(image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            with self.assertRaises(a.AgreementError):
                a.write_plots(output,a.analyze_pairs(pairs),synthetic=True)

    def test_dataset_output_namespace_rejected(self):
        with patch.object(a,"REPO",self.root):
            output=self.root/'temporary/../data/forbidden'
            with self.assertRaises(a.AgreementError):
                a.write_artifacts(output,a.analyze_pairs([pair()]),self.manifest(1),{})
            self.assertFalse((self.root/'data').exists())


class LoaderRunnerTests(SyntheticOnlyTests):
    def setup_loader(self):
        root=self.root/'SYNTHETIC_pilot'
        files=full_files()
        spec=fixture_spec()
        for owner,blobs in files.items():
            folder=root/owner/'finalized'
            folder.mkdir(parents=True)
            for blob in blobs:
                (root/blob.relative_path).write_bytes(blob.content)
        active={"run_id":a.GENUINE_RUN,"record_annotation_run_id":a.PILOT_VERSION}
        control=root/'_control'
        (control/'run_initialization').mkdir(parents=True)
        active_path=control/'run_initialization/active_run.json'
        active_path.write_text(json.dumps(active))
        binding={"version":"stage32h_runtime_binding_v1","run_id":a.GENUINE_RUN,
                 "parent_commit":a.EXECUTION_COMMIT,"runtime_commit":a.SOURCE_RUNTIME_COMMIT,
                 "active_run_sha256":a._sha(active_path.read_bytes()),"runtime_requirements":a.RUNTIME_REQUIREMENTS}
        binding_path=control/'run_initialization/runtime_binding.json'
        binding_path.write_text(json.dumps(binding))
        # Construct synthetic freeze metadata in memory. Never call write_raw_freeze_new.
        freeze=raw_freeze_record(spec,check_completion(spec,files),
                                frozen_at_utc="2026-01-01T00:00:00Z",code_commit=a.SOURCE_RUNTIME_COMMIT)
        (control/'raw_freezes').mkdir()
        freeze_path=control/'raw_freezes/raw_annotation_freeze.json'
        freeze_path.write_text(json.dumps(freeze))
        for target,name,value in ((a,"REAL_ROOT",root),(a,"RAW_FREEZE_SHA256",a._sha(freeze_path.read_bytes()))):
            p=patch.object(target,name,value)
            p.start()
            self.addCleanup(p.stop)
        for p in (patch.object(a,"_verify_frozen_sources"),patch.object(a,"read_active",return_value=active),
                  patch.object(a.PilotSpec,"from_frozen",return_value=spec)):
            p.start()
            self.addCleanup(p.stop)
        return root,files,freeze_path,binding_path

    def test_loader_preflight_cannot_invoke_metrics(self):
        self.setup_loader()
        with patch.object(a,"analyze_pairs",side_effect=AssertionError("NO METRICS")), \
             patch.object(a,"event_metrics",side_effect=AssertionError("NO METRICS")), \
             patch.object(a,"temporal_metrics",side_effect=AssertionError("NO METRICS")):
            output=io.StringIO()
            with redirect_stdout(output):
                a.main(["preflight"])
            report=json.loads(output.getvalue())
        self.assertEqual(report["canonical_pair_count"],12)
        self.assertEqual(report["finalized_counts"],{"A01":12,"A02":12})
        self.assertFalse(report["agreement_calculated_by_preflight"])
        self.assertNotIn("event_presence",output.getvalue())

    def test_freeze_digest_rejected_before_record_collection(self):
        _,_,freeze_path,_=self.setup_loader()
        freeze_path.write_bytes(freeze_path.read_bytes()+b' ')
        with patch.object(a,"collect_finalized",side_effect=AssertionError("Must reject before records")):
            with self.assertRaisesRegex(a.AgreementError,"manifest SHA"):
                a.load_genuine()

    def test_source_mutation_rejected(self):
        root,files,_,_=self.setup_loader()
        path=root/files["A01"][0].relative_path
        path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaises(ValueError):
            a.load_genuine()

    def test_wrong_binding_rejected(self):
        _,_,_,path=self.setup_loader()
        binding=json.loads(path.read_text())
        binding["runtime_commit"]="0"*40
        path.write_text(json.dumps(binding))
        with self.assertRaisesRegex(a.AgreementError,"binding"):
            a.load_genuine()

    def test_wrong_run_session_rejected_even_with_matching_fixture_hash(self):
        root,files,freeze_path,_=self.setup_loader()
        path=root/files["A01"][0].relative_path
        wrapper=json.loads(path.read_bytes())
        wrapper["record"]["annotator"]["session_id"]="other-run"
        path.write_text(json.dumps(wrapper))
        freeze=json.loads(freeze_path.read_bytes())
        freeze["finalized_file_sha256"][files["A01"][0].relative_path]=a._sha(path.read_bytes())
        freeze_path.write_text(json.dumps(freeze))
        with patch.object(a,"RAW_FREEZE_SHA256",a._sha(freeze_path.read_bytes())):
            with self.assertRaisesRegex(a.AgreementError,"Session/run"):
                a.load_genuine()

    def test_symlink_source_denied_before_collection(self):
        root,files,_,_=self.setup_loader()
        path=root/files["A01"][0].relative_path
        content=path.read_bytes()
        path.unlink()
        target=self.root/'SYNTHETIC_elsewhere.json'
        target.write_bytes(content)
        path.symlink_to(target)
        with self.assertRaises(ValueError):
            a.load_genuine()

    def test_loader_has_no_arbitrary_input_root_argument(self):
        with self.assertRaises(TypeError):
            a.load_genuine(self.root/'Subject.6')

    def test_manifest_rejects_wrong_commit_and_non_utc_timestamp(self):
        self.setup_loader()
        loaded=a.load_genuine()
        base={"code_commit":"b"*40,"run_id":"stage32i-SYNTHETIC","timestamp":"2026-01-01T00:00:00Z","output":self.root/'SYNTHETIC_output'}
        for key,value in (("code_commit",a.SOURCE_RUNTIME_COMMIT),("timestamp","2026-01-01Z"),
                          ("timestamp","2026-01-01T00:00:00"),("run_id","stage32i-../escape")):
            with self.subTest(key=key,value=value),self.assertRaises(a.AgreementError):
                a.analysis_manifest(loaded,**{**base,key:value})

    def test_runtime_commit_cannot_be_analysis_commit(self):
        with self.assertRaises(a.AgreementError):
            a.verify_analysis_commit(a.SOURCE_RUNTIME_COMMIT)
        with self.assertRaises(a.AgreementError):
            a.verify_analysis_commit(a.SPEC_COMMIT)

    def test_run_rejected_before_loader_when_commit_invalid(self):
        with patch.object(a,"load_genuine",side_effect=AssertionError("No source read allowed")):
            with self.assertRaises(a.AgreementError):
                a.run_genuine(code_commit=a.SOURCE_RUNTIME_COMMIT,run_id="stage32i-SYNTHETIC",timestamp="2026-01-01T00:00:00Z")

    def synthetic_manifest(self,loaded,**kwargs):
        result=self.original_manifest(loaded,**kwargs)
        result.update(artifact_kind="synthetic_test_only",source_genuine_run_id="SYNTHETIC_TEST_ONLY")
        return result

    def runner_context(self):
        self.original_manifest=a.analysis_manifest
        for p in (patch.object(a,"verify_analysis_commit"),patch.object(a,"OUTPUT_ROOT",self.root/'SYNTHETIC_outputs'),
                  patch.object(a,"analysis_manifest",side_effect=self.synthetic_manifest)):
            p.start()
            self.addCleanup(p.stop)

    def test_end_to_end_runner_uses_only_synthetic_loader_and_postverifies(self):
        self.setup_loader()
        self.runner_context()
        with patch.object(a,"load_genuine",wraps=a.load_genuine) as loader:
            output=a.run_genuine(code_commit="b"*40,run_id="stage32i-SYNTHETIC",timestamp="2026-01-01T00:00:00Z")
        self.assertEqual(loader.call_count,3)
        manifest=json.loads((output/'stage32i_analysis_manifest.json').read_text())
        self.assertEqual(manifest["completion_gate"],"PASS")
        self.assertEqual(manifest["artifact_kind"],"synthetic_test_only")
        self.assertNotEqual(manifest["analysis_code_commit"],manifest["source_annotation_runtime_commit"])
        self.assertEqual(len(manifest["analysis_output_paths"]),15)
        self.assertEqual(json.loads((output/'stage32i_input_validation.json').read_text())["raw_integrity_after_analysis"],"PASS")

    def test_postcheck_failure_never_leaves_a_pass_claim(self):
        self.setup_loader()
        loaded=a.load_genuine()
        self.runner_context()
        for failure_at in (2,3):
            calls=[loaded,a.AgreementError("synthetic post-check failure")] if failure_at==2 else [loaded,loaded,a.AgreementError("synthetic final failure")]
            with patch.object(a,"load_genuine",side_effect=calls):
                with self.assertRaises(a.AgreementError):
                    a.run_genuine(code_commit="b"*40,run_id=f"stage32i-SYNTHETIC-{failure_at}",timestamp="2026-01-01T00:00:00Z")
            output=self.root/f'SYNTHETIC_outputs/stage32i-SYNTHETIC-{failure_at}'
            self.assertEqual(json.loads((output/'stage32i_analysis_manifest.json').read_text())["completion_gate"],"HOLD")
            self.assertEqual(json.loads((output/'stage32i_input_validation.json').read_text())["raw_integrity_after_analysis"],"FAIL")
            self.assertIn('Stage 3.2i Analysis Execution: HOLD',(output/'stage32i_report.md').read_text())


if __name__ == "__main__":
    unittest.main()
