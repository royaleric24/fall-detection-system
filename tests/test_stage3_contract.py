"""Stage 3.0 metadata/access regression tests; never load pose or video data."""

import copy
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.datasets.assign_dataset_split import subject_assignments
from ml.preprocessing import contract as c


class Stage3ContractTests(unittest.TestCase):
    def setUp(self):
        self.config = c.read_contract()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name).resolve()
        source = self.config["source"]
        # Copy metadata only into fixtures. Real Stage 2 files are never written.
        for name in (source["split_config"], "artifacts/dataset_inspection/inventory.csv",
                     source["evidence_root"] + "/extraction_run.json",
                     source["evidence_root"] + "/manifest.csv"):
            dest = self.repo / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((c.REPO / name).read_bytes())
        self.config_path = self.repo / "contract.json"
        self.save_config()

    def save_config(self):
        self.config_path.write_text(json.dumps(self.config))

    def select(self, *args, **kwargs):
        return c.select_sources(*args, repo=self.repo, config_path=self.config_path, **kwargs)

    def fake_npz_existence(self):
        original = Path.is_file
        return patch.object(Path, "is_file", lambda p: p.suffix == ".npz" or original(p))

    def test_exact_split_disjoint_complete(self):
        config = json.loads((c.REPO / "configs/dataset_split.json").read_text())
        expected = {"train": [8, 4, 3, 9, 1, 2], "validation": [10, 5], "test": [6, 7]}
        self.assertEqual({s: config[s] for s in expected}, expected)
        self.assertEqual({s: list(ids) for s, ids in c.FROZEN_SUBJECTS.items()}, expected)
        sets = [set(ids) for ids in expected.values()]
        self.assertEqual(set.union(*sets), set(range(1, 11)))
        self.assertTrue(all(not a & b for i, a in enumerate(sets) for b in sets[i + 1:]))
        self.assertEqual(len(subject_assignments(config)), 10)

    def test_default_train_explicit_validation_and_no_pose_reads(self):
        original_open = Path.open

        def metadata_only(path, *args, **kwargs):
            self.assertNotIn(path.suffix, (".npz", ".avi"))
            self.assertNotIn(path.name, ("summary.json", "FULL_EXTRACTION_REPORT.md"))
            return original_open(path, *args, **kwargs)

        with self.fake_npz_existence(), patch.object(Path, "open", metadata_only):
            train = self.select()
            validation = self.select(("validation",))
            both = self.select(("train", "validation"))
        self.assertEqual((len(train), len(validation), len(both)), (60, 20, 80))
        self.assertEqual({r.subject_id for r in train}, {8, 4, 3, 9, 1, 2})
        self.assertEqual({r.subject_id for r in validation}, {10, 5})
        self.assertTrue(all(r.split == "train" for r in train))
        self.assertFalse(any(hasattr(r, "pose_missing_count") for r in both))
        self.assertTrue(all(r.stage2_run_id == c.RUN_ID for r in both))
        self.assertTrue(all(r.source_npz.suffix == ".npz" and r.source_video.endswith(".avi") for r in both))

    def test_test_requests_rejected_before_any_io(self):
        with patch.object(Path, "open", side_effect=AssertionError("Unexpected I/O")):
            for splits in (("test",), ("train", "test"), ("validation", "test")):
                with self.subTest(splits=splits), self.assertRaisesRegex(c.ContractError, "Held-out"):
                    self.select(splits)

    def test_fitting_rejects_validation_and_test(self):
        with self.fake_npz_existence():
            self.assertEqual(len(self.select(purpose="fit_statistics")), 60)
        with patch.object(Path, "open", side_effect=AssertionError("Unexpected I/O")):
            for splits in (("validation",), ("test",), ("train", "validation")):
                with self.subTest(splits=splits), self.assertRaises(c.ContractError):
                    self.select(splits, purpose="fit_statistics")

    def test_subject_identity_cannot_be_disguised_as_train(self):
        for subject, split in ((6, "train"), (7, "validation"), (5, "train"), (11, "train"), (True, "train")):
            with self.subTest(subject=subject), self.assertRaises(c.ContractError):
                c.require_subject_access(subject, split)
        for subject in (6, 7):
            with self.assertRaisesRegex(c.ContractError, "Held-out"):
                c.require_subject_access(subject, "test")

    def test_unknown_modes_and_split_requests_fail_closed(self):
        for mode in ("final", "frozen_application", "evaluation", "anything"):
            with self.subTest(mode=mode), self.assertRaisesRegex(c.ContractError, "not implemented"):
                self.select(purpose=mode)
        for splits in ((), ("all",), ("train", "train"), "train"):
            with self.subTest(splits=splits), self.assertRaises(c.ContractError):
                self.select(splits)

    def test_official_identity_and_pinned_metadata_match(self):
        source = self.config["source"]
        self.assertEqual(source["run_id"], "4df7dd5f-fb38-4908-8233-1a81deb8dc05")
        self.assertEqual(source["execution_commit"], "4b7557a5b3a8a4b40906b57a01a264bdeef9edb6")
        self.assertEqual(source["evidence_commit"], "a28e0289c53463b3676b6c9275d4ee18a840fdae")
        for name, key in ((source["split_config"], "split_sha256"),
                          (source["evidence_root"] + "/extraction_run.json", "run_sha256"),
                          (source["evidence_root"] + "/manifest.csv", "manifest_sha256")):
            self.assertEqual(hashlib.sha256((c.REPO / name).read_bytes()).hexdigest(), source[key])

    def test_source_identity_changes_rejected(self):
        for key in ("run_id", "execution_commit", "evidence_commit", "pose_root", "split_config"):
            original = self.config["source"][key]
            self.config["source"][key] = "changed"
            self.save_config()
            with self.subTest(key=key), self.assertRaisesRegex(c.ContractError, "identity"):
                c.read_contract(self.config_path)
            self.config["source"][key] = original

    def test_metadata_byte_mutation_rejected(self):
        source = self.config["source"]
        for name in (source["split_config"], source["evidence_root"] + "/extraction_run.json",
                     source["evidence_root"] + "/manifest.csv", "artifacts/dataset_inspection/inventory.csv"):
            path = self.repo / name
            before = path.read_bytes()
            path.write_bytes(before + b"\n")
            with self.subTest(name=name), self.assertRaisesRegex(c.ContractError, "checksum"):
                self.select()
            path.write_bytes(before)

    def test_manifest_subject_spoof_rejected_even_with_updated_digest(self):
        path = self.repo / self.config["source"]["evidence_root"] / "manifest.csv"
        reader = csv.DictReader(io.StringIO(path.read_text()))
        rows = list(reader)
        rows[0]["subject_id"] = "6"
        output = io.StringIO()
        writer = csv.DictWriter(output, reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        path.write_text(output.getvalue())
        self.config["source"]["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.save_config()
        with self.assertRaisesRegex(c.ContractError, "identity/split"):
            self.select()

    def test_missing_or_redirected_npz_rejected_without_loading(self):
        with self.assertRaisesRegex(c.ContractError, "Missing or redirected"):
            self.select()
        npz = self.repo / self.config["source"]["pose_root"] / "Subject.1/Fall backwards/FallBackwardsS1.npz"
        npz.parent.mkdir(parents=True)
        target = self.repo / "held-out.npz"
        target.write_bytes(b"not pose data")
        npz.symlink_to(target)
        with self.assertRaisesRegex(c.ContractError, "Missing or redirected"):
            self.select()

    def test_every_undecided_parameter_blocks_build(self):
        self.assertEqual(set(self.config["decisions"]), set(c.DECISIONS))
        self.assertTrue(all(v == "UNDECIDED" for v in self.config["decisions"].values()))
        with self.assertRaisesRegex(c.ContractError, "UNDECIDED") as error:
            c.require_build_ready(self.config)
        for key in c.DECISIONS:
            self.assertIn(key, str(error.exception))

    def test_missing_decisions_do_not_get_defaults(self):
        del self.config["decisions"]["window_stride_ms"]
        self.save_config()
        with self.assertRaisesRegex(c.ContractError, "no defaults"):
            c.read_contract(self.config_path)
        with self.assertRaisesRegex(c.ContractError, "window_stride_ms"):
            c.require_build_ready(self.config)
        for missing in (None, ""):
            altered = copy.deepcopy(self.config)
            altered["decisions"]["window_stride_ms"] = missing
            with self.assertRaisesRegex(c.ContractError, "window_stride_ms"):
                c.require_build_ready(altered)

    def test_build_flag_and_video_label_shortcut_cannot_enable_builder(self):
        self.config["build_enabled"] = True
        self.config["decisions"] = dict.fromkeys(c.DECISIONS, "unreviewed value")
        self.config["decisions"]["temporal_supervision_protocol"] = "broadcast_video_label"
        with self.assertRaisesRegex(c.ContractError, "no authorized builder or window supervision"):
            c.require_build_ready(self.config)

    def test_config_cannot_enable_build_or_self_approve_review(self):
        original = copy.deepcopy(self.config)
        for key, value in (("build_enabled", True), ("build_enabled", None),
                           ("review_status", "PASS")):
            self.config = copy.deepcopy(original)
            self.config[key] = value
            self.save_config()
            with self.subTest(key=key, value=value), self.assertRaisesRegex(c.ContractError, "Gate Review"):
                c.read_contract(self.config_path)

    def test_output_rejects_raw_pose_evidence_and_ancestors(self):
        source = self.config["source"]
        for name in (source["pose_root"], source["pose_root"] + "/new.npz",
                     source["evidence_root"] + "/new.json", "data/raw/new.npz", "data", "."):
            with self.subTest(name=name), self.assertRaisesRegex(c.ContractError, "immutable"):
                c.validate_output_path(Path(name), repo=self.repo)

    def test_output_rejects_symlinks_traversal_and_existing_files(self):
        allowed = self.repo / "data/processed/caucafall_v5"
        allowed.mkdir(parents=True)
        target = self.repo / self.config["source"]["pose_root"]
        target.mkdir(parents=True)
        (allowed / "alias").symlink_to(target, target_is_directory=True)
        for path in (allowed / "alias/out.npz", allowed / "../../../interim/out.npz"):
            with self.subTest(path=path), self.assertRaises(c.ContractError):
                c.validate_output_path(path, repo=self.repo)
        existing = allowed / "existing.npz"
        existing.write_bytes(b"preserve me")
        with self.assertRaisesRegex(c.ContractError, "existing"):
            c.validate_output_path(existing, repo=self.repo)
        self.assertEqual(existing.read_bytes(), b"preserve me")

    def test_output_accepts_only_separate_future_namespaces_without_writes(self):
        for name in ("data/processed/caucafall_v5/version/run/video.npz",
                     "artifacts/preprocessing/version/run/manifest.json"):
            output = c.validate_output_path(Path(name), repo=self.repo)
            self.assertEqual(output, self.repo / name)
            self.assertFalse(output.parent.exists())
        for name in ("ml/preprocessing/out.npz", "data/processed/caucafall_v5", "elsewhere/out.npz"):
            with self.subTest(name=name), self.assertRaises(c.ContractError):
                c.validate_output_path(Path(name), repo=self.repo)


if __name__ == "__main__":
    unittest.main()
