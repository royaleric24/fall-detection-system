"""Synthetic full-run lifecycle tests. Never decode raw data or import MediaPipe."""

import json
import tempfile
import unittest
import uuid
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ml.datasets import execute_pose_run as exe
from ml.datasets import preflight_pose_run as pre


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        # Existing inventory is metadata only. All video bytes below are synthetic.
        inventory = pre.read_csv(pre.core.REPO / "artifacts/dataset_inspection/split_inventory.csv")
        self.meta = [dict(subject_id=int(r['subject'].split('.')[1]), original_activity=r['activity'],
                          binary_label=r['label'], split=r['split'], source_relative_video_path=r['relative_video_path'],
                          source_fps=float(r['fps']), width=int(float(r['width'])), height=int(float(r['height'])),
                          expected_frame_count=int(float(r['frame_count']))) for r in inventory]
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.source = self.root / "data/raw/caucafall_v5/CAUCAFall"
        self.run_id = str(uuid.uuid4())
        self.run = self.root / "artifacts/pose_extraction/runs" / self.run_id
        self.run.mkdir(parents=True)
        (self.run / "results").mkdir()
        self.output = self.root / "data/interim/caucafall_v5/pose_raw_v1_runs" / self.run_id
        self.head = "a" * 40
        self.model = self.root / "ml/checkpoints/model.task"
        self.model.parent.mkdir(parents=True)
        self.model.write_bytes(b"synthetic model")
        for m in self.meta:
            path = self.source / m['source_relative_video_path']
            path.parent.mkdir(parents=True)
            path.write_bytes(m['source_relative_video_path'].encode())
        for name in (*pre.FROZEN_FILES, "ml/datasets/preflight_pose_run.py", "ml/datasets/execute_pose_run.py", "tests/test_fake.py"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic frozen code")
        stack = ExitStack()
        self.addCleanup(stack.close)
        for key, value in (("REPO", self.root), ("SOURCE_ROOT", self.source), ("MODEL", self.model)):
            stack.enter_context(patch.object(pre.core, key, value))
        stack.enter_context(patch.object(pre, "inventory_metadata", return_value=self.meta))
        stack.enter_context(patch.object(pre, "verify_checkpoint", return_value={}))
        self.git = stack.enter_context(patch.object(pre, "git", side_effect=lambda *args: self.head if args[0] == 'rev-parse' else ''))
        self.dirty = stack.enter_context(patch.object(exe.subprocess, "check_output", return_value=b""))
        self.runtime_value = dict(git_commit=self.head, git_dirty=False, source_root=str(self.source),
                                  model_sha256=pre.core.sha256(self.model), schema_version="pose_raw_v1")
        self.runtime = stack.enter_context(patch.object(pre, "runtime", return_value=self.runtime_value))
        self.tests = stack.enter_context(patch.object(pre, "run_tests", return_value=dict(command=['synthetic'], exit_code=0, output='synthetic tests OK\n')))
        self.modules = stack.enter_context(patch.object(exe, "runtime_modules", return_value=(object(), object())))
        self.extract = stack.enter_context(patch.object(pre.core, "extract_arrays", side_effect=self.fake_extract))
        checksums = pre.source_checksums(self.meta, self.source)
        (self.run / "source_checksums.csv").write_text(pre.csv_text(checksums, ['relative_video_path', 'sha256']))
        (self.run / "tests_before.txt").write_text('synthetic tests OK\n')
        self.rows = pre.manifest_rows(self.meta, checksums, self.run_id)
        self.write_rows()
        hashes = pre.implementation_hashes()
        hashes.update({p.relative_to(self.root).as_posix(): pre.core.sha256(p)
                       for p in (self.run / 'source_checksums.csv', self.run / 'tests_before.txt')})
        self.provenance = dict(self.runtime_value, run_id=self.run_id, status='pending', git_dirty=False,
                               dataset_name='CAUCAFall', dataset_version='V5',
                               checkpoint_commit=pre.CHECKPOINT, output_root=str(self.output), result_root=str(self.run / 'results'),
                               source_checksum_manifest_path=(self.run / 'source_checksums.csv').relative_to(self.root).as_posix(),
                               scope=[m['source_relative_video_path'] for m in self.meta],
                               file_sha256=hashes, fresh_tracker_per_video=True,
                               preflight=dict(checks='passed', clean_worktree=True,
                                              tests=dict(exit_code=0, output='synthetic tests OK\n')))
        self.seal()

    def fake_extract(self, source, model, meta, progress, cv2, mp):
        n = meta['expected_frame_count']
        progress.update(decoded_frame_count=n, extracted_frame_count=n, pose_detected_count=0, pose_missing_count=n)
        return dict(frame_index=np.arange(n, dtype=np.int32), timestamp_ms=np.arange(n, dtype=np.int64) * 50,
                    pose_detected=np.zeros(n, dtype=bool), landmarks=np.full((n, 33, 4), np.nan, dtype=np.float32))

    def seal(self):
        self.provenance['identity_sha256'] = pre.identity_sha256(self.provenance)
        (self.run / 'extraction_run.json').write_text(json.dumps(self.provenance))

    def write_rows(self):
        (self.run / 'manifest.csv').write_text(pre.csv_text(self.rows, pre.FIELDS))

    def executor(self):
        return exe.Executor(self.run, self.head)

    def blocked(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.executor().execute()
        self.modules.assert_not_called()
        self.extract.assert_not_called()

    def complete_one(self):
        obj = self.executor()
        obj.validate()
        obj.provenance['status'] = 'running'
        obj.persist_run()
        obj.process(obj.rows[0], obj.metadata[0], None, None)
        return obj

    def test_run_id_and_namespace_mismatches(self):
        self.provenance['run_id'] = str(uuid.uuid4())
        self.seal()
        self.blocked()
        self.provenance['run_id'] = self.run_id
        self.provenance['output_root'] = str(self.root / 'data/interim/caucafall_v5/pose_raw_v1')
        self.seal()
        self.blocked()

    def test_symlink_input_rejected(self):
        manifest = self.run / 'manifest.csv'
        elsewhere = self.root / 'elsewhere.csv'
        manifest.rename(elsewhere)
        manifest.symlink_to(elsewhere)
        self.blocked()

    def test_altered_provenance_blocks_inference(self):
        self.provenance['fresh_tracker_per_video'] = False
        (self.run / 'extraction_run.json').write_text(json.dumps(self.provenance))
        self.blocked()

    def test_altered_source_checksum_blocks_inference(self):
        (self.source / self.meta[0]['source_relative_video_path']).write_bytes(b'changed')
        self.blocked()

    def test_altered_checksum_manifest_blocks_inference(self):
        path = self.run / 'source_checksums.csv'
        path.write_text(path.read_text().replace('sha256', 'different_header', 1))
        self.blocked()

    def test_malformed_100_row_manifest(self):
        self.rows[1] = self.rows[0].copy()
        self.write_rows()
        self.blocked()

    def test_missing_manifest_row(self):
        self.rows.pop()
        self.write_rows()
        self.blocked()

    def test_invalid_transition(self):
        for initial, final in [('pending','complete'), ('complete','running'), ('failed','running'), ('running','pending')]:
            with self.subTest(initial=initial, final=final), self.assertRaises(exe.RunBlocked):
                exe.transition({'status':initial},final)

    def test_complete_row_missing_pair(self):
        self.rows[0].update(status='complete', extracted_frame_count=self.meta[0]['expected_frame_count'],
                            pose_detected_count=0, pose_missing_count=self.meta[0]['expected_frame_count'])
        self.provenance['status'] = 'incomplete'
        self.seal()
        self.write_rows()
        self.blocked()

    def test_result_npz_mismatch(self):
        obj = self.complete_one()
        _, report = obj.paths(self.meta[0])
        result = json.loads(report.read_text())
        result['pose_missing_count'] -= 1
        report.write_text(json.dumps(result))
        self.extract.reset_mock()
        self.blocked()

    def test_complete_pair_skipped_only_after_validation(self):
        self.complete_one()
        self.extract.reset_mock()
        self.assertEqual(self.executor().execute(), 0)
        self.assertEqual(self.extract.call_count, 99)

    def test_pending_unexpected_final_artifact(self):
        output, _ = self.executor().paths(self.meta[0])
        output.parent.mkdir(parents=True)
        output.write_bytes(b'orphan')
        self.blocked()
        self.assertEqual(output.read_bytes(), b'orphan')

    def test_failure_transition_and_99_cannot_be_complete(self):
        def fail_first(*args):
            if args[2] == self.meta[0]:
                raise pre.core.ExtractionError('video_decode_failure', 'synthetic decode failure', 0)
            return self.fake_extract(*args)
        self.extract.side_effect = fail_first
        self.assertEqual(self.executor().execute(), 1)
        rows = pre.read_csv(self.run / 'manifest.csv')
        self.assertEqual(rows[0]['status'], 'failed')
        self.assertEqual(rows[0]['error_frame_index'], '0')
        self.assertEqual(sum(r['status']=='complete' for r in rows),99)
        self.assertEqual(json.loads((self.run/'extraction_run.json').read_text())['status'],'incomplete')
        self.assertEqual(json.loads((self.run/'summary.json').read_text())['totals']['statuses'], {'failed':1,'complete':99})

    def test_interrupt_and_conservative_resume(self):
        self.extract.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.executor().execute()
        rows = pre.read_csv(self.run / 'manifest.csv')
        self.assertEqual(rows[0]['status'], 'incomplete')
        self.assertEqual(rows[0]['error_type'], 'interrupted_run')
        self.assertEqual(rows[1]['status'], 'pending')
        self.modules.reset_mock()
        self.extract.reset_mock()
        self.blocked()

    def test_stale_running_is_unfinished_not_complete(self):
        self.rows[0].update(status='running', extracted_frame_count=0, pose_detected_count=0, pose_missing_count=0)
        self.provenance['status']='running'
        self.seal()
        self.write_rows()
        self.blocked()

    def test_synthetic_100_success_order_heldout_and_historical_not_reused(self):
        historical = self.root / 'data/interim/caucafall_v5/pose_raw_v1/Subject.1/Fall backwards/FallBackwardsS1.npz'
        historical.parent.mkdir(parents=True)
        historical.write_bytes(b'untouched development evidence')
        self.assertEqual(self.executor().execute(), 0)
        processed = [call.args[2]['source_relative_video_path'] for call in self.extract.call_args_list]
        self.assertEqual(processed, [m['source_relative_video_path'] for m in self.meta])
        self.assertEqual(sum(call.args[2]['subject_id'] in (6,7) for call in self.extract.call_args_list),20)
        self.assertEqual(historical.read_bytes(), b'untouched development evidence')
        summary = json.loads((self.run/'summary.json').read_text())
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(summary['totals']['validated_extracted_frames'],19877)
        self.assertEqual(len(summary['zero_pose_videos']),100)
        self.assertEqual(len(list(self.output.rglob('*.npz'))),100)
        self.assertTrue((self.run/'FULL_EXTRACTION_REPORT.md').is_file())
        self.extract.reset_mock()
        self.modules.reset_mock()
        self.assertEqual(self.executor().execute(),0)
        self.extract.assert_not_called()
        self.modules.assert_not_called()

    def test_implementation_hash_mismatch_blocks_inference(self):
        (self.root / 'ml/datasets/execute_pose_run.py').write_text('changed')
        self.blocked()

    def test_head_mismatch_blocks_inference(self):
        self.head='b'*40
        self.blocked()

    def test_dirty_code_rejected_owned_run_metadata_allowed(self):
        self.dirty.return_value=b' M ml/datasets/pose_raw.py\0'
        self.blocked()
        relative=(self.run/'manifest.csv').relative_to(self.root).as_posix()
        self.dirty.return_value=('?? '+relative+'\0').encode()
        self.executor().validate()

    def test_run_artifact_allowlist_rejects_unrelated_file(self):
        (self.run / 'unexpected.py').write_text('not run metadata')
        self.blocked()

    def test_exclusive_sentinel_preserves_existing_lock(self):
        (self.run / 'executor.lock').write_text('other owner')
        with self.assertRaises(FileExistsError):
            self.executor().execute()
        self.extract.assert_not_called()
        self.assertEqual((self.run/'executor.lock').read_text(),'other owner')

    def test_post_tests_failure_does_not_finalize_run(self):
        self.tests.side_effect = ValueError('post tests failed')
        with self.assertRaisesRegex(ValueError, 'post tests failed'):
            self.executor().execute()
        self.assertEqual(json.loads((self.run/'extraction_run.json').read_text())['status'],'incomplete')
        self.assertFalse((self.run/'FULL_EXTRACTION_REPORT.md').exists())

    def test_invalid_status_and_pending_counter_rejected(self):
        self.rows[0]['status']='unknown'
        self.write_rows()
        self.blocked()
        self.rows[0]['status']='pending'
        self.rows[0]['extracted_frame_count']='0'
        self.write_rows()
        self.blocked()

    def test_final_reload_failure_never_marks_row_complete(self):
        with patch.object(exe,'load_validated',side_effect=ValueError('corrupt final NPZ')):
            with self.assertRaisesRegex(ValueError,'corrupt final NPZ'):
                self.executor().execute()
        self.assertEqual(pre.read_csv(self.run/'manifest.csv')[0]['status'],'failed')
        self.assertEqual(self.extract.call_count,1)

    def test_atomic_manifest_failure_preserves_previous_ledger(self):
        before=(self.run/'manifest.csv').read_bytes()
        with patch.object(exe.os,'replace',side_effect=OSError('injected')):
            with self.assertRaises(OSError):
                exe.atomic_text(self.run/'manifest.csv','invalid')
        self.assertEqual((self.run/'manifest.csv').read_bytes(),before)
        self.assertFalse(list(self.run.glob('.ledger-*')))

    def test_result_publication_failure_is_failed_without_false_completion(self):
        with patch.object(pre.core, 'publish_video_pair', side_effect=pre.core.ExtractionError('output_persistence_failure','injected')):
            self.assertEqual(self.executor().execute(),1)
        rows=pre.read_csv(self.run/'manifest.csv')
        self.assertTrue(all(r['status']=='failed' for r in rows))
        self.assertFalse(list((self.run/'results').rglob('*.json')))

    def test_model_change_during_extract_stops(self):
        def change(*args):
            arrays=self.fake_extract(*args)
            self.model.write_bytes(b'changed model')
            return arrays
        self.extract.side_effect=change
        with self.assertRaises(exe.RunBlocked):
            self.executor().execute()
        self.assertEqual(self.extract.call_count,1)
        self.assertEqual(pre.read_csv(self.run/'manifest.csv')[0]['status'],'failed')
        self.assertFalse(list(self.output.rglob('*.npz')))


if __name__ == '__main__':
    unittest.main()
