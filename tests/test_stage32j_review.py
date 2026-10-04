"""Synthetic Stage 3.2j domain tests. No genuine AVI or review initialization."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from ml.preprocessing import stage32j_review as d


def eligibility():
    return {'reviewer_pseudonymous_id': 'SYNTHETIC_R01', 'review_mode': d.MODES[0],
            **dict.fromkeys(d.EXPOSURES, False), 'eligibility_status': 'PASS', 'eligibility_notes': 'Synthetic test only'}


def fixture(root: Path, *, video: bool = False):
    source = root / 'source'
    source.mkdir()
    pilot, rows, flags = {}, [], []
    for n in range(1, 4):
        clip = f'synthetic-clip-{n}'
        path = source / f'Subject.{n}' / 'synthetic.avi'
        path.parent.mkdir()
        if video:
            import cv2
            import numpy as np
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'FFV1'), 20, (32, 24))
            assert writer.isOpened()
            for index in range(8):
                writer.write(np.full((24, 32, 3), (index * 25, n * 30, 200), dtype=np.uint8))
            writer.release()
        else:
            path.write_bytes(b'SYNTHETIC SOURCE ONLY')
        pilot[clip] = {'split':'train', 'subject_id':n, 'source_video':str(path.relative_to(source)),
                       'source_fps':20, 'expected_frame_count':8, 'width':32, 'height':24,
                       'source_avi_sha256':d.sha256_file(path)}
        for boundary in d.upstream.BOUNDARIES:
            observed = n < 3 and boundary != 'recovery_start'
            s1 = 'observed' if observed else 'not_observed'
            s2 = s1
            if n == 2 and boundary == 'recovery_start':
                s1, s2 = 'unjudgeable', 'observed'
            if n == 3:
                s1, s2 = ('unjudgeable', 'not_applicable') if boundary == 'fall_transition_start' else ('not_applicable','not_applicable')
            row = {'annotation_unit_id':clip, 'boundary_type':boundary, 'A01_status':s1, 'A02_status':s2,
                   'event_presence_A01':'fall_observed' if n < 3 else 'uncertain',
                   'event_presence_A02':'fall_observed' if n < 3 else 'no_fall_observed',
                   'A01_interval_width_frames':3 if s1=='observed' else 'NA',
                   'A02_interval_width_frames':4 if s2=='observed' else 'NA'}
            for owner,status in (('A01',s1),('A02',s2)):
                for part,value in zip(('earliest','preferred','latest'),(1,2+(n==2 and owner=='A02'),4)):
                    row[f'{owner}_{part}'] = str(value) if status=='observed' else 'NA'
            rows.append(row)
            codes = []
            if n == 2 and observed:
                codes = ['PREFERRED_FRAME_MISMATCH', 'INTERVAL_DISJOINT']
            if s1 != s2:
                codes.append('BOUNDARY_STATUS_MISMATCH')
            if n == 3:
                codes.append('EVENT_PRESENCE_MISMATCH')
            if codes:
                flags.append({**row, 'discrepancy_flags':'|'.join(codes)})
    artifact = root / 'synthetic-upstream.json'
    artifact.write_bytes(d.canonical_bytes({'synthetic':True}))
    return {'synthetic':True, 'source_root':str(source), 'pilot':pilot, 'status_rows':rows,
            'discrepancies':flags, 'input_sha256':{str(artifact):d.sha256_file(artifact)}}


def phase1(target='fall_transition_start'):
    result = {'visual_characteristics':['NO_OBVIOUS_VISUAL_AMBIGUITY'], 'boundary_semantic_clarity':'clear', 'phase1_visual_rationale':''}
    if target == 'recovery_coverage':
        result.update({key:values[0] for key,values in d.RECOVERY_ENUMS.items()})
        result.update(recovery_coverage_characteristics=['POST_FALL_PERIOD_VISIBLE'], recovery_coverage_rationale='Synthetic visible coverage description.')
    return result


def phase2():
    return {'candidate_relation':d.RELATIONS[0], 'mechanisms':[{'code':d.MECHANISMS[0], 'confidence':'medium'}],
            'protocol_clarification_candidate':False, 'protocol_clarification_area':None,
            'protocol_clarification_rationale':'', 'phase2_mechanism_rationale':'Synthetic visible mechanism description.'}


def make_run(root: Path, *, video=False):
    return d.ReviewRun.create(root / 'run', fixture(root, video=video), eligibility(),
                              tool_commit='synthetic-test-tool', run_id='synthetic-stage32j-unit')


def open_phase1(run):
    run.advance('CASE_SET_FROZEN')
    run.advance('PHASE1_OPEN')


def lock_all_phase1(run):
    context, _ = run.read()
    for case in context['cases']:
        run.record_playback(case['review_case_id'], completed=True)  # Domain unit seam; HTTP never exposes it.
        run.lock_response(1, case['review_case_id'], phase1(case['review_target']))


def open_phase2(run):
    lock_all_phase1(run)
    run.advance('PHASE1_LOCKED')
    run.advance('PHASE2_OPEN')


def lock_all_phase2(run):
    for case in run.read()[0]['cases']:
        if case['phase2_required']:
            run.lock_response(2, case['review_case_id'], phase2())


class PureTests(unittest.TestCase):
    def test_eligibility_modes_and_each_exposure(self):
        self.assertEqual(d.validate_eligibility(eligibility())['review_mode'], d.MODES[0])
        for key in d.EXPOSURES:
            item = eligibility(); item[key] = True
            with self.subTest(key=key), self.assertRaises(d.ReviewError): d.validate_eligibility(item)
            item['review_mode'] = d.MODES[1]
            self.assertEqual(d.validate_eligibility(item)['review_mode'], d.MODES[1])
        for owner in ('A01','A02'):
            item = eligibility(); item['reviewer_pseudonymous_id']=owner
            with self.assertRaises(d.ReviewError): d.validate_eligibility(item)
        item = eligibility(); del item[d.EXPOSURES[0]]
        with self.assertRaises(d.ReviewError): d.validate_eligibility(item)

    def test_every_state_edge(self):
        for a in d.STATES:
            for b in d.STATES:
                with self.subTest(a=a,b=b):
                    if d.STATES.index(b)==d.STATES.index(a)+1: d.validate_transition(a,b)
                    else:
                        with self.assertRaises(d.ReviewError): d.validate_transition(a,b)

    def test_serialization_order_utf8_and_fixed_array_order(self):
        self.assertEqual(d.canonical_bytes({'b':'中','a':1}), b'{\n  "a": 1,\n  "b": "\\u4e2d"\n}\n')
        self.assertEqual(d.digest({'a':1,'b':2}),d.digest({'b':2,'a':1}))
        row=phase1(); row['visual_characteristics']=['OTHER','GRADUAL_TRANSITION'];row['phase1_visual_rationale']='Visible sequence.'
        self.assertEqual(d.validate_phase1(row,'grounded_start')['visual_characteristics'],['GRADUAL_TRANSITION','OTHER'])
        with self.assertRaises(ValueError): d.canonical_bytes(float('nan'))

    def test_permutation_known_independent_sha_parity(self):
        seed='00000000-0000-4000-8000-000000000000'
        # Known complete SHA vectors independently checked with shasum -a 256.
        vectors={'JC0001':'452f59ca078159f788271cd167dfa2b9b2c06dacb732d0428248f7c9d536f012',
                 'JC0002':'07e640541191e11cae25f58e3daac3eeccfb5c4ceb00d3e5c29945ab0be952c2',
                 'JC0004':'37d61c759decf51879edbe449b63f2992ba64a2fea454d49f47ae288980656cb'}
        orientations=set()
        for n in range(1,20):
            case=f'JC{n:04d}'; first=d.candidate_permutation(seed,case)
            self.assertEqual(first,d.candidate_permutation(seed,case)); orientations.add(first['candidate_x_source'])
        self.assertEqual(orientations,{'A01','A02'})
        for case,expected in vectors.items():
            self.assertEqual(hashlib.sha256((seed+'|'+case).encode()).hexdigest(),expected)
            bit=int(expected[-1],16)&1
            self.assertEqual(d.candidate_permutation(seed,case)['candidate_x_source'],'A02' if bit else 'A01')

    def test_exact_phase1_enums_rationale_and_contradiction(self):
        for visual in d.VISUAL:
            for clarity in d.CLARITY:
                row=phase1();row.update(visual_characteristics=[visual],boundary_semantic_clarity=clarity,phase1_visual_rationale='Visible structure.')
                d.validate_phase1(row,'grounded_start')
        row=phase1();row['visual_characteristics']=['NO_OBVIOUS_VISUAL_AMBIGUITY','SEVERE_OCCLUSION']
        with self.assertRaises(d.ReviewError): d.validate_phase1(row,'grounded_start')
        row['phase1_visual_rationale']='The visible event is clear despite obstruction elsewhere.'
        d.validate_phase1(row,'grounded_start')
        for field,value in [('visual_characteristics',['UNKNOWN']),('visual_characteristics',[]),('boundary_semantic_clarity','bad')]:
            row=phase1();row[field]=value
            with self.assertRaises(d.ReviewError): d.validate_phase1(row,'grounded_start')

    def test_recovery_exact_vocabulary_and_no_coordinate_schema(self):
        for key,values in d.RECOVERY_ENUMS.items():
            for value in values:
                row=phase1('recovery_coverage');row[key]=value;d.validate_phase1(row,'recovery_coverage')
            row[key]='unknown'
            with self.assertRaises(d.ReviewError): d.validate_phase1(row,'recovery_coverage')
        for code in d.COVERAGE:
            row=phase1('recovery_coverage');row['recovery_coverage_characteristics']=[code];d.validate_phase1(row,'recovery_coverage')
        row['recovery_coverage_rationale']=''
        with self.assertRaises(d.ReviewError): d.validate_phase1(row,'recovery_coverage')
        self.assertFalse(any('frame' in key for key in (*d.PHASE1_INPUT,*d.RECOVERY_INPUT)))

    def test_phase2_enums_and_conditionals(self):
        for relation in d.RELATIONS:
            for confidence in d.CONFIDENCE:
                row=phase2();row['candidate_relation']=relation;row['mechanisms']=[{'code':code,'confidence':confidence} for code in reversed(d.MECHANISMS)]
                self.assertEqual([x['code'] for x in d.validate_phase2(row)['mechanisms']],list(d.MECHANISMS))
        for area in d.CLARIFICATION:
            row=phase2();row.update(protocol_clarification_candidate=True,protocol_clarification_area=area,protocol_clarification_rationale='Potential ambiguity.');d.validate_phase2(row)
        for change in ({'mechanisms':[]},{'mechanisms':[{'code':'UNKNOWN','confidence':'high'}]},
                       {'mechanisms':[{'code':'OTHER','confidence':'certain'}]}, {'phase2_mechanism_rationale':''},
                       {'candidate_relation':'winner'}, {'protocol_clarification_candidate':True},
                       {'protocol_clarification_candidate':1}, {'protocol_clarification_rationale':'unexpected'}):
            row=phase2();row.update(change)
            with self.subTest(change=change),self.assertRaises(d.ReviewError): d.validate_phase2(row)

    def test_all_forbidden_fields_and_unknown_annotation_fields(self):
        for key in d.FORBIDDEN | {'preferred_frame','earliest_frame','latest_frame','earliest_plausible_frame','latest_plausible_frame','full_clip_viewed_at_1x'}:
            for target in ('grounded_start','recovery_coverage'):
                row=phase1(target);row[key]=1
                with self.subTest(key=key,target=target),self.assertRaises(d.ReviewError): d.validate_phase1(row,target)
            row=phase2();row[key]=1
            with self.assertRaises(d.ReviewError): d.validate_phase2(row)
        with self.assertRaises(d.ReviewError): d.audit_fields({'nested':[{'winner':'X'}]})
        row=phase1();row['phase1_visual_rationale']='<script>alert(1)</script>'
        self.assertEqual(d.validate_phase1(row,'grounded_start')['phase1_visual_rationale'],row['phase1_visual_rationale'])


class RunTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='stage32j-test-')
        self.root=Path(self.temp.name).resolve()
        self.run=make_run(self.root)
        self.context=self.run.read()[0]
        self.first=self.context['cases'][0]

    def tearDown(self): self.temp.cleanup()

    def test_construction_counts_consolidation_exact_matches_and_aliases(self):
        cases=self.context['cases'];counts=Counter(case['review_target'] for case in cases)
        self.assertEqual(counts,{'fall_transition_start':2,'grounded_start':2,'recovery_coverage':3,'event_presence':1,'boundary_status':1})
        self.assertEqual(len(cases),9);self.assertEqual(sum(c['phase2_required'] for c in cases),7)
        self.assertEqual(len({(c['neutral_clip_id'],c['review_target']) for c in cases}),9)
        self.assertEqual({c['clip_alias'] for c in cases},{'J001','J002','J003'})
        bundle=self.context['bundle']
        self.assertEqual(cases,d.build_cases(dict(reversed(list(bundle['pilot'].items()))),list(reversed(bundle['status_rows'])),list(reversed(bundle['discrepancies']))))
        self.assertEqual(cases[0]['objective_stage32i_flags'],[])
        self.assertTrue(any(len(c['objective_stage32i_flags'])>1 for c in cases))

    def test_incomplete_or_invalid_source_rows_fail(self):
        bundle=copy.deepcopy(self.context['bundle'])
        with self.assertRaises(d.ReviewError): d.build_cases(bundle['pilot'],bundle['status_rows'][:-1],bundle['discrepancies'])
        bundle['status_rows'][0]['A01_preferred']='8'
        with self.assertRaises(d.ReviewError): d.build_cases(bundle['pilot'],bundle['status_rows'],bundle['discrepancies'])

    def test_phase1_projection_and_global_gates(self):
        open_phase1(self.run)
        for case in self.context['cases']:
            safe=json.dumps(self.run.reviewer_case(case['review_case_id']))
            for word in ('A01','A02','neutral_clip_id','objective_stage32i_flags','earliest','candidate','synthetic-clip'):
                self.assertNotIn(word,safe)
        self.assertNotIn(self.context['candidate_permutation_seed'],json.dumps(self.run.reviewer_cases()))
        with self.assertRaises(d.ReviewError): self.run.advance('PHASE1_LOCKED')
        with self.assertRaises(d.ReviewError): self.run.candidates(self.first['review_case_id'])
        key=self.first['review_case_id'];self.run.record_playback(key,completed=True);self.run.lock_response(1,key,phase1())
        with self.assertRaises(d.ReviewError): self.run.advance('PHASE1_LOCKED')
        with self.assertRaises(d.ReviewError): self.run.candidates(key)
        with self.assertRaises(d.ReviewError): self.run.identity_summary()

    def test_phase1_lock_requires_playback_complete_schema_and_restart_immutable(self):
        open_phase1(self.run);key=self.first['review_case_id']
        with self.assertRaises(d.ReviewError): self.run.lock_response(1,key,phase1())
        self.run.record_playback(key,completed=True)
        with self.assertRaises(d.ReviewError): self.run.lock_response(1,key,{})
        self.run.lock_response(1,key,phase1());row=self.run.read()[1]['phase1'][key]
        restart=d.ReviewRun(self.run.directory)
        self.assertEqual(restart.read()[1]['phase1'][key],row)
        self.assertEqual(row['phase1_content_sha256'],d.response_hash(row,1))
        for run in (self.run,restart):
            with self.assertRaises(d.ReviewError): run.lock_response(1,key,phase1())
        altered=copy.deepcopy(row);altered['phase1_lock_timestamp']='2000-01-01T00:00:00Z'
        self.assertNotEqual(d.response_hash(altered,1),row['phase1_content_sha256'])

    def test_phase2_lock_global_gate_restart_and_anonymous_projection(self):
        open_phase1(self.run);open_phase2(self.run);key=self.first['review_case_id']
        safe=json.dumps(self.run.candidates(key))
        for word in ('A01','A02','source','difference','width','synthetic-clip'):
            self.assertNotIn(word,safe)
        self.assertEqual(set(self.run.candidates(key)),{'X','Y'})
        with self.assertRaises(d.ReviewError): self.run.lock_response(2,key,{})
        self.run.lock_response(2,key,phase2())
        with self.assertRaises(d.ReviewError): self.run.advance('PHASE2_LOCKED')
        with self.assertRaises(d.ReviewError): self.run.identity_summary()
        restart=d.ReviewRun(self.run.directory)
        with self.assertRaises(d.ReviewError): restart.lock_response(2,key,phase2())
        self.assertEqual(restart.read()[1]['phase2'],self.run.read()[1]['phase2'])

    def _tamper(self,path,edit):
        value=json.loads(path.read_bytes());edit(value);os.chmod(path,0o600);path.write_bytes(d.canonical_bytes(value))

    def test_phase1_persisted_mutation_creates_durable_hold(self):
        open_phase1(self.run);key=self.first['review_case_id'];self.run.record_playback(key,completed=True);self.run.lock_response(1,key,phase1())
        path=sorted((self.run.admin/'journal').glob('*.json'))[-1]
        self._tamper(path,lambda rev:rev['snapshot']['phase1'][key].update(phase1_visual_rationale='tampered'))
        with self.assertRaises(d.IntegrityError): d.ReviewRun(self.run.directory)
        self.assertTrue(list((self.run.admin/'faults').iterdir()))

    def test_phase2_persisted_mutation_blocks_phase3(self):
        open_phase1(self.run);open_phase2(self.run);lock_all_phase2(self.run)
        key=self.first['review_case_id'];path=sorted((self.run.admin/'journal').glob('*.json'))[-1]
        self._tamper(path,lambda rev:rev['snapshot']['phase2'][key].update(phase2_mechanism_rationale='tampered'))
        with self.assertRaises(d.IntegrityError): self.run.advance('PHASE2_LOCKED')

    def test_deleted_record_revision_is_not_recreated(self):
        open_phase1(self.run);key=self.first['review_case_id'];self.run.record_playback(key,completed=True);self.run.lock_response(1,key,phase1())
        sorted((self.run.admin/'journal').glob('*.json'))[-1].unlink()
        with self.assertRaises(d.IntegrityError): self.run.lock_response(1,key,phase1())

    def test_uncommitted_journal_tail_fails_closed(self):
        (self.run.admin/'journal'/'000001.json').write_text('{}')
        with self.assertRaises(d.IntegrityError): self.run.read()

    def test_mapping_hash_tamper_blocks_loading(self):
        path=self.run.admin/'stage32j_candidate_mapping.json'
        self._tamper(path,lambda value:value[self.first['review_case_id']].update(candidate_x_source='A02' if value[self.first['review_case_id']]['candidate_x_source']=='A01' else 'A01'))
        with self.assertRaises(d.IntegrityError): self.run.read()

    def test_seed_tamper_blocks_loading(self):
        self._tamper(self.run.admin/'genesis_manifest.json',lambda value:value.update(candidate_permutation_seed='00000000-0000-4000-8000-000000000000'))
        with self.assertRaises(d.IntegrityError): self.run.read()

    def test_upstream_mutation_causes_hold(self):
        path=Path(next(iter(self.context['bundle']['input_sha256'])));path.write_text('changed')
        with self.assertRaises(d.IntegrityError): self.run.read()

    def test_technical_hold_and_coordinate_mechanism(self):
        open_phase1(self.run);open_phase2(self.run);key=self.first['review_case_id'];row=phase2()
        row['mechanisms']=[{'code':'POSSIBLE_COORDINATE_OR_PLAYBACK_PROBLEM','confidence':'low'}]
        self.run.lock_response(2,key,row)
        state=self.run.read()[1]
        self.assertEqual(self.run.case_status(self.first,state),'TECHNICAL_HOLD')
        with self.assertRaises(d.ReviewError): self.run.advance('PHASE2_LOCKED')
        with self.assertRaises(d.ReviewError): self.run.candidates(key)

    def test_complete_workflow_outputs_audit_no_adjudication(self):
        open_phase1(self.run);open_phase2(self.run);lock_all_phase2(self.run)
        locked=copy.deepcopy(self.run.read()[1]);self.run.advance('PHASE2_LOCKED');self.run.advance('PHASE3_UNBLINDED')
        summary=self.run.identity_summary();self.assertEqual(len(summary),9);d.audit_fields(summary)
        self.run.complete();context,state=d.ReviewRun(self.run.directory).read()
        self.assertEqual(state['state'],'COMPLETE')
        for phase in ('phase1','phase2'): self.assertEqual(state[phase],locked[phase])
        for path in self.run.directory.rglob('*.csv'):
            with path.open(newline='') as stream:
                records=list(csv.DictReader(stream));d.audit_fields(records)
            self.assertTrue(records)
        report=(self.run.directory/'stage32j_report.md').read_text()
        self.assertIn(d.LIMITATION,report)
        for section in 'ABCDEFGHI': self.assertIn('## '+section+'.',report)
        audit=json.loads((self.run.directory/'stage32j_execution_audit.json').read_text());self.assertFalse(audit['adjudication_performed'])
        for name,sha in audit['output_sha256'].items(): self.assertEqual(d.sha256_file(self.run.directory/name),sha)
        with self.assertRaises(d.ReviewError): self.run.advance('PHASE1_OPEN')
        self.assertEqual(self.run.directory.stat().st_mode&0o777,0o700)
        self.assertEqual((self.run.admin/'stage32j_candidate_mapping.json').stat().st_mode&0o777,0o400)

    def test_export_failure_blocks_completion(self):
        open_phase1(self.run);open_phase2(self.run);lock_all_phase2(self.run);self.run.advance('PHASE2_LOCKED');self.run.advance('PHASE3_UNBLINDED')
        with patch.object(d,'export_outputs',side_effect=OSError('synthetic disk failure')):
            with self.assertRaises(d.IntegrityError): self.run.complete()
        with self.assertRaises(d.IntegrityError): self.run.read()

    def test_completed_export_mutation_is_detected_on_restart(self):
        open_phase1(self.run);open_phase2(self.run);lock_all_phase2(self.run)
        self.run.advance('PHASE2_LOCKED');self.run.advance('PHASE3_UNBLINDED');self.run.complete()
        path=self.run.directory/'stage32j_report.md';os.chmod(path,0o600);path.write_text('tampered')
        with self.assertRaises(d.IntegrityError): d.ReviewRun(self.run.directory)

    def test_post_export_input_change_cannot_certify_complete(self):
        open_phase1(self.run);open_phase2(self.run);lock_all_phase2(self.run)
        self.run.advance('PHASE2_LOCKED');self.run.advance('PHASE3_UNBLINDED')
        original=d.export_outputs
        def export_then_change(*args):
            original(*args)
            Path(next(iter(self.context['bundle']['input_sha256']))).write_text('changed during export')
        with patch.object(d,'export_outputs',side_effect=export_then_change):
            with self.assertRaises(d.IntegrityError): self.run.complete()
        latest=json.loads(sorted((self.run.admin/'journal').glob('*.json'))[-1].read_text())
        self.assertEqual(latest['snapshot']['state'],'PHASE3_UNBLINDED')
        audit=json.loads((self.run.directory/'stage32j_execution_audit.json').read_text())
        self.assertNotIn('state',audit);self.assertIn('COMPLETE journal',audit['completion_authority'])
        with self.assertRaises(d.IntegrityError): self.run.read()

    def test_two_instances_cannot_overwrite_same_lock(self):
        open_phase1(self.run);other=d.ReviewRun(self.run.directory);key=self.first['review_case_id']
        self.run.record_playback(key,completed=True);self.run.lock_response(1,key,phase1())
        with self.assertRaises(d.ReviewError): other.lock_response(1,key,phase1())

    def test_source_avi_changed_after_review_blocks_completion(self):
        open_phase1(self.run);open_phase2(self.run);lock_all_phase2(self.run)
        self.run.advance('PHASE2_LOCKED');self.run.advance('PHASE3_UNBLINDED')
        (self.root/'source'/'Subject.1'/'synthetic.avi').write_bytes(b'changed after review')
        with self.assertRaises(d.IntegrityError): self.run.complete()
        self.assertFalse((self.run.directory/'stage32j_report.md').exists())

    def test_all_source_hashes_remain_unchanged_through_export(self):
        baseline={name:d.sha256_file(Path(name)) for name in self.context['bundle']['input_sha256']}
        open_phase1(self.run);open_phase2(self.run);lock_all_phase2(self.run)
        self.run.advance('PHASE2_LOCKED');self.run.advance('PHASE3_UNBLINDED');self.run.complete()
        self.assertEqual(baseline,{name:d.sha256_file(Path(name)) for name in baseline})

    def test_heldout_and_path_allowlist_before_read(self):
        bundle=self.context['bundle'];pilot=copy.deepcopy(bundle['pilot']);clip='synthetic-clip-1';root=Path(bundle['source_root'])
        self.assertTrue(d.resolve_source(root,pilot,clip).is_file())
        with self.assertRaises(d.ReviewError): d.resolve_source(root,pilot,'not-pilot')
        for subject in (5,10,6,7):
            altered=copy.deepcopy(pilot);altered[clip].update(subject_id=subject,source_video=f'Subject.{subject}/forbidden.avi')
            with patch.object(d,'sha256_file',side_effect=AssertionError('must reject before read')):
                with self.subTest(subject=subject),self.assertRaises(d.ReviewError): d.resolve_source(root,altered,clip)
        altered=copy.deepcopy(pilot);altered[clip]['split']='validation'
        with self.assertRaises(d.ReviewError): d.resolve_source(root,altered,clip)
        for relative in ('../escape.avi','/tmp/arbitrary.avi','Subject.1/../../escape.avi','Subject.1\\escape.avi'):
            altered=copy.deepcopy(pilot);altered[clip]['source_video']=relative
            with self.assertRaises(d.ReviewError): d.resolve_source(root,altered,clip)
        link=root/'Subject.1'/'escape.avi';link.symlink_to(self.root/'synthetic-upstream.json')
        altered=copy.deepcopy(pilot);altered[clip]['source_video']='Subject.1/escape.avi'
        with self.assertRaises(d.ReviewError): d.resolve_source(root,altered,clip)

    def test_output_roots_read_only_inputs_and_no_official_synthetic_output(self):
        bundle=self.context['bundle']
        for output in (Path(bundle['source_root'])/'new',d.REPO/'data'/'synthetic-test',d.OUTPUT_ROOT/'synthetic-test'):
            with self.assertRaises(d.ReviewError): d.ReviewRun.create(output,bundle,eligibility(),tool_commit='test',run_id='synthetic-stage32j-x')
        with self.assertRaises(d.ReviewError): d.ReviewRun.create(self.root/'other',bundle,eligibility(),tool_commit='test',run_id='stage32j-genuine-looking')


if __name__ == '__main__': unittest.main()
