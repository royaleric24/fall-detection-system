"""Stage 3.2j administrative domain. No genuine run is created by importing/preflight."""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import html
import json
import os
import re
import subprocess
import tempfile
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from ml.preprocessing import stage32i_agreement as upstream
from ml.preprocessing.annotation_pilot import SOURCE_ROOT, IndexedFrameSource, sha256_file

REPO = upstream.REPO
SPEC_COMMIT = '7cd6a30bd2da1bea6ffe31fe65184517369d973f'
SPEC_PATH = 'docs/stage3/stage32j_discrepancy_review_spec.md'
RESULTS_COMMIT = '8f20d645becf3eb1603f450b47c57621352d57c2'
ANALYSIS_CODE_COMMIT = '340ff0ca129d7829f85bc8fe2e5ae6173692bc05'
ANALYSIS_RUN = 'stage32i-3d88c8a8f4ba4c188e93ab1864a1ba21'
RESULTS_ROOT = upstream.OUTPUT_ROOT / ANALYSIS_RUN
OUTPUT_ROOT = REPO / 'artifacts/temporal_annotation/stage32j_discrepancy_review'
EXPORTS = ('stage32j_review_case_index.csv', 'stage32j_phase1_visual_characterization.csv',
           'stage32j_recovery_coverage_review.csv', 'stage32j_phase2_candidate_display.csv',
           'stage32j_phase2_discrepancy_mechanisms.csv', 'stage32j_phase3_identity_pattern_summary.csv',
           'stage32j_execution_audit.json', 'stage32j_report.md',
           '_admin/stage32j_review_manifest.json', '_admin/stage32j_reviewer_eligibility.json')
TOOL_FILES = ('ml/preprocessing/stage32j_review.py', 'ml/preprocessing/stage32j_review_web.py',
              'tests/test_stage32j_review.py', 'tests/test_stage32j_review_web.py',
              'docs/stage3/stage32j_implementation.md')
STATES = ('CREATED', 'CASE_SET_FROZEN', 'PHASE1_OPEN', 'PHASE1_LOCKED', 'PHASE2_OPEN',
          'PHASE2_LOCKED', 'PHASE3_UNBLINDED', 'COMPLETE')
CASE_STATUSES = ('NOT_STARTED', 'PHASE1_IN_PROGRESS', 'PHASE1_LOCKED', 'PHASE2_IN_PROGRESS',
                 'PHASE2_LOCKED', 'PHASE3_UNBLINDED', 'COMPLETE', 'TECHNICAL_HOLD')
MODES = ('STRICT_BLINDED_REVIEW', 'NONBLINDED_EXPLORATORY_REVIEW')
EXPOSURES = ('is_A01', 'is_A02', 'has_seen_stage32i_aggregate_results', 'has_seen_stage32i_per_case_results',
             'has_seen_annotator_identity_patterns', 'has_participated_in_prior_discrepancy_discussion')
TARGETS = ('fall_transition_start', 'grounded_start', 'recovery_coverage', 'event_presence', 'boundary_status')
VISUAL = ('ABRUPT_TRANSITION', 'GRADUAL_TRANSITION', 'MULTI_STAGE_TRANSITION', 'MULTIPLE_PLAUSIBLE_VISUAL_CUES',
          'PARTIAL_OCCLUSION', 'SEVERE_OCCLUSION', 'SUBJECT_PARTLY_OUT_OF_FRAME', 'MOTION_BLUR', 'LOW_VISUAL_CLARITY',
          'FURNITURE_OR_OBJECT_CONTACT', 'MULTIPLE_BODY_CONTACT_EVENTS', 'SLIDING_OR_PROGRESSIVE_SETTLING',
          'BOUNCE_OR_RECONTACT', 'CLIP_START_TRUNCATION', 'CLIP_END_TRUNCATION', 'NO_OBVIOUS_VISUAL_AMBIGUITY', 'OTHER')
CLARITY = ('clear', 'potentially_ambiguous', 'strongly_ambiguous', 'not_assessable')
COVERAGE = ('SUBJECT_REMAINS_GROUNDED', 'POST_FALL_PERIOD_VISIBLE', 'RECOVERY_BEHAVIOR_PARTIALLY_VISIBLE',
            'RECOVERY_BEHAVIOR_CLEARLY_VISIBLE', 'CLIP_ENDS_BEFORE_MEANINGFUL_RECOVERY_OPPORTUNITY',
            'CLIP_ENDS_DURING_POSSIBLE_RECOVERY', 'RECOVERY_VISIBILITY_OBSCURED',
            'RECOVERY_STATUS_SEMANTICALLY_AMBIGUOUS', 'COVERAGE_UNCLEAR', 'OTHER')
RECOVERY_ENUMS = {'post_fall_period_visible': ('yes', 'no', 'unclear'),
                  'recovery_opportunity_within_clip': ('yes', 'no', 'unclear'),
                  'recovery_behavior_visibility': ('none_visible', 'partially_visible', 'clearly_visible', 'unclear'),
                  'clip_end_truncation_relevant': ('yes', 'no', 'unclear'),
                  'recovery_visibility_obstruction': ('none', 'partial', 'severe', 'unclear')}
RELATIONS = ('SAME_PHYSICAL_SUBEVENT_DIFFERENT_POINT', 'DIFFERENT_PLAUSIBLE_PHYSICAL_SUBEVENTS',
             'SIMILAR_EVENT_REGION_DIFFERENT_INTERVAL_WIDTH', 'STATUS_SEMANTIC_DIFFERENCE',
             'CANDIDATES_BOTH_PLAUSIBLE_FROM_VISIBLE_EVIDENCE', 'CANDIDATE_RELATION_NOT_ASSESSABLE', 'OTHER')
MECHANISMS = ('SEMANTIC_BOUNDARY_AMBIGUITY', 'GRADUAL_TRANSITION', 'MULTI_STAGE_TRANSITION',
              'MULTIPLE_PLAUSIBLE_VISUAL_CUES', 'VISUAL_OCCLUSION', 'OUT_OF_FRAME', 'MOTION_BLUR_OR_LOW_CLARITY',
              'OBJECT_OR_FURNITURE_INTERACTION', 'MULTIPLE_CONTACT_EVENTS', 'PROGRESSIVE_SETTLING_OR_SLIDING',
              'CLIP_TRUNCATION_OR_CENSORING', 'UNCERTAINTY_WIDTH_CALIBRATION_DIFFERENCE',
              'STATUS_INTERPRETATION_DIFFERENCE', 'POSSIBLE_UI_OR_WORKFLOW_EFFECT',
              'POSSIBLE_COORDINATE_OR_PLAYBACK_PROBLEM', 'NO_CLEAR_MECHANISM_IDENTIFIED', 'OTHER')
CONFIDENCE = ('low', 'medium', 'high')
CLARIFICATION = ('BOUNDARY_DEFINITION', 'UNCERTAINTY_INTERVAL_SEMANTICS', 'CENSORING_RULE',
                 'OBSERVABILITY_RULE', 'RECOVERY_APPLICABILITY', 'ANNOTATION_UI_INSTRUCTION', 'OTHER')
PHASE1_INPUT = ('visual_characteristics', 'boundary_semantic_clarity', 'phase1_visual_rationale')
RECOVERY_INPUT = (*RECOVERY_ENUMS, 'recovery_coverage_characteristics', 'recovery_coverage_rationale')
PHASE2_INPUT = ('candidate_relation', 'mechanisms', 'protocol_clarification_candidate',
                'protocol_clarification_area', 'protocol_clarification_rationale', 'phase2_mechanism_rationale')
FLAGS = ('EVENT_PRESENCE_MISMATCH', 'BOUNDARY_STATUS_MISMATCH', 'PREFERRED_FRAME_MISMATCH', 'INTERVAL_DISJOINT')
FORBIDDEN = frozenset(('winner', 'correct', 'correct_candidate', 'correct_annotator', 'preferred_candidate',
                       'final', 'final_event_presence', 'final_status', 'final_boundary_status', 'final_preferred_frame',
                       'final_earliest_frame', 'final_latest_frame', 'consensus', 'consensus_frame', 'adjudicated',
                       'adjudicated_label', 'ground_truth', 'reviewer_preferred_frame', 'replacement_label',
                       'my_label', 'my_boundary', 'final_frame', 'recovery_start_frame'))
LIMITATION = ('Stage 3.2j characterizes plausible mechanisms underlying annotation discrepancies and temporal '
              'uncertainty differences. It does not determine which annotator is correct, does not produce consensus '
              'or adjudicated annotations, and does not define downstream ML supervision. All original Stage 3.2h '
              'annotations and frozen Stage 3.2i agreement results remain unchanged.')
EXPLORATORY_LIMITATION = ('The reviewer had prior exposure to Stage 3.2i findings and therefore this review is '
                         'considered exploratory rather than strictly blinded. Its mechanism classifications should '
                         'not be treated as independent blinded evidence.')


class ReviewError(ValueError):
    """Administrative diagnostic; HTTP must replace its text with a generic error."""


class IntegrityError(ReviewError):
    """HOLD: missing or changed frozen state/input."""


def require(condition: bool, message: str = 'Invalid review operation') -> None:
    if not condition:
        raise ReviewError(message)


def canonical_bytes(value: Any) -> bytes:
    """Existing Stage 3.2i JSON whitespace/key convention; list order is explicit."""
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def no_symlinks(path: Path) -> None:
    require(not any(part.is_symlink() for part in (path, *path.parents)), 'Symlink prohibited')


def exact_keys(value: Mapping[str, Any], keys: Sequence[str]) -> None:
    require(isinstance(value, dict) and set(value) == set(keys), 'Unexpected or missing fields')


def audit_fields(value: Any) -> None:
    """Response/output key audit, not a classifier of free-text human meaning."""
    if isinstance(value, dict):
        require(not set(value) & FORBIDDEN, 'Forbidden adjudication field')
        for child in value.values():
            audit_fields(child)
    elif isinstance(value, list):
        for child in value:
            audit_fields(child)


def text_field(value: Any, *, required: bool = True) -> None:
    require(isinstance(value, str) and len(value) <= 12000 and (bool(value.strip()) or not required), 'Invalid rationale')


def enum_list(value: Any, allowed: Sequence[str]) -> list[str]:
    require(isinstance(value, list) and bool(value) and all(isinstance(item, str) and item in allowed for item in value), 'Invalid selection')
    require(len(value) == len(set(value)), 'Duplicate selection')
    return sorted(value, key=allowed.index)


def validate_eligibility(value: Mapping[str, Any]) -> dict[str, Any]:
    exact_keys(value, ('reviewer_pseudonymous_id', 'review_mode', *EXPOSURES, 'eligibility_status', 'eligibility_notes'))
    require(re.fullmatch(r'[A-Za-z0-9_-]{1,64}', value['reviewer_pseudonymous_id']) is not None, 'Invalid reviewer pseudonym')
    require(value['review_mode'] in MODES and value['eligibility_status'] == 'PASS', 'Eligibility must pass')
    require(all(type(value[key]) is bool for key in EXPOSURES), 'Eligibility must be explicit')
    text_field(value['eligibility_notes'], required=False)
    if value['review_mode'] == MODES[0]:
        require(not any(value[key] for key in EXPOSURES) and value['reviewer_pseudonymous_id'] not in ('A01', 'A02'),
                'Strict reviewer is ineligible')
    return copy.deepcopy(dict(value))


def validate_phase1(value: Mapping[str, Any], target: str) -> dict[str, Any]:
    audit_fields(value)
    exact_keys(value, (*PHASE1_INPUT, *(RECOVERY_INPUT if target == 'recovery_coverage' else ())))
    row = copy.deepcopy(dict(value))
    row['visual_characteristics'] = enum_list(row['visual_characteristics'], VISUAL)
    require(row['boundary_semantic_clarity'] in CLARITY, 'Unknown semantic clarity')
    text_field(row['phase1_visual_rationale'], required=row['visual_characteristics'] != ['NO_OBVIOUS_VISUAL_AMBIGUITY'])
    if target == 'recovery_coverage':
        for key, choices in RECOVERY_ENUMS.items():
            require(row[key] in choices, 'Unknown coverage value')
        row['recovery_coverage_characteristics'] = enum_list(row['recovery_coverage_characteristics'], COVERAGE)
        text_field(row['recovery_coverage_rationale'])
    return row


def validate_phase2(value: Mapping[str, Any]) -> dict[str, Any]:
    audit_fields(value)
    exact_keys(value, PHASE2_INPUT)
    row = copy.deepcopy(dict(value))
    require(row['candidate_relation'] in RELATIONS, 'Unknown relation')
    require(isinstance(row['mechanisms'], list) and bool(row['mechanisms']), 'Mechanisms required')
    for mechanism in row['mechanisms']:
        exact_keys(mechanism, ('code', 'confidence'))
        require(mechanism['code'] in MECHANISMS and mechanism['confidence'] in CONFIDENCE, 'Unknown mechanism/confidence')
    require(len({item['code'] for item in row['mechanisms']}) == len(row['mechanisms']), 'Duplicate mechanism')
    row['mechanisms'].sort(key=lambda item: MECHANISMS.index(item['code']))
    require(type(row['protocol_clarification_candidate']) is bool, 'Clarification must be boolean')
    if row['protocol_clarification_candidate']:
        require(row['protocol_clarification_area'] in CLARIFICATION, 'Clarification area required')
        text_field(row['protocol_clarification_rationale'])
    else:
        require(row['protocol_clarification_area'] is None and row['protocol_clarification_rationale'] == '', 'Unused clarification must be empty')
    text_field(row['phase2_mechanism_rationale'])
    return row


def validate_transition(current: str, requested: str) -> None:
    require(current in STATES and requested in STATES and STATES.index(requested) == STATES.index(current) + 1,
            'Backward or skipped transition prohibited')


def candidate_permutation(seed: str, case_id: str) -> dict[str, str]:
    require(str(uuid.UUID(seed)) == seed, 'Seed must be canonical UUID')
    bit = hashlib.sha256((seed + '|' + case_id).encode('utf-8')).digest()[-1] & 1
    return {'candidate_x_source': 'A01' if bit == 0 else 'A02', 'candidate_y_source': 'A02' if bit == 0 else 'A01'}


def build_cases(pilot: Mapping[str, Any], status_rows: Sequence[Mapping[str, Any]],
                discrepancies: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pure administrative membership construction; never called by genuine preflight."""
    rows = {(row['annotation_unit_id'], row['boundary_type']): dict(row) for row in status_rows}
    require(len(rows) == len(status_rows) == len(pilot) * 3, 'Incomplete/duplicate boundary rows')
    require(set(rows) == {(clip, boundary) for clip in pilot for boundary in upstream.BOUNDARIES}, 'Wrong pilot identities')
    for row in rows.values():
        require(row['A01_status'] in upstream.STATUSES and row['A02_status'] in upstream.STATUSES, 'Unknown status')
        require(row['event_presence_A01'] in upstream.EVENTS and row['event_presence_A02'] in upstream.EVENTS, 'Unknown event')
        for owner in ('A01', 'A02'):
            if row[f'{owner}_status'] == 'observed':
                values = [row[f'{owner}_{part}'] for part in ('earliest', 'preferred', 'latest')]
                require(all(type(value) is int or (isinstance(value, str) and re.fullmatch(r'0|[1-9][0-9]*', value)) for value in values), 'Invalid candidate coordinate')
                e, p, l = map(int, values)
                require(0 <= e <= p <= l < pilot[row['annotation_unit_id']]['expected_frame_count'], 'Invalid candidate interval')
    for clip in pilot:
        for owner in ('A01', 'A02'):
            require(len({rows[(clip, boundary)][f'event_presence_{owner}'] for boundary in upstream.BOUNDARIES}) == 1, 'Inconsistent event presence')
    cases: dict[tuple[str, str], dict[str, Any]] = {}
    def add(clip: str, target: str, group: str, boundary: str, flags: Sequence[str] = ()) -> None:
        key = (clip, target)
        if key not in cases:
            cases[key] = {'neutral_clip_id': clip, 'review_target': target, 'review_group': group,
                          'objective_stage32i_flags': [], 'phase1_required': True, 'phase2_required': target != 'recovery_coverage',
                          'phase3_eligible': True, 'source_video_reference': pilot[clip]['source_video'], 'source_rows': {}}
        case = cases[key]
        case['source_rows'][boundary] = rows[(clip, boundary)]
        case['objective_stage32i_flags'] = sorted(set(case['objective_stage32i_flags']) | set(flags), key=FLAGS.index)
        if flags:
            case['phase2_required'] = True
    for clip in sorted(pilot):
        for boundary in upstream.BOUNDARIES[:2]:
            row = rows[(clip, boundary)]
            if row['A01_status'] == row['A02_status'] == 'observed':
                add(clip, boundary, 'TEMPORAL_CALIBRATION', boundary)
        add(clip, 'recovery_coverage', 'RECOVERY_COVERAGE', 'recovery_start')
    for row in discrepancies:
        clip, boundary = row['annotation_unit_id'], row['boundary_type']
        require((clip, boundary) in rows, 'Discrepancy outside pilot')
        flags = row['discrepancy_flags'].split('|')
        require(bool(flags) and all(flag in FLAGS for flag in flags), 'Unknown objective flag')
        if 'EVENT_PRESENCE_MISMATCH' in flags:
            add(clip, 'event_presence', 'OBJECTIVE_DISCREPANCY_SUPPLEMENT', boundary, ['EVENT_PRESENCE_MISMATCH'])
        remaining = [flag for flag in flags if flag != 'EVENT_PRESENCE_MISMATCH']
        if remaining:
            target = 'recovery_coverage' if boundary == 'recovery_start' else boundary
            if (clip, target) not in cases:
                target = 'boundary_status'
            add(clip, target, 'OBJECTIVE_DISCREPANCY_SUPPLEMENT', boundary, remaining)
    aliases = {clip: f'J{index:03d}' for index, clip in enumerate(sorted(pilot), 1)}
    ordered = sorted(cases.values(), key=lambda case: (case['neutral_clip_id'], TARGETS.index(case['review_target'])))
    for index, case in enumerate(ordered, 1):
        case.update(review_case_id=f'JC{index:04d}', clip_alias=aliases[case['neutral_clip_id']])
    return ordered


def resolve_source(source_root: Path, pilot: Mapping[str, Any], clip_id: str, *, verify_hash: bool = True) -> Path:
    """Resolve one allowlisted source; reject held-out/traversal before any file read."""
    require(clip_id in pilot, 'Clip outside pilot')
    source = pilot[clip_id]
    require(source['split'] == 'train' and type(source['subject_id']) is int and source['subject_id'] in {1,2,3,4,8,9},
            'Held-out source prohibited')
    relative = PurePosixPath(source['source_video'])
    require(not relative.is_absolute() and '..' not in relative.parts and '\\' not in str(relative)
            and relative.suffix.lower() == '.avi' and relative.parts[0] == f"Subject.{source['subject_id']}", 'Invalid source reference')
    path = source_root / str(relative)
    no_symlinks(path)
    require(path.resolve().is_relative_to(source_root.resolve()) and path.is_file(), 'Source unavailable')
    require(source['source_fps'] == 20 and type(source['expected_frame_count']) is int and source['expected_frame_count'] > 0,
            'Source timeline mismatch')
    if verify_hash and sha256_file(path) != source['source_avi_sha256']:
        raise IntegrityError('Source AVI hash mismatch')
    return path


def _git(*args: str) -> bytes:
    return subprocess.check_output(['git', *args], cwd=REPO)


def genuine_bundle() -> dict[str, Any]:
    """Read-only metadata/raw integrity gate; never creates cases, seed or aliases."""
    require((REPO / SPEC_PATH).read_bytes() == _git('show', f'{SPEC_COMMIT}:{SPEC_PATH}'), 'Review specification changed')
    loaded = upstream.load_genuine()
    relative_root = str(RESULTS_ROOT.relative_to(REPO))
    names = _git('ls-tree', '-r', '--name-only', RESULTS_COMMIT, '--', relative_root).decode().splitlines()
    require(len(names) == 24, 'Unexpected frozen analysis output set')
    require({path.name for path in RESULTS_ROOT.iterdir()} == {Path(path).name for path in names}, 'Analysis output set changed')
    identities = dict(loaded.identities)
    for relative in names:
        path = REPO / relative
        no_symlinks(path)
        content = path.read_bytes()
        require(content == _git('show', f'{RESULTS_COMMIT}:{relative}'), 'Frozen analysis output changed')
        identities[str(path)] = hashlib.sha256(content).hexdigest()
    manifest = json.loads((RESULTS_ROOT / 'stage32i_analysis_manifest.json').read_text())
    require(manifest['analysis_code_commit'] == ANALYSIS_CODE_COMMIT and manifest['analysis_run_id'] == ANALYSIS_RUN
            and manifest['analysis_spec_commit'] == upstream.SPEC_COMMIT and manifest['completion_gate'] == 'PASS', 'Analysis identity mismatch')
    def rows(name: str) -> list[dict[str, str]]:
        with (RESULTS_ROOT / name).open(newline='') as stream:
            return list(csv.DictReader(stream))
    return {'synthetic': False, 'source_root': str(SOURCE_ROOT), 'pilot': dict(loaded.spec.private_by_id),
            'status_rows': rows('boundary_status_pairs.csv'), 'discrepancies': rows('discrepancy_inventory.csv'),
            'input_sha256': identities}


def genuine_preflight() -> dict[str, Any]:
    bundle = genuine_bundle()
    rows, inventory = bundle['status_rows'], bundle['discrepancies']
    pilot = set(bundle['pilot'])
    require(len(pilot) == 12 and len(rows) == 36, 'Pilot count mismatch')
    temporal = {(row['annotation_unit_id'], row['boundary_type']) for row in rows
                if row['boundary_type'] in upstream.BOUNDARIES[:2] and row['A01_status'] == row['A02_status'] == 'observed'}
    require(Counter(boundary for _, boundary in temporal) == dict.fromkeys(upstream.BOUNDARIES[:2], 7), 'Temporal case counts changed')
    require(all(row['annotation_unit_id'] in pilot for row in rows + inventory), 'Unexpected input identity')
    require(not any('EVENT_PRESENCE_MISMATCH' in row['discrepancy_flags'] for row in inventory), 'Unexpected event supplement')
    require(all(row['boundary_type'] == 'recovery_start' or (row['annotation_unit_id'], row['boundary_type']) in temporal
                for row in inventory), 'Unexpected supplement')
    recovery = {row['annotation_unit_id'] for row in inventory if row['boundary_type'] == 'recovery_start'}
    require(len(recovery) == 1, 'Recovery eligibility changed')
    return {'integrity': 'PASS', 'temporal_fall_transition_start': 7, 'temporal_grounded_start': 7,
            'recovery_coverage': 12, 'additional_supplement': 0, 'logical_cases': 26,
            'phase1_required': 26, 'phase2_required': 15, 'source_AVI_decoded': False, 'review_run_created': False}


def _write_new(path: Path, value: Any) -> None:
    no_symlinks(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(canonical_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def response_hash(row: Mapping[str, Any], phase: int) -> str:
    return digest({key: value for key, value in row.items() if key != f'phase{phase}_content_sha256'})


def verify_response(row: Mapping[str, Any], phase: int, case: Mapping[str, Any], context: Mapping[str, Any]) -> None:
    prefix = f'phase{phase}'
    inputs = (*PHASE1_INPUT, *(RECOVERY_INPUT if case['review_target'] == 'recovery_coverage' else ())) if phase == 1 else PHASE2_INPUT
    managed = ('review_run_id', 'review_case_id', 'reviewer_pseudonymous_id', f'{prefix}_completed',
               f'{prefix}_locked', f'{prefix}_lock_timestamp', f'{prefix}_content_sha256')
    if phase == 1:
        managed += ('clip_alias', 'review_target', 'full_clip_viewed_at_1x', 'slow_playback_used', 'frame_step_used')
    exact_keys(row, (*inputs, *managed))
    require(row[f'{prefix}_content_sha256'] == response_hash(row, phase), 'Locked response hash mismatch')
    require(row['review_run_id'] == context['run_id'] and row['review_case_id'] == case['review_case_id']
            and row['reviewer_pseudonymous_id'] == context['eligibility']['reviewer_pseudonymous_id'], 'Locked response owner mismatch')
    require(row[f'{prefix}_locked'] is True and row[f'{prefix}_completed'] is True, 'Incomplete locked response')
    upstream._utc_timestamp(row[f'{prefix}_lock_timestamp'])
    inputs_only = {key: row[key] for key in inputs}
    if phase == 1:
        validate_phase1(inputs_only, case['review_target'])
        require(row['clip_alias'] == case['clip_alias'] and row['review_target'] == case['review_target']
                and row['full_clip_viewed_at_1x'] is True
                and type(row['slow_playback_used']) is bool and type(row['frame_step_used']) is bool, 'Invalid playback evidence')
    else:
        validate_phase2(inputs_only)


class ReviewRun:
    """Protected append-only revisions + atomic HEAD; no HTTP administration.

    Every operation reloads and verifies the chain under a cross-process flock.
    A crash leaving an uncommitted tail fails closed instead of rolling back locks.
    """

    def __init__(self, directory: Path):
        no_symlinks(directory)
        self.directory = directory.resolve()
        require(not self.directory.is_relative_to(REPO / 'data'), 'Inputs cannot be a review workspace')
        self.admin = self.directory / '_admin'
        self.read()

    @classmethod
    def create(cls, directory: Path, bundle: Mapping[str, Any], eligibility: Mapping[str, Any],
               *, tool_commit: str, run_id: str) -> 'ReviewRun':
        """Explicit admin operation. Tests MUST pass a synthetic bundle/temp root."""
        eligibility = validate_eligibility(eligibility)
        synthetic = bundle.get('synthetic') is True
        no_symlinks(directory)
        directory = directory.resolve()
        source_root = Path(bundle['source_root']).resolve()
        require(not directory.exists() and not directory.is_relative_to(source_root)
                and not source_root.is_relative_to(directory), 'Output overlaps source or existing run')
        require(not directory.is_relative_to(REPO / 'data'), 'Dataset namespace is read-only')
        if synthetic:
            require(run_id.startswith('synthetic-stage32j-') and not directory.is_relative_to(REPO), 'Synthetic runs require temporary non-repository output')
            require(any(directory.is_relative_to(root) for root in (Path(tempfile.gettempdir()).resolve(), Path('/tmp').resolve())),
                    'Synthetic output must be temporary')
        else:
            require(re.fullmatch(r'stage32j-[0-9a-f]{32}', run_id) is not None and directory == OUTPUT_ROOT / run_id,
                    'Genuine output must use fixed run root')
            verify_tool_commit(tool_commit, clean=True)
            require(bundle == genuine_bundle(), 'Genuine bundle must match frozen sources')
        cases = build_cases(bundle['pilot'], bundle['status_rows'], bundle['discrepancies'])
        seed = str(uuid.uuid4())
        mapping = {case['review_case_id']: {'review_case_id': case['review_case_id'], **candidate_permutation(seed, case['review_case_id'])}
                   for case in cases if case['phase2_required']}
        aliases = {case['clip_alias']: case['neutral_clip_id'] for case in cases}
        context = {'version': 1, 'run_id': run_id, 'synthetic': synthetic, 'tool_commit': tool_commit,
                   'specification_commit': SPEC_COMMIT, 'eligibility': eligibility, 'cases': cases,
                   'review_case_set_sha256': digest(cases), 'candidate_permutation_seed': seed,
                   'candidate_permutation_mapping_sha256': digest(mapping), 'alias_mapping_sha256': digest(aliases),
                   'bundle': copy.deepcopy(dict(bundle)), 'created_at_utc': utc_now()}
        directory.mkdir(parents=True, mode=0o700)
        os.chmod(directory, 0o700)
        admin = directory / '_admin'
        admin.mkdir(mode=0o700)
        (admin / 'journal').mkdir(mode=0o700)
        (admin / 'faults').mkdir(mode=0o700)
        fd = os.open(admin / 'writer.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        _write_new(admin / 'genesis_manifest.json', context)
        _write_new(admin / 'stage32j_candidate_mapping.json', mapping)
        _write_new(admin / 'stage32j_clip_alias_mapping.json', aliases)
        obj = cls.__new__(cls)
        obj.directory, obj.admin = directory, admin
        initial = {'state': 'CREATED', 'phase1': {}, 'phase2': {}, 'playback': {}, 'holds': {}, 'started': {},
                   'transition_timestamps': {'CREATED': utc_now()}}
        obj._append(context, initial, None)
        return cls(directory)

    @contextmanager
    def _locked(self):
        no_symlinks(self.admin / 'writer.lock')
        with (self.admin / 'writer.lock').open('r+b') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _fault(self, reason: str) -> None:
        """Independent immutable fault evidence even when the journal cannot load."""
        path = self.admin / 'faults' / (hashlib.sha256(reason.encode()).hexdigest() + '.json')
        try:
            _write_new(path, {'gate': 'HOLD', 'case_status': 'TECHNICAL_HOLD', 'reason': reason, 'timestamp': utc_now()})
        except (OSError, ReviewError):
            # A missing/unwritable workspace still fails closed at the caller.
            pass

    def _load(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        for path in (self.admin, self.admin / 'journal', self.admin / 'faults'):
            no_symlinks(path)
        if any((self.admin / 'faults').iterdir()):
            raise IntegrityError('Persistent technical integrity HOLD')
        def read(path: Path) -> Any:
            no_symlinks(path)
            return json.loads(path.read_bytes())
        context = read(self.admin / 'genesis_manifest.json')
        mapping = read(self.admin / 'stage32j_candidate_mapping.json')
        aliases = read(self.admin / 'stage32j_clip_alias_mapping.json')
        require(digest(mapping) == context['candidate_permutation_mapping_sha256'], 'Candidate mapping hash mismatch')
        require(digest(aliases) == context['alias_mapping_sha256'], 'Alias mapping hash mismatch')
        require(digest(context['cases']) == context['review_case_set_sha256'], 'Case set hash mismatch')
        validate_eligibility(context['eligibility'])
        by_id = {case['review_case_id']: case for case in context['cases']}
        for case in context['cases']:
            require(aliases[case['clip_alias']] == case['neutral_clip_id'], 'Alias identity mismatch')
            if case['phase2_required']:
                require(mapping[case['review_case_id']] == {'review_case_id': case['review_case_id'],
                        **candidate_permutation(context['candidate_permutation_seed'], case['review_case_id'])}, 'Permutation mismatch')
        head = read(self.admin / 'HEAD.json')
        require(type(head['revision']) is int and head['revision'] >= 0, 'Invalid journal head')
        expected = [f'{index:06d}.json' for index in range(head['revision'] + 1)]
        require(sorted(path.name for path in (self.admin / 'journal').iterdir()) == expected, 'Journal gap or uncommitted tail')
        previous = None
        for index, filename in enumerate(expected):
            revision = read(self.admin / 'journal' / filename)
            payload = {key: value for key, value in revision.items() if key != 'sha256'}
            require(revision['sha256'] == digest(payload) and revision['revision'] == index
                    and revision['context_sha256'] == digest(context), 'Journal content mismatch')
            require(revision['previous_sha256'] == (previous['sha256'] if previous else None), 'Journal chain mismatch')
            state = revision['snapshot']
            require(state['state'] in STATES, 'Unknown run state')
            if previous:
                old = previous['snapshot']
                if state['state'] != old['state']:
                    validate_transition(old['state'], state['state'])
                for area in ('phase1', 'phase2', 'holds'):
                    require(all(state[area].get(key) == value for key, value in old[area].items()), 'Immutable history changed')
            for phase in (1, 2):
                for case_id, row in state[f'phase{phase}'].items():
                    require(case_id in by_id and (phase == 1 or by_id[case_id]['phase2_required']), 'Unexpected locked case')
                    verify_response(row, phase, by_id[case_id], context)
            previous = revision
        require(previous['sha256'] == head['sha256'], 'Journal HEAD mismatch')
        current = previous['snapshot']
        if current['state'] == 'COMPLETE':
            require(set(current.get('output_sha256', {})) == set(EXPORTS), 'Incomplete certified exports')
            for name, expected_hash in current['output_sha256'].items():
                path = self.directory / name
                no_symlinks(path)
                require(sha256_file(path) == expected_hash, 'Certified export changed')
        for phase, minimum in ((1, 'PHASE1_LOCKED'), (2, 'PHASE2_LOCKED')):
            if STATES.index(current['state']) >= STATES.index(minimum):
                require(set(current[f'phase{phase}']) == {case['review_case_id'] for case in context['cases'] if case[f'phase{phase}_required']},
                        'Persisted global lock incomplete')
        self._verify_inputs(context)
        return context, previous['snapshot'], previous

    def _verify_inputs(self, context: Mapping[str, Any]) -> None:
        bundle = context['bundle']
        if not context['synthetic']:
            verify_tool_commit(context['tool_commit'], clean=False)
            require(bundle == genuine_bundle(), 'Frozen upstream inputs changed')
        else:
            for name, expected in bundle.get('input_sha256', {}).items():
                path = Path(name)
                no_symlinks(path)
                require(sha256_file(path) == expected, 'Synthetic source artifact changed')

    def read(self) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            with self._locked():
                context, state, _ = self._load()
                return context, state
        except (ReviewError, OSError, KeyError, TypeError, ValueError) as exc:
            self._fault(str(exc))
            raise IntegrityError('Review integrity HOLD; administrative investigation required') from exc

    def _append(self, context: Mapping[str, Any], snapshot: Mapping[str, Any], previous: Mapping[str, Any] | None) -> None:
        revision = {'revision': previous['revision'] + 1 if previous else 0, 'context_sha256': digest(context),
                    'previous_sha256': previous['sha256'] if previous else None, 'snapshot': snapshot}
        revision['sha256'] = digest(revision)
        _write_new(self.admin / 'journal' / f"{revision['revision']:06d}.json", revision)
        _sync_directory(self.admin / 'journal')
        head = self.admin / 'HEAD.json'
        no_symlinks(head)
        temporary = self.admin / f'.head-{uuid.uuid4().hex}.tmp'
        _write_new(temporary, {'revision': revision['revision'], 'sha256': revision['sha256']})
        os.replace(temporary, head)
        _sync_directory(self.admin)

    def _change(self, operation: Callable[[dict[str, Any], dict[str, Any]], None]) -> None:
        with self._locked():
            try:
                context, state, previous = self._load()
            except (ReviewError, OSError, KeyError, TypeError, ValueError) as exc:
                self._fault(str(exc))
                raise IntegrityError('Review integrity HOLD') from exc
            operation(context, state)
            self._append(context, state, previous)

    @staticmethod
    def case(context: Mapping[str, Any], case_id: str) -> dict[str, Any]:
        candidates = [case for case in context['cases'] if case['review_case_id'] == case_id]
        require(len(candidates) == 1, 'Unknown review case')
        return candidates[0]

    @staticmethod
    def _gate(context: Mapping[str, Any], state: Mapping[str, Any], phase: int) -> None:
        require(not state['holds'], 'Technical hold blocks phase transition')
        required = {case['review_case_id'] for case in context['cases'] if case[f'phase{phase}_required']}
        require(set(state[f'phase{phase}']) == required, 'All required rows must be locked')

    def advance(self, target: str) -> None:
        """Trusted administrative control, deliberately absent from reviewer HTTP."""
        require(target != 'COMPLETE', 'Use complete() to certify outputs')
        def operation(context, state):
            validate_transition(state['state'], target)
            require(not state['holds'], 'Technical hold blocks transition')
            if target in ('PHASE1_LOCKED', 'PHASE2_OPEN'):
                self._gate(context, state, 1)
            if target in ('PHASE2_LOCKED', 'PHASE3_UNBLINDED'):
                self._gate(context, state, 1)
                self._gate(context, state, 2)
            state['state'] = target
            state['transition_timestamps'][target] = utc_now()
        self._change(operation)

    def start_case(self, case_id: str) -> None:
        def operation(context, state):
            case = self.case(context, case_id)
            require(state['state'] in ('PHASE1_OPEN', 'PHASE2_OPEN') and case_id not in state['holds'], 'Review unavailable')
            if state['state'] == 'PHASE2_OPEN':
                require(case['phase2_required'], 'Case does not require Phase 2')
            state['started'][case_id] = state['state']
        self._change(operation)

    def record_playback(self, case_id: str, *, completed: bool = False, slow: bool = False, step: bool = False) -> None:
        """Trusted playback controller only; no route accepts these completion fields."""
        def operation(context, state):
            case = self.case(context, case_id)
            require(state['state'] in ('PHASE1_OPEN', 'PHASE2_OPEN') and case_id not in state['holds'], 'Review unavailable')
            evidence = state['playback'].setdefault(case['clip_alias'], {'full': False, 'slow': False, 'step': False, 'completed_at_utc': None})
            require(completed or evidence['full'], 'Full first pass required')
            if completed and not evidence['full']:
                evidence['full'] = True
                evidence['completed_at_utc'] = utc_now()
            evidence['slow'] |= slow
            evidence['step'] |= step
        self._change(operation)

    def technical_hold(self, case_id: str, reason: str) -> None:
        text_field(reason)
        def operation(context, state):
            self.case(context, case_id)
            require(state['state'] != 'COMPLETE', 'Run complete')
            state['holds'].setdefault(case_id, {'reason': reason, 'timestamp': utc_now(), 'case_status': 'TECHNICAL_HOLD'})
        self._change(operation)

    def lock_response(self, phase: int, case_id: str, submitted: Mapping[str, Any]) -> None:
        require(phase in (1, 2), 'Unknown phase')
        def operation(context, state):
            case = self.case(context, case_id)
            require(state['state'] == f'PHASE{phase}_OPEN' and case_id not in state['holds'], 'Phase unavailable')
            require(case[f'phase{phase}_required'] and case_id not in state[f'phase{phase}'], 'Locked or ineligible case')
            row = validate_phase1(submitted, case['review_target']) if phase == 1 else validate_phase2(submitted)
            if phase == 1:
                playback = state['playback'].get(case['clip_alias'], {})
                require(playback.get('full') is True, 'Full 1x first pass required')
                row.update(clip_alias=case['clip_alias'], review_target=case['review_target'], full_clip_viewed_at_1x=True,
                           slow_playback_used=playback['slow'], frame_step_used=playback['step'])
            row.update(review_run_id=context['run_id'], review_case_id=case_id,
                       reviewer_pseudonymous_id=context['eligibility']['reviewer_pseudonymous_id'])
            row.update({f'phase{phase}_completed': True, f'phase{phase}_locked': True, f'phase{phase}_lock_timestamp': utc_now()})
            row[f'phase{phase}_content_sha256'] = response_hash(row, phase)
            state[f'phase{phase}'][case_id] = row
            if phase == 2 and any(item['code'] == 'POSSIBLE_COORDINATE_OR_PLAYBACK_PROBLEM' for item in row['mechanisms']):
                state['holds'][case_id] = {'reason': 'POSSIBLE_COORDINATE_OR_PLAYBACK_PROBLEM', 'timestamp': utc_now(), 'case_status': 'TECHNICAL_HOLD'}
        self._change(operation)

    @staticmethod
    def case_status(case: Mapping[str, Any], state: Mapping[str, Any]) -> str:
        key = case['review_case_id']
        if key in state['holds']:
            return 'TECHNICAL_HOLD'
        if state['state'] == 'COMPLETE':
            return 'COMPLETE'
        if state['state'] == 'PHASE3_UNBLINDED':
            return 'PHASE3_UNBLINDED'
        if key in state['phase2']:
            return 'PHASE2_LOCKED'
        if state['started'].get(key) == 'PHASE2_OPEN':
            return 'PHASE2_IN_PROGRESS'
        if key in state['phase1']:
            return 'PHASE1_LOCKED'
        return 'PHASE1_IN_PROGRESS' if key in state['started'] else 'NOT_STARTED'

    def reviewer_cases(self) -> dict[str, Any]:
        context, state = self.read()
        require(STATES.index(state['state']) >= STATES.index('PHASE1_OPEN'), 'Review not open')
        phase2 = STATES.index(state['state']) >= STATES.index('PHASE2_OPEN')
        return {'state': state['state'], 'review_mode': context['eligibility']['review_mode'],
                'cases': [{key: case[key] for key in ('review_case_id', 'clip_alias', 'review_target')}
                          for case in context['cases'] if not phase2 or case['phase2_required']]}

    def reviewer_case(self, case_id: str) -> dict[str, Any]:
        context, state = self.read()
        require(state['state'] in ('PHASE1_OPEN', 'PHASE2_OPEN'), 'Review not open')
        case = self.case(context, case_id)
        if state['state'] == 'PHASE2_OPEN':
            require(case['phase2_required'], 'Case does not require Phase 2')
        source = context['bundle']['pilot'][case['neutral_clip_id']]
        phase = 1 if state['state'] == 'PHASE1_OPEN' else 2
        # Explicit reconstruction only: never serialize the context or source row.
        locked = state[f'phase{phase}'].get(case_id)
        safe_fields = (*PHASE1_INPUT, *RECOVERY_INPUT) if phase == 1 else PHASE2_INPUT
        return {**{key: case[key] for key in ('review_case_id', 'clip_alias', 'review_target')},
                'state': state['state'], 'case_status': self.case_status(case, state),
                'frame_count': source['expected_frame_count'], 'fps': 20,
                'full_clip_viewed_at_1x': state['playback'].get(case['clip_alias'], {}).get('full', False),
                'locked_response': {key: copy.deepcopy(locked[key]) for key in safe_fields if key in locked} if locked else None}

    def candidates(self, case_id: str) -> dict[str, Any]:
        context, state = self.read()
        require(state['state'] == 'PHASE2_OPEN' and not state['holds'], 'Candidate display unavailable')
        self._gate(context, state, 1)
        case = self.case(context, case_id)
        require(case['phase2_required'], 'No candidate display for coverage-only case')
        orientation = candidate_permutation(context['candidate_permutation_seed'], case_id)
        projected = {}
        for label, owner in (('X', orientation['candidate_x_source']), ('Y', orientation['candidate_y_source'])):
            if case['review_group'] == 'TEMPORAL_CALIBRATION':
                row = case['source_rows'][case['review_target']]
                projected[label] = {full: int(row[f'{owner}_{short}']) for short, full in
                                    zip(('earliest', 'preferred', 'latest'), upstream.COORDINATES)}
            elif case['review_target'] == 'event_presence':
                projected[label] = {'event_presence': next(iter(case['source_rows'].values()))[f'event_presence_{owner}']}
            else:
                projected[label] = {boundary: {'status': row[f'{owner}_status']} for boundary, row in case['source_rows'].items()}
        return projected

    def open_frames(self, case_id: str) -> IndexedFrameSource:
        context, state = self.read()
        require(state['state'] in ('PHASE1_OPEN', 'PHASE2_OPEN') and case_id not in state['holds'], 'Media unavailable')
        case = self.case(context, case_id)
        source = context['bundle']['pilot'][case['neutral_clip_id']]
        try:
            path = resolve_source(Path(context['bundle']['source_root']), context['bundle']['pilot'], case['neutral_clip_id'])
            return IndexedFrameSource.open_avi(path, source['expected_frame_count'], 20, source['width'], source['height'])
        except (ValueError, OSError) as exc:
            self.technical_hold(case_id, 'Source or canonical AVI decoding integrity failure')
            raise IntegrityError('Source integrity HOLD') from exc

    def identity_summary(self) -> list[dict[str, Any]]:
        """Administrative Phase 3 only; joins existing values, computes no new agreement."""
        context, state = self.read()
        require(state['state'] in ('PHASE3_UNBLINDED', 'COMPLETE'), 'Unblinding not permitted')
        self._gate(context, state, 1)
        self._gate(context, state, 2)
        result = []
        for case in context['cases']:
            key = case['review_case_id']
            phase2 = state['phase2'].get(key)
            orientation = candidate_permutation(context['candidate_permutation_seed'], key) if phase2 else {}
            result.append({'review_case_id': key, 'clip_alias': case['clip_alias'], 'review_target': case['review_target'],
                           'phase1': state['phase1'][key], 'phase2': phase2, 'orientation': orientation,
                           'frozen_stage32i_rows': copy.deepcopy(case['source_rows'])})
        return result

    def complete(self) -> None:
        """Admin certification: exclusive outputs, input recheck, then terminal state."""
        with self._locked():
            try:
                context, state, previous = self._load()
            except (ReviewError, OSError, KeyError, TypeError, ValueError) as exc:
                self._fault(str(exc))
                raise IntegrityError('Completion integrity HOLD') from exc
            validate_transition(state['state'], 'COMPLETE')
            self._gate(context, state, 1)
            self._gate(context, state, 2)
            certified = copy.deepcopy(state)
            certified['state'] = 'COMPLETE'
            certified['transition_timestamps']['COMPLETE'] = utc_now()
            try:
                for clip_id in context['bundle']['pilot']:
                    resolve_source(Path(context['bundle']['source_root']), context['bundle']['pilot'], clip_id)
                export_outputs(self, context, certified)
                self._verify_inputs(context)
                for clip_id in context['bundle']['pilot']:
                    resolve_source(Path(context['bundle']['source_root']), context['bundle']['pilot'], clip_id)
                certified['output_sha256'] = {name: sha256_file(self.directory / name) for name in EXPORTS}
                self._append(context, certified, previous)
            except Exception as exc:
                self._fault('Completion failed: ' + str(exc))
                raise IntegrityError('Completion HOLD') from exc


def _csv_new(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    no_symlinks(path)
    def cell(value: Any) -> Any:
        if value is None:
            return 'NA'
        if isinstance(value, (list, dict)):
            return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
        if type(value) is bool:
            return 'true' if value else 'false'
        return value
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o400)
    with os.fdopen(fd, 'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows({key: cell(row.get(key)) for key in columns} for row in rows)
        stream.flush()
        os.fsync(stream.fileno())


def export_outputs(run: ReviewRun, context: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    """Protected administrative exports; no files are served by reviewer routes."""
    require(state['state'] == 'COMPLETE' and not state['holds'], 'Cannot export incomplete review')
    root = run.directory
    cases = context['cases']
    p1 = [state['phase1'][case['review_case_id']] for case in cases]
    p2 = [state['phase2'][case['review_case_id']] for case in cases if case['phase2_required']]
    index = [{**{key: case[key] for key in ('review_case_id', 'clip_alias', 'neutral_clip_id', 'review_target', 'review_group',
                                          'objective_stage32i_flags', 'phase1_required', 'phase2_required', 'phase3_eligible', 'source_video_reference')},
              'case_status': 'COMPLETE'} for case in cases]
    _csv_new(root / 'stage32j_review_case_index.csv', index, tuple(index[0]))
    common = ('review_run_id', 'review_case_id', 'reviewer_pseudonymous_id', 'clip_alias', 'review_target',
              'full_clip_viewed_at_1x', 'slow_playback_used', 'frame_step_used', *PHASE1_INPUT,
              'phase1_completed', 'phase1_locked', 'phase1_lock_timestamp', 'phase1_content_sha256')
    _csv_new(root / 'stage32j_phase1_visual_characterization.csv', p1, common)
    _csv_new(root / 'stage32j_recovery_coverage_review.csv', [row for row in p1 if row['review_target'] == 'recovery_coverage'],
             (*common, *RECOVERY_INPUT))
    _csv_new(root / 'stage32j_phase2_discrepancy_mechanisms.csv', p2,
             ('review_run_id', 'review_case_id', 'reviewer_pseudonymous_id', *PHASE2_INPUT,
              'phase2_completed', 'phase2_locked', 'phase2_lock_timestamp', 'phase2_content_sha256'))
    display, joined = [], []
    for case in cases:
        key = case['review_case_id']
        response = state['phase2'].get(key)
        orientation = candidate_permutation(context['candidate_permutation_seed'], key) if response else {}
        candidate_values = {}
        if response:
            for label, owner in (('X', orientation['candidate_x_source']), ('Y', orientation['candidate_y_source'])):
                if case['review_group'] == 'TEMPORAL_CALIBRATION':
                    row = case['source_rows'][case['review_target']]
                    candidate_values[label] = dict(zip(upstream.COORDINATES, (int(row[f'{owner}_{part}']) for part in ('earliest','preferred','latest'))))
                elif case['review_target'] == 'event_presence':
                    candidate_values[label] = {'event_presence': next(iter(case['source_rows'].values()))[f'event_presence_{owner}']}
                else:
                    candidate_values[label] = {boundary: {'status': row[f'{owner}_status']} for boundary, row in case['source_rows'].items()}
            display.append({'review_case_id': key, 'clip_alias': case['clip_alias'], 'review_target': case['review_target'],
                            'X': candidate_values['X'], 'Y': candidate_values['Y']})
        joined.append({'review_case_id': key, 'clip_alias': case['clip_alias'], 'review_target': case['review_target'],
                       'candidate_x_source': orientation.get('candidate_x_source'), 'candidate_y_source': orientation.get('candidate_y_source'),
                       'phase1': state['phase1'][key], 'phase2': response, 'frozen_stage32i_rows': case['source_rows']})
    _csv_new(root / 'stage32j_phase2_candidate_display.csv', display, ('review_case_id','clip_alias','review_target','X','Y'))
    _csv_new(root / 'stage32j_phase3_identity_pattern_summary.csv', joined,
             ('review_case_id','clip_alias','review_target','candidate_x_source','candidate_y_source','phase1','phase2','frozen_stage32i_rows'))
    _write_new(run.admin / 'stage32j_reviewer_eligibility.json', context['eligibility'])
    manifest = {'stage': '3.2j', 'specification_version': SPEC_COMMIT, 'stage32j_protocol_commit': SPEC_COMMIT,
                'stage32j_tool_commit': context['tool_commit'], 'stage32j_review_run_id': context['run_id'],
                'review_mode': context['eligibility']['review_mode'], 'reviewer_pseudonymous_id': context['eligibility']['reviewer_pseudonymous_id'],
                'artifact_kind': 'synthetic_test_only' if context['synthetic'] else 'derived_review',
                'source_annotation_run_id': None if context['synthetic'] else upstream.GENUINE_RUN,
                'stage32g_execution_protocol_commit': None if context['synthetic'] else upstream.EXECUTION_COMMIT,
                'source_runtime_commit': None if context['synthetic'] else upstream.SOURCE_RUNTIME_COMMIT,
                'raw_freeze_sha256': None if context['synthetic'] else upstream.RAW_FREEZE_SHA256,
                'stage32i_analysis_run_id': None if context['synthetic'] else ANALYSIS_RUN,
                'stage32i_spec_commit': None if context['synthetic'] else upstream.SPEC_COMMIT,
                'stage32i_analysis_code_commit': None if context['synthetic'] else ANALYSIS_CODE_COMMIT,
                'stage32i_results_commit': None if context['synthetic'] else RESULTS_COMMIT,
                'review_case_set_sha256': context['review_case_set_sha256'],
                'candidate_permutation_seed': context['candidate_permutation_seed'],
                'candidate_permutation_mapping_sha256': context['candidate_permutation_mapping_sha256'],
                'case_count': len(cases), 'phase1_required_count': len(p1), 'phase2_required_count': len(p2),
                'phase1_lock_timestamp': state['transition_timestamps']['PHASE1_LOCKED'],
                'phase2_lock_timestamp': state['transition_timestamps']['PHASE2_LOCKED'],
                'phase3_unblind_timestamp': state['transition_timestamps']['PHASE3_UNBLINDED'],
                'phase1_rows_sha256': digest({key: row['phase1_content_sha256'] for key,row in state['phase1'].items()}),
                'phase2_rows_sha256': digest({key: row['phase2_content_sha256'] for key,row in state['phase2'].items()}),
                'source_artifact_paths': sorted(context['bundle'].get('input_sha256', {})),
                'source_avi_sha256': {row['source_video']: row['source_avi_sha256']
                                      for row in context['bundle']['pilot'].values()},
                'output_artifact_paths': sorted((*EXPORTS, '_admin/stage32j_candidate_mapping.json',
                                                '_admin/stage32j_clip_alias_mapping.json')),
                'gate_type': 'PROCESS_METHODOLOGY',
                'completion_authority': 'Verified COMPLETE journal, matching output hashes, and no persistent HOLD'}
    _write_new(run.admin / 'stage32j_review_manifest.json', manifest)
    counts = {target: {'visual_characteristics': dict(Counter(code for row in p1 if row['review_target'] == target for code in row['visual_characteristics'])),
                       'mechanisms': dict(Counter(item['code'] for case in cases if case['review_target'] == target
                                                 for item in state['phase2'].get(case['review_case_id'], {}).get('mechanisms', [])))}
              for target in TARGETS}
    calibration = {}
    for target in upstream.BOUNDARIES[:2]:
        details = []
        for case in cases:
            if case['review_group'] != 'TEMPORAL_CALIBRATION' or case['review_target'] != target:
                continue
            key = case['review_case_id']
            frozen = case['source_rows'][target]
            widths = [int(frozen[f'{owner}_interval_width_frames']) for owner in ('A01', 'A02')]
            orientation = 'equal' if widths[0] == widths[1] else ('A01 wider' if widths[0] > widths[1] else 'A02 wider')
            details.append({'review_case_id': key, 'frozen_width_orientation': orientation,
                            'mechanisms': state['phase2'][key]['mechanisms'],
                            'candidate_relation': state['phase2'][key]['candidate_relation'],
                            'rationale': state['phase2'][key]['phase2_mechanism_rationale']})
        calibration[target] = {
            'orientation_counts': dict(Counter(row['frozen_width_orientation'] for row in details)),
            'width_mechanism_orientation_counts': dict(Counter(row['frozen_width_orientation'] for row in details
                if any(item['code'] == 'UNCERTAINTY_WIDTH_CALIBRATION_DIFFERENCE' for item in row['mechanisms']))),
            'case_specific_evidence': details}
    coverage = [{key: row[key] for key in ('review_case_id', *RECOVERY_INPUT)} for row in p1
                if row['review_target'] == 'recovery_coverage']
    unresolved = [row['review_case_id'] for row in p1 if row['boundary_semantic_clarity'] == 'not_assessable']
    unresolved += [row['review_case_id'] for row in p2
                   if row['candidate_relation'] == 'CANDIDATE_RELATION_NOT_ASSESSABLE'
                   or any(item['code'] == 'NO_CLEAR_MECHANISM_IDENTIFIED' for item in row['mechanisms'])]
    sections = [('# Stage 3.2j review report', 'SYNTHETIC TEST ONLY' if context['synthetic'] else 'Descriptive review; no adjudication.'),
                ('## A. Provenance', f"Protocol: {SPEC_COMMIT}\nTool: {context['tool_commit']}\nRun: {context['run_id']}\nMode: {manifest['review_mode']}\nReviewer: {manifest['reviewer_pseudonymous_id']}\nStage 3.2i run: {manifest['stage32i_analysis_run_id']}\nStage 3.2i results: {manifest['stage32i_results_commit']}"),
                ('## B. Review-set completeness', json.dumps({group: sum(case['review_group'] == group for case in cases)
                    for group in ('TEMPORAL_CALIBRATION', 'RECOVERY_COVERAGE', 'OBJECTIVE_DISCREPANCY_SUPPLEMENT')}, sort_keys=True)),
                ('## C. Visual characteristics', json.dumps({'boundary_specific_counts': {key:value['visual_characteristics'] for key,value in counts.items()},
                    'descriptions': [{key:row[key] for key in ('review_case_id','review_target','boundary_semantic_clarity','phase1_visual_rationale')} for row in p1]}, sort_keys=True)),
                ('## D. Discrepancy mechanisms', json.dumps({key:value['mechanisms'] for key,value in counts.items()}, sort_keys=True)),
                ('## E. Uncertainty calibration', 'Orientation counts describe recurring versus case-specific frozen interval-width patterns; case-specific frozen rationale records proposed explanations or unexplained cases. These are descriptive associations, not causal conclusions or an annotator ranking.\n' + json.dumps(calibration, sort_keys=True)),
                ('## F. Recovery coverage', 'Coverage/opportunity, visibility and truncation responses below characterize the limitations behind absent or rare jointly observed recovery boundaries; no new recovery location is inferred.\n' + json.dumps({'counts':dict(Counter(code for row in coverage for code in row['recovery_coverage_characteristics'])), 'case_evidence':coverage}, sort_keys=True)),
                ('## G. Protocol clarification candidates', json.dumps([{'review_case_id':row['review_case_id'], 'area':row['protocol_clarification_area'], 'rationale':row['protocol_clarification_rationale']} for row in p2 if row['protocol_clarification_candidate']], ensure_ascii=True)),
                ('## H. Technical holds', 'None unresolved. Any technical hold blocks completion.'),
                ('## I. Adjudication readiness', 'All required characterization records are locked. ' +
                 ('Unassessable/unexplained cases remain: ' + ', '.join(sorted(set(unresolved))) + '. Sufficiency for Stage 3.2k design is not established automatically. '
                  if unresolved else 'No case was marked unassessable or without a clear mechanism; the recorded structure is available for Stage 3.2k design consideration. ') +
                 'Methodological review and a separately frozen Stage 3.2k protocol remain necessary; no adjudication is authorized by this report.'),
                ('## Limitations', LIMITATION + ('\n\n' + EXPLORATORY_LIMITATION if manifest['review_mode']==MODES[1] else ''))]
    report = '\n\n'.join(title + '\n\n' + html.escape(body).replace('`', '&#96;') for title,body in sections) + '\n'
    path = root / 'stage32j_report.md'
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        stream.write(report)
        stream.flush()
        os.fsync(stream.fileno())
    audit = {'completion_authority': 'Verified COMPLETE journal, matching output hashes, and no persistent HOLD',
             'original_inputs_verified_before_export': True, 'phase1_rows': len(p1), 'phase2_rows': len(p2),
             'protected_mapping_verified': True, 'adjudication_performed': False, 'downstream_supervision_designed': False,
             'output_sha256': {str(path.relative_to(root)): sha256_file(path) for path in sorted(root.iterdir()) if path.is_file()},
             'verified_at_utc': utc_now()}
    _write_new(root / 'stage32j_execution_audit.json', audit)
    _sync_directory(root)
    _sync_directory(run.admin)


def verify_tool_commit(commit: str, *, clean: bool) -> None:
    require(re.fullmatch('[0-9a-f]{40}', commit) is not None and commit not in (SPEC_COMMIT, RESULTS_COMMIT, ANALYSIS_CODE_COMMIT),
            'Distinct implementation freeze commit required')
    if clean:
        require(_git('rev-parse', 'HEAD').decode().strip() == commit and not _git('status', '--porcelain', '--untracked-files=all'),
                'Genuine initialization requires clean implementation checkout')
    require(subprocess.run(['git','merge-base','--is-ancestor',SPEC_COMMIT,commit], cwd=REPO).returncode == 0, 'Tool lineage mismatch')
    for relative in TOOL_FILES:
        require((REPO / relative).read_bytes() == _git('show', f'{commit}:{relative}'), 'Tool differs from implementation freeze')


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('preflight', help='Read-only aggregate checks; no review run or source AVI')
    create = commands.add_parser('create', help='Future authorized genuine initialization only')
    create.add_argument('--eligibility', type=Path, required=True)
    create.add_argument('--tool-commit', required=True)
    advance = commands.add_parser('advance', help='Administrative forward transition; never reviewer HTTP')
    advance.add_argument('--run-directory', type=Path, required=True)
    advance.add_argument('--state', choices=STATES[1:], required=True)
    args = parser.parse_args(argv)
    if args.command == 'preflight':
        print(json.dumps(genuine_preflight(), indent=2, sort_keys=True))
    elif args.command == 'create':
        run_id = 'stage32j-' + uuid.uuid4().hex
        run = ReviewRun.create(OUTPUT_ROOT / run_id, genuine_bundle(), json.loads(args.eligibility.read_text()),
                               tool_commit=args.tool_commit, run_id=run_id)
        print(json.dumps({'run_directory':str(run.directory), 'state':'CREATED'}))
    else:
        run = ReviewRun(args.run_directory)
        run.complete() if args.state == 'COMPLETE' else run.advance(args.state)
        print(json.dumps({'state': run.read()[1]['state']}))


if __name__ == '__main__':
    main()
