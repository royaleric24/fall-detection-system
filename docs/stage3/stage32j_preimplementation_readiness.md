# Stage 3.2j pre-implementation readiness

Readiness: **PASS for implementation**, not authorization to begin genuine review.
Reviewed on 2026-10-04 against the complete
[authoritative specification](stage32j_discrepancy_review_spec.md), Sections 1–66,
including the final Stage transition. No protocol text was reconstructed or
amended. The submitted specification's SHA-256 is
`c83a5fc410f10e67d9e3fb9b7f012e77d9eb4ae33d359bf179135d20025d3ba6`.
No reviewer was assigned, no genuine aliases or candidate mapping were created,
and no source AVI was opened. No Stage 3.2j tool or review artifact was created.

## Verified upstream identities

| Identity | Frozen value |
| --- | --- |
| Execution protocol | `44d1543747f96fc8056adf2f71cdd6db3bdb07af` |
| Annotation runtime | `725d6cd063c307fb50ddc2725ecfd1666c7f17b5` |
| Annotation run | `stage32h-6747d0191488418d932955fda609ec0e` |
| Raw-freeze manifest SHA-256 | `750e46599836bafdf1644b48cd247a3a3ea16c3d99035f7c578682ea4d06fdb8` |
| Analysis specification | `013418a92494b8350cce8856e339c7ad3f44514a` |
| Analysis implementation | `340ff0ca129d7829f85bc8fe2e5ae6173692bc05` |
| Analysis run | `stage32i-3d88c8a8f4ba4c188e93ab1864a1ba21` |
| Analysis results | `8f20d645becf3eb1603f450b47c57621352d57c2` |

All 24 files in the Stage 3.2i results directory matched their Git blobs in the
results freeze. The directory has no additional files. The Stage 3.2i implementation
and tests still match the implementation freeze. The manifest identities,
completion gate and input hashes match the frozen chain.

`stage32i_agreement.load_genuine()` passed, including `read_active()`, frozen
source-file checks, `require_agreement_ready()` and schema/completion validation.
Its 27 input identities match the frozen Stage 3.2i manifest: 24 originals plus
the raw-freeze manifest, active marker and runtime binding. Computation functions
and AVI decoding were blocked during this check. No agreement analysis was rerun.

Original records are under
`data/annotation_runs/stage32f_train_pilot_v1/{A01,A02}/finalized/`.
Their wrapper remains `state = FINALIZED`; owner is
`record.annotator.pseudonymous_id`; canonical identity is
`record.source.neutral_clip_id`. The genuine run is bound through the active
marker, runtime binding and session IDs, not by replacing the historical pilot
version in `creation.annotation_run_id`.

The existing annotation launcher requires its historical runtime commit at HEAD.
Do not rebind or invoke that launcher for review. A new review launcher must
verify historical inputs separately and bind its own protocol/tool commits.

## Actual artifact contracts and deterministic case membership

The input root is
`artifacts/temporal_annotation/stage32i_agreement/stage32i-3d88c8a8f4ba4c188e93ab1864a1ba21/`.
It must be pinned to the results commit, not discovered across analysis runs.

| Source | Actual schema and use |
| --- | --- |
| `stage32i_analysis_manifest.json` | `analysis_spec_commit` equals `analysis_spec_version`; `source_annotation_run_id` equals `source_genuine_run_id`. Separate execution/runtime/code/run identities, raw-freeze SHA, `input_file_sha256`, output paths, FPS, category/status order, and `artifact_kind = derived_analysis` are present. |
| `paired_annotations.csv` | 12 unique `annotation_unit_id` values; A01/A02 record IDs, relative annotation-file references, event values, and boundary status/E/P/L fields. |
| `boundary_status_pairs.csv` | 36 unique `annotation_unit_id × boundary_type` rows. Includes `A01_status`, `A02_status`, `temporal_comparable`, availability, event values, `A01_earliest/preferred/latest`, A02 equivalents, and derived temporal quantities. This is an administrative input, never a Phase 1 payload. |
| `boundary_temporal_comparable.csv` | Same columns, filtered to observed/observed rows; 7 fall-transition and 7 grounded-start rows. |
| `discrepancy_inventory.csv` | Same columns plus pipe-delimited `discrepancy_flags`. Multiple flags can describe the same clip/boundary. Event flags, if present, must consolidate by clip/event target rather than become three boundary copies. |

`annotation_unit_id` is the exported name of the canonical `neutral_clip_id`.
CSV booleans are `true`/`false`; `NA` is missing, not zero; JSON uses null.
The review importer must validate these representations without changing them.
`A01_source_file`/`A02_source_file` identify annotation JSON, not AVI files.

Structural inspection supports the following frozen membership, without creating
a genuine case index or disclosing case identities:

| Group / target | Logical cases | Phase 1 required | Phase 2 required |
| --- | ---: | ---: | ---: |
| TEMPORAL_CALIBRATION / fall_transition_start | 7 | 7 | 7 |
| TEMPORAL_CALIBRATION / grounded_start | 7 | 7 | 7 |
| RECOVERY_COVERAGE / recovery_coverage | 12 | 12 | 1 |
| Additional OBJECTIVE_DISCREPANCY_SUPPLEMENT | 0 | 0 | 0 |
| Administrative workload count | 26 | 26 | 15 |

There are 12 physical source clips. Workload counts are not independent sample
sizes or pooled agreement statistics. Recovery's one status discrepancy joins
its existing `recovery_coverage` case through the upstream `recovery_start`
boundary. All objective discrepancy targets are already covered. Exact-match
temporal pairs remain included. Phase 2 eligibility is administrative and hidden
from the reviewer during Phase 1.

Later construction should sort canonical source identities and frozen target
order before assigning opaque aliases/case IDs. Consolidate by clip and target;
retain multiple source flags only in the protected index. Freeze membership,
required-phase flags and source references separately from mutable progress:
changing `case_status` must not change the frozen selection or its hash.

## Source AVI resolution and playback

`PilotSpec.from_frozen()` provides the pinned private manifest. Join a canonical
ID to `private_by_id`, then resolve `source_video` relative to
`annotation_pilot.SOURCE_ROOT` (`data/raw/caucafall_v5/CAUCAFall`). The 12 paths
are unique relative `.avi` paths, with no traversal or symlink, and exist as
files. Only these twelve Train paths were stat-checked; no AVI bytes were read.
The metadata carries source SHA-256, expected frame count, FPS, width and height.
All sources belong to Train subjects 1, 2, 3, 4, 8, 9. No Validation/Test path
was required or inspected. Historical decode-integrity artifacts validated
without decoding media again; this is not a claim of a fresh video decode.

`IndexedFrameSource` / `iter_indexed_frames()` already use sequential source-AVI
decode ordinals starting at zero, validate metadata/counts, and support exact
cached forward/backward access. They avoid `CAP_PROP_POS_FRAMES` seeking.
`frame_png_bytes()` losslessly transports an already indexed decoded frame to
the browser; it does not introduce a PNG-file coordinate system.

The existing web UI has stepping and jumping, but **no continuous 1.0x/0.5x
playback or full-pass completion tracker**. Implement those in the new review
UI using the same source-AVI index and 20 FPS timebase: 50 ms/frame at 1x and
100 ms/frame at 0.5x. Never generate intermediate frames. A submitted checkbox
alone must not establish a completed first pass. Track ordered display progress
and timing, require the full 1x pass before submission, and test pause/replay,
endpoints, stale responses, browser visibility and timing failures. Do not
silently count a skipped/interrupted pass as complete. Actual playback timing
is an implementation acceptance check, not an already implemented capability.

No pose/model overlay, enhancement, stabilization, optical flow, or alternate
PNG/TXT coordinate mapping is needed. The future media endpoint resolves a
server-owned alias/case reference; it must reject arbitrary paths and IDs outside
the pinned pilot before opening a file. Rehash only the selected allowed AVI
before future playback. No dataset traversal is necessary.

## Reviewer and information boundaries

Eligibility can use the exact Section 5 record. Strict review requires all five
listed exposure/ownership booleans false and, under Section 4.1, no prior
discrepancy discussion. Missing/unknown eligibility cannot qualify as strict.
The project lead has seen aggregate results and must not be assigned as strict
R01; this assistant also has prior result exposure. No assignment is needed for
this specification freeze. Strict review is feasible with an eligible independent
reviewer. If none is available, declare NONBLINDED_EXPLORATORY_REVIEW before
review starts and retain its evidentiary limitations. Do not silently switch modes.

The current `annotation_web.annotator_state()` returns a neutral ID, event values
and boundaries. Its frame URL/header also reveals that ID. GET endpoints are not
separated into reviewer/admin roles, and the local page embeds the action token.
`AnnotationSession` exposes label-writing actions. These handlers, payloads,
tokens and sessions are **not safe to reuse as the Stage 3.2j reviewer API**.

A separate handler/session with explicit allowlisted response projections is
compatible with the existing small Python HTTP architecture. Reuse decoding and
frame rendering, not annotation mutation routes or the old public payload.
Reviewer routes must enforce phase and role server-side for every request,
including direct URL attempts; no generic repository/static-file endpoint.
Check headers, errors, HTML/JS config, downloads and browser storage for leaks.

Before the global Phase 1 lock, expose only reviewer-safe identity/target,
source-frame display metadata, definitions and Phase 1 forms. Exclude candidate
data, source IDs/paths, record IDs, flags, differences, interval quantities,
annotator identities and aggregate results. Do not transmit the administrative
index/manifest and rely on CSS to hide it.

After every required Phase 1 row is locked and verified, Phase 2 may expose only
anonymous X/Y E/P/L for temporal targets or X/Y statuses for the recovery status
case. It must not expose the source owner, seed/mapping, derived differences or
annotator aggregates. After every required Phase 2 row is locked and verified,
an administrative Phase 3 join can add owner orientation to immutable earlier
responses and frozen Stage 3.2i metrics. No earlier answer is reopened.

Protected administrative files must be absent from reviewer-served content and
inaccessible through the reviewer's filesystem/repository permissions. An
`_admin` directory name or a shared local account alone is not a security boundary.
Strict review needs an isolated reviewer session/account or browser-only access
that does not grant access to this results-containing repository/history.

## State, form and lock compatibility

Keep the eight Section 12 run states in a separate Stage 3.2j state machine.
The old DRAFT/FINALIZED model supplies an exclusive-create storage pattern, not
multi-phase locks. Do not extend or change the Stage 3.2h state machine.
Implement explicit forward transitions with all-row/hash gates, crash-safe
persistence and revalidation on restart. No partial identity unblinding.
TECHNICAL_HOLD belongs to the specified case-status vocabulary and prevents
completion; it must not be resolved by dropping a required case or rewriting
an already locked response.

Sections 20, 21–23 and 35 can be represented as JSON records and CSV exports.
Existing strict-key, enum/list and nonempty-text validators provide useful
patterns, but the annotation-record validator itself must not validate review
forms. Multi-select visual/coverage values, mechanism objects with separate
code/confidence enums, conditional rationales, and lock fields are representable.
Use the specialized coverage fields without requesting any recovery frame.
The declared response field sets contain no forbidden adjudication field.
Prohibited examples in the specification are prohibitions, not schema members.
Reject extra fields and equivalent replacement-label/ranking inputs, including
free-text violations; do not assume a field-name scan alone checks free-text meaning.

For locks, follow the existing Stage 3.2h/3.2i JSON convention: UTF-8, sorted keys,
indentation of 2, final newline, and no non-finite numbers. Freeze array ordering
and the exact hash payload in the implementation documentation/tests. A row
digest must exclude its own digest field; bind run, case, reviewer, locked data
and lock timestamp. Store digest metadata without self-reference, and keep
authoritative locked JSON separate from derived CSV exports. Phase-wide hashes
use a deterministic ordered manifest of required row identities and hashes.
This specifies a hash scope using existing serialization, not a new serializer.

The existing `O_CREAT | O_EXCL`, read-only file mode, flush/fsync and
refusal-to-resume-finalized patterns can support new immutable lock records.
They need new review-specific stores/guards; file mode alone does not prevent an
administrator from tampering, so reverify hashes before reveal and after review.

Candidate permutation is exactly SHA256(UUID seed + `|` + case ID), using the
least-significant digest bit, with the Section 25 orientation. Administrative
seed/mapping are frozen before Phase 2 and hash-verified before Phase 3. Synthetic
checks confirmed reproducible parity and equivalent byte/hex bit extraction;
no genuine seed or candidate mapping was generated.

## Recommended paths and future verification

| Purpose | Future path |
| --- | --- |
| Domain, import validation, case/state/lock logic | `ml/preprocessing/stage32j_review.py` |
| Reviewer-safe web UI and handlers | `ml/preprocessing/stage32j_review_web.py` |
| Domain/workflow tests | `tests/test_stage32j_review.py` |
| HTTP/UI isolation and playback tests | `tests/test_stage32j_review_web.py` |
| Implementation documentation | `docs/stage3/stage32j_implementation.md` |
| Review output root | `artifacts/temporal_annotation/stage32j_discrepancy_review/<review_run_id>/` |
| Protected alias map | `<review output root>/_admin/stage32j_clip_alias_mapping.json` |
| Protected X/Y map | `<review output root>/_admin/stage32j_candidate_mapping.json` |
| Protected run manifest | `<review output root>/_admin/stage32j_review_manifest.json` |

Publish a separate reviewer-safe projection; never serve `_admin` or the raw
case index. These are recommendations only; none of these runtime paths/files
was created. The next implementation freeze must document exact storage/hash
layouts and demonstrate the security boundary before any genuine review.

Before and after future review, run the read-only raw verification and compare
all 24 Stage 3.2i result files with the pinned results-commit blobs. Record input
hashes separately from review lock/mapping hashes. Do not regenerate Stage 3.2i
outputs, rebind the annotation runtime, or create another raw freeze.

Required future synthetic tests:

1. Every allowed forward transition; rejection of backward/skipped transitions,
   wrong roles, stale sessions, partial locks and restart/concurrent requests.
2. Phase 1 candidate/identity/aggregate firewall through all HTTP responses,
   headers, downloads, errors and direct endpoint requests; opaque aliases only.
3. Phase 1 form validation, conditional rationales and first full 1x pass;
   immutable per-row and global locks, hash tampering and crash recovery.
4. Fixed SHA-256 permutation vectors, protected seed/mapping, order independence,
   mapping immutability and no mapping access before permitted unblinding.
5. Phase 2 access only after all Phase 1 locks; anonymous candidate projection,
   form/rationale/mechanism validation and immutable Phase 2 locks.
6. Phase 3 only after all required Phase 2 hashes and mapping verify; read-only
   joins without editable retrospective answers.
7. Exact-match temporal inclusion, both temporal targets, all recovery clips,
   additional objective supplements, multi-flag/event deduplication, recovery
   target translation and phase eligibility/count invariants.
8. Recovery coverage fields without frame-label fields; reject unknown,
   adjudicative and semantically equivalent replacement/ranking fields.
9. Train pilot allowlist, traversal/symlink/held-out/arbitrary-path rejection
   before media access; correct source hashes and canonical zero-based frames.
10. Generated-video 1x/0.5x timing, pause/replay and bidirectional stepping,
    full-pass completion, display synchronization and technical-hold paths.
11. Stable JSON/CSV round trips, multi-select/mechanism encoding, deterministic
    row/case/mapping hashes, source/result mutation detection, and no original
    annotation writes. Genuine review responses must never become fixtures.

## Checks executed in this readiness turn

- Complete UTF-8 document read; ordered Sections 1–66 and Stage transition
  checked; fenced blocks balanced; actual Phase 1/recovery/Phase 2 field sets
  checked against forbidden fields.
- Commit/blob, manifest, raw-freeze and source-record verification; CSV header
  and aggregate case-membership checks. Data reads were restricted to the 27
  frozen annotation/control files; no media decoding or held-out traversal.
- Twelve source references checked by metadata, symlink checks and exact-path
  stat only. Historical pilot decode artifacts validated read-only.
- Synthetic SHA-256 parity, deterministic JSON ordering and hash mutation checks.
- Python 3.13.15 / OpenCV 4.12.0.88: `python -B -m unittest
  tests.test_stage32f_annotation_tool tests.test_stage32h_readiness -v`.
  Of 31 tests, 30 initially passed; the synthetic loopback HTTP test encountered
  sandbox socket-bind denial. That single test passed on an authorized retry.
  Thus all 31 distinct regressions passed, with no repository-code change.
- The authoritative specification is preserved byte-for-byte. Default
  `git diff --check` reports only its four intentional Markdown hard-break lines
  (3–6); they are not protocol omissions or implementation errors. No such
  whitespace issue was introduced in this readiness document.

No unresolved repository/schema incompatibility was found. Reviewer selection,
new playback controls, phase/role enforcement and immutable review locks remain
mandatory work for the separately authorized implementation/execution stages;
their feasibility here is not a claim that the current annotation UI already
implements the Stage 3.2j protocol.
