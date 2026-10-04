# Stage 3.2j review-tool implementation freeze

Implementation readiness: **PASS**, verified on 2026-10-05 (Asia/Shanghai).
This documents implementation and synthetic validation only. No genuine review
run, reviewer assignment, candidate seed/mapping, Phase 1/2 response, source AVI
decode, or identity unblinding occurred. The focused commit containing this file
is the implementation freeze; it is deliberately distinct from the specification
commit and is supplied explicitly to future initialization.

Authoritative methodology: [Sections 1–66](stage32j_discrepancy_review_spec.md),
commit `7cd6a30bd2da1bea6ffe31fe65184517369d973f`, byte SHA-256
`c83a5fc410f10e67d9e3fb9b7f012e77d9eb4ae33d359bf179135d20025d3ba6`.
No methodology or historical implementation was amended. See also the
[pre-implementation readiness record](stage32j_preimplementation_readiness.md).

## Architecture and scope

- `ml/preprocessing/stage32j_review.py`: administrative input validation,
  deterministic case construction, eligibility, exact response schemas,
  serialization/hashing, irreversible state machine, persistence and exports.
- `ml/preprocessing/stage32j_review_web.py`: explicit reviewer projections,
  loopback HTTP, exact source-frame delivery, playback evidence, and forms.
- `tests/test_stage32j_review.py`: synthetic schemas, construction, provenance,
  locking, state, exports and path enforcement.
- `tests/test_stage32j_review_web.py`: synthetic HTTP attacks, playback state
  machine, FFV1 AVI integration and immutable API behavior.

This local research tool does not change the frozen edge/cloud architecture.
It adds no service dependency or annotation-writing route. Existing sequential
AVI decoding and lossless PNG encoding utilities are reused; old annotation
sessions, handlers, source-identifying payloads and label controls are not.

## Frozen inputs and provenance

| Input | Identity |
| --- | --- |
| Stage 3.2g protocol | `44d1543747f96fc8056adf2f71cdd6db3bdb07af` |
| Stage 3.2h runtime | `725d6cd063c307fb50ddc2725ecfd1666c7f17b5` |
| Annotation run | `stage32h-6747d0191488418d932955fda609ec0e` |
| Raw-freeze SHA-256 | `750e46599836bafdf1644b48cd247a3a3ea16c3d99035f7c578682ea4d06fdb8` |
| Stage 3.2i specification | `013418a92494b8350cce8856e339c7ad3f44514a` |
| Stage 3.2i implementation | `340ff0ca129d7829f85bc8fe2e5ae6173692bc05` |
| Analysis run | `stage32i-3d88c8a8f4ba4c188e93ab1864a1ba21` |
| Stage 3.2i results | `8f20d645becf3eb1603f450b47c57621352d57c2` |

`genuine_bundle()` verifies the frozen specification, calls the existing
`load_genuine()` integrity loader (`read_active`, raw-freeze/completion/schema
validation and runtime binding), and compares the exact 24 result files to their
Git blobs. It neither reruns agreement nor writes any upstream file. Canonical
`record.source.neutral_clip_id` joins the CSV `annotation_unit_id` to the pinned
pilot manifest. Administrative imports retain CSV `NA`/null and the frozen
boundary/status/event vocabularies; observed E/P/L must be ordered integers in
source-frame bounds. The file-count/hash gate detects unexpected result files.

Before initialization the implementation commit must exist, descend from the
specification freeze, match all five implementation/test/documentation files,
and be HEAD in a clean working tree. Later reads reverify the frozen tool bytes
and upstream inputs without rebinding the historical annotation launcher.

## Case construction and eligibility

Cases are sorted by canonical clip identity and the frozen target order, then
receive opaque `J001…` physical aliases and `JC0001…` logical IDs. Selection is
all observed/observed temporal pairs, all pilot recovery coverage targets, plus
only uncovered objective discrepancies. Multiple flags consolidate into one
clip/target; recovery flags join the existing coverage case. Event flags across
boundaries consolidate to one clip/event case. Phase 2 eligibility is hidden in
Phase 1, including exact-match temporal context cases.

The strict eligibility record requires all six ownership/exposure booleans to
be explicitly false, including previous discrepancy discussion under Section
4.1. The pseudonym cannot be A01/A02. Both frozen review modes are supported;
exploratory outputs carry the mandated limitation. Tests use `SYNTHETIC_R01`.
No real reviewer is assigned by default. The already exposed project lead and
assistant must not be treated as eligible independent blinded reviewers.

A random UUID is generated only by explicit run creation. For each eligible
logical case, the least-significant bit of SHA256 of `seed + "|" + case_id`
selects the frozen orientation. The seed appears only in protected administrative
manifests (immutable genesis and derived completion manifest). Mapping and alias
files are separate and hashed. The read-only aggregate preflight never calls
case/alias construction, UUID generation, permutation or AVI decoding.

## State, locking and crash behavior

The exact irreversible run chain is:

```text
CREATED → CASE_SET_FROZEN → PHASE1_OPEN → PHASE1_LOCKED
→ PHASE2_OPEN → PHASE2_LOCKED → PHASE3_UNBLINDED → COMPLETE
```

Case statuses use only the frozen vocabulary. A per-case lock cannot open Phase
2. All required Phase 1 records/hashes must pass before global Phase 1 lock and
Phase 2 opening; all required Phase 2 records and mapping hashes must pass before
Phase 3. There is no backward, skip, overwrite, deletion or hold-clear endpoint.
Phase 3 is administrative and never reveals identity through reviewer routes.

Run directories and `_admin` directories are mode 0700. Immutable files are
created exclusively with `O_EXCL | O_NOFOLLOW`, mode 0400, and fsynced. A mode
0600 `writer.lock` uses a cross-process exclusive flock. Every domain operation
reloads the chain under that lock. `_admin/genesis_manifest.json` is the immutable
administrative creation manifest; the numbered `_admin/journal/*.json` revisions
contain state snapshots, a context hash and the preceding revision hash. An
atomically replaced, fsynced `HEAD.json` commits the latest revision.

The verifier checks the complete chain, exact revision set, row hashes, frozen
identities and immutable prior rows. A missing revision or crash-created extra
tail causes HOLD rather than an inferred rollback. Persistent integrity faults
are exclusive files in `_admin/faults`; ordinary invalid requests are rejected
without fabricating a technical incident. Source/index/timing concerns and the
coordinate/playback mechanism create a case TECHNICAL_HOLD and block completion.
Investigation and any amendment/recovery procedure require a separate decision;
this tool cannot edit previously locked evidence to repair a run.

### Canonical content hash

The project convention is `json.dumps(sort_keys=True, indent=2,
allow_nan=False)` plus one LF, encoded as UTF-8. Python's default ASCII escaping
of non-ASCII characters is retained. Dictionary insertion order is irrelevant.
Multi-select enums and mechanism objects are ordered by frozen vocabulary order;
duplicates and unknown values are rejected. Ordered case arrays remain ordered.

Every persisted Phase 1/2 field except its own `phaseN_content_sha256` enters the
row hash: validated responses, run/case/reviewer identity, immutable completion
and lock booleans, lock timestamp, and Phase 1 alias/target/playback evidence.
The timestamp is deliberately part of the immutable lock event, not a mutable
last-updated value. Subsequent playback never alters a locked row. Restart
recalculates hashes; an owner/case substitution also fails schema/identity checks.

### Completion and exported artifacts

Phase 3 joins frozen Stage 3.2i quantities, locked responses and verified
orientation. Exports include all Section 48 artifacts. Eligibility, mapping and
main manifest live in `_admin`; the case index, response/display/identity CSVs,
report and audit are also administrative artifacts, never HTTP-served files.
The report contains Sections A–I, boundary-specific descriptive counts, frozen
rationale, interval-width orientation associations, recovery coverage, possible
protocol clarifications and limitations. Unassessable/unexplained cases remain
explicit; no score, causal conclusion, corrected label or adjudication is inferred.

Completion hashes the allowlisted source AVIs before and after export, writes
exclusive exports, rechecks raw/upstream integrity, then
commits COMPLETE with the exact export hash set. Exported manifest/audit point
to that verified journal as the completion authority; partially written exports
alone do not certify completion. Any failure leaves a persistent HOLD. On later
load every certified output is rehashed, including the main manifest and audit.
No rerun overwrites partial outputs. The unchanged original raw freeze remains
the reference before and after analysis/review.

## Reviewer firewall and forms

Only `/` returns static HTML without authentication. It contains no token,
case-specific values, source paths, mapping, seed or administrative bootstrap.
The administrator provides a random token in the URL fragment; the UI removes
the fragment from the address bar and sends `X-Review-Token` on every API/frame
request. The token authorizes the reviewer surface only. Host/Origin checks,
no CORS, no-store, nosniff, CSP and loopback binding limit unintended access.

| Reviewer route | Projection / gate |
| --- | --- |
| GET `/review/cases` | Current phase/mode and case ID, alias, target only |
| GET `/review/definitions` | Three definitions selected from frozen protocol Git content |
| GET `/review/case` | Safe case state, source frame count/FPS, completion evidence, own locked form values |
| GET `/review/schema` | Current-phase exact input enum/text fields only |
| GET `/review/candidates` | Global PHASE2_OPEN only; anonymous E/P/L or relevant status/event values |
| GET `/review/frame` | Current session/index/ticket, source-decoded PNG bytes only |
| POST `/review/open`, `/review/playback` | Server-owned case and controlled playback operations |
| POST `/review/lock` | Exact input schema; all managed lock/playback fields server-derived |
| POST `/review/hold` | Persistent case concern; no release operation |

No generic static-file, administrative, unblinding, label, voting or consensus
route exists. Traversal, extra query fields, unknown fields and arbitrary paths
are rejected. Errors use a constant safe message, never exception content.
Candidate values are unavailable until the global Phase 2 gate, including after
one individual Phase 1 case locks. Anonymous values contain no owner field names,
signed differences, widths, discrepancy flags or source identity.

Recovery receives all common Phase 1 fields plus the seven exact specialized
coverage fields. It has no frame-answer input. Phase 2 supports multiple
mechanisms with explicit confidence and conditional clarification rationale.
Response schemas reject unknown/adjudication fields. Visible instructions
prohibit replacement frames, intervals, labels and rankings. Human free text
cannot be perfectly classified; it is never parsed into a derived annotation.
A detected violation requires HOLD and separate investigation under Section 62.
UI free text uses `textContent`, not HTML insertion; report text is HTML-escaped.

## Playback and exact coordinates

Source evidence is resolved exclusively from a pinned pilot entry: Train,
subject 1/2/3/4/8/9, relative `.avi`, matching `Subject.<id>` prefix, no parent or
absolute path, no symlink component, correct source hash/20 FPS/count/dimensions.
Rejecting Validation 5/10, Test 6/7 and non-pilot IDs occurs before file reads.
There is no recursive dataset discovery. Genuine output paths are fixed under
`artifacts/temporal_annotation/stage32j_discrepancy_review/<run_id>`; synthetic
runs require temporary non-repository roots and `synthetic-stage32j-` IDs.

The selected AVI is sequentially decoded into ordinal source frames using the
existing `IndexedFrameSource`. A single-clip cache avoids duplicate decoding;
source hashes are rechecked when reopening that cache. Lossless in-memory PNG
encoding transports those decoded frames to the browser. No PNG file sequence,
video transcode, approximate time seek, interpolation, overlay, pose extraction,
resampling or alternative coordinate system is used.

The first pass starts at frame 0, rate 1.0 only. Backend one-time tickets require
ordered frame delivery and visible display acknowledgement through the last
frame. Monotonic deadlines require elapsed clip time at 20 FPS (50 ms/frame),
including the final frame duration. Stepping, pausing and slow mode are rejected
before completion. Browser reload/new case resets incomplete in-memory progress;
visibility loss aborts it. A posted completion boolean cannot certify a pass.
Successful completion persists per physical clip, with target-specific responses.
Afterward, replay at 1.0/0.5 (100 ms/frame), pause and clamped exact ±1 stepping
are permitted, with canonical zero-based source index displayed.

Continuous playback uses absolute deadlines and a maximum 200 ms scheduling lag.
Exceeding that engineering tolerance triggers TECHNICAL_HOLD, not silent frame
skipping or completion. This is a playback integrity check, not a new agreement
threshold. Cached frame delivery checks session/HEAD/fault state; full domain
and upstream verification occurs on opening, persistent playback evidence, form
lock and administrative gates instead of adding disk/Git work to every frame.

## Validation and interruption recovery

Recovery audit found HEAD still at the specification freeze on `main`, three
untracked partial files, no staged changes, an unfinished SHA test vector, no
Web test/documentation file and no official review output directory. The
existing 27 raw/control hashes, 24 result hashes and specification bytes matched
the prior baseline. Work resumed at test completion; no frozen operation was
replayed or overwritten.

Validation outcomes:

- **51 Stage 3.2j synthetic tests PASS**: 30 domain, 5 playback state-machine,
  16 HTTP/video tests. Subtests exhaust valid/invalid state edges, enums,
  exposures and forbidden fields. An execution audit hook prohibited any access
  under repository `data/`; observed accesses were **0**.
- Independent `shasum -a 256` vectors checked both even and odd final digest
  parity for UUID `00000000-0000-4000-8000-000000000000`, including `JC0001`
  (digest ends `012`) and `JC0004` (ends `6cb`).
- Synthetic case fixture: temporal transition **2**, grounded **2**, recovery
  **3**, additional supplement **2**; total/Phase 1 **9**, Phase 2 **7**.
- Synthetic FFV1 AVI tests compare every decoded pixel at selected exact indices,
  frame 0/1/last, backward/forward and clamping. Fake monotonic time separately
  exercises acknowledgements, deadlines, slow mode, stale sessions and aborts.
- A real isolated headless Chrome smoke test used the same synthetic eight-frame
  AVI with wall-clock playback, exact backward stepping and irreversible form
  lock. It passed without page errors. No user browser profile was used.
- Existing Stage 3.2e/f/g/h regressions: **71 PASS**; Stage 3.2i: **51 PASS**.
- Genuine aggregate-only preflight: transition **7**, grounded **7**, recovery
  **12**, supplement **0**, logical/Phase 1 **26**, Phase 2 **15**. `read_active`
  and raw-freeze verification PASS. All **27** raw/control and **24** result files
  unchanged. Guarded reads allowed only those 27 data files and four required
  annotation/control directories. Metrics, case/alias creation, UUID/permutation
  and AVI decoding were patched to fail if invoked. No held-out data access.

Executed test commands from the repository root (existing Python 3.13.15,
OpenCV-contrib 4.12.0.88, NumPy 2.2.6 runtime):

```bash
/Users/eric/.cache/uv/archive-v0/4I9mztpx7XjSSYaf/bin/python -B -m unittest \
  tests.test_stage32j_review tests.test_stage32j_review_web -q

/Users/eric/.cache/uv/archive-v0/4I9mztpx7XjSSYaf/bin/python -B -m unittest \
  tests.test_stage32e_manual_annotation_protocol \
  tests.test_stage32f_annotation_tool \
  tests.test_stage32g_execution_protocol tests.test_stage32h_readiness -q

UV_CACHE_DIR=/private/tmp/stage32i-plot-cache \
MPLCONFIGDIR=/private/tmp/stage32i-plot-cache/mplconfig \
uv run --offline --python 3.13.15 --no-project --with matplotlib==3.11.2 \
  python -B -m unittest tests.test_stage32i_agreement -q
```

The same Stage 3.2j suite was also executed through an audit-hook wrapper and
read-only preflight wrapper in `/private/tmp/stage32j-implementation-5qeealj8`.
That directory holds local check evidence and a synthetic-only browser screenshot;
it is not a genuine run, committed artifact or future runtime dependency. HTTP
checks require local socket permission in the Codex sandbox. No package or
browser installation was performed. Test corrections included a Python HTTP
handler name collision and a synthetic test's assumption of filesystem order.

## Operational limitations and future prerequisites

> reviewer uses only the reviewer UI and is not given administrative/repository
> artifact access that would reveal protected mappings.

Permissions and URL isolation do not prevent the same OS account/root from
reading or rewriting protected files, modifying code, or rolling back an entire
workspace plus its journal. SHA-256 is integrity evidence, not a digital signature
or hostile-administrator defense. Retain the frozen Git/provenance evidence
outside reviewer access. Use one server and one active reviewer browser session.
The loopback bearer-token UI is not an Internet deployment or multi-user service.
A full browser reload discards the in-memory token; reopen the authorized fragment
URL supplied by the administrator. An incomplete first pass remains incomplete.

Browser acknowledgements cannot prove human attention or defeat an adversarial
custom client. Real-browser validation covered synthetic 32×24, eight-frame
video; genuine resolution/length performance has deliberately not been measured.
The cache may require substantial memory. Any future decode/index/timing anomaly
must HOLD rather than weaken the playback gate. Existing OpenCV sequential decode
cannot distinguish EOF from every decoder failure based solely on `read()`;
expected count, metadata and source hash are independently checked.

Before any separately authorized genuine creation: freeze this implementation,
assign an eligible reviewer, complete eligibility and declare mode, arrange
operational isolation, verify raw freeze and all Stage 3.2i results, and use a
clean checkout of the implementation commit. Explicit creation freezes the case
set, aliases and candidate mapping before any video access; administrative
forward transitions then open Phase 1. No protected findings should be disclosed
during onboarding. The already exposed lead cannot be automatically assigned
strict R01. Do not silently substitute exploratory mode.

Future administrator commands (not executed for genuine review in this turn):

```bash
python -m ml.preprocessing.stage32j_review preflight
python -m ml.preprocessing.stage32j_review create \
  --eligibility /protected/reviewer_eligibility.json \
  --tool-commit <full-implementation-freeze-SHA>
python -m ml.preprocessing.stage32j_review advance \
  --run-directory <protected-run-directory> --state CASE_SET_FROZEN
python -m ml.preprocessing.stage32j_review advance \
  --run-directory <protected-run-directory> --state PHASE1_OPEN
python -m ml.preprocessing.stage32j_review_web --run-directory <protected-run-directory>
```

Later transitions must follow every frozen state in order. `COMPLETE` invokes
exclusive export and pre/post verification; Phase 3 never reopens either form.
A later genuine execution/result freeze and methodological review remain required
before considering a separately frozen Stage 3.2k. This implementation freeze
authorizes neither genuine review execution nor adjudication.
