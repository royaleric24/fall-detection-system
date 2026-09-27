# Stage 3.2d — temporal supervision strategy contract

Status: **pending external Gate Review**. The frozen parent is
`d99478c68f2bc1bcaa736ca610affc5a8135eee6`. The external technical review
selected the strategy recorded in
[the versioned JSON contract](../configs/stage32d_temporal_supervision_strategy.json).
This decision chooses a source timeline and annotation route. It does not
approve or produce temporal labels, a final annotation ontology, windows,
preprocessing, a model-ready dataset, or training. Earlier Stage 2 and Stage 3
contracts and evidence remain unchanged.

## Layered decision precedence

At the Stage 3.0 Gate, the broader `temporal_supervision_protocol` was
`UNDECIDED`. That historical contract remains correct and immutable. Stage
3.2d is the later authoritative resolution **only** for the primary Route B
strategy, the canonical AVI timeline and Stage 2 `pose_raw_v1` source, the AVI
`frame_index` coordinate and derived `timestamp_ms`, the semantic/reference
role of official PNG/TXT, the prohibition of PNG-to-AVI frame-label joins,
and video-level weak supervision's fallback role. Downstream work must use
these explicit Stage 3.2d resolutions rather than treating Stage 3.0's
historical `UNDECIDED` value as the current value for those subdecisions.

**The latest explicit resolution wins only within its declared scope.** A
later contract does not implicitly resolve other undecided fields. The
broader temporal-supervision protocol is not fully resolved: manual
annotation ontology and state definitions, event-boundary definitions,
uncertainty/ambiguity representation, annotation schema details, annotator
agreement threshold, adjudication protocol details, and the final window-label
rule remain `UNDECIDED` for Stage 3.2e or later. This strategy record does
not enable annotation, preprocessing, windows, or training.

## Selected route and source coordinates

**Route B — manual temporal annotation directly against the Stage 2 decoded
AVI timeline — is the selected primary strategy.** The canonical pose source
is the frozen Stage 2 `pose_raw_v1` run
`4df7dd5f-fb38-4908-8233-1a81deb8dc05`, recorded in
[the Stage 3.0 contract](../configs/stage3_contract.json). The locally
distributed AVI files have a frozen measured rate of **20 FPS** in the
[dataset inventory](../artifacts/dataset_inspection/inventory.csv) and
[Stage 2 extraction contract](pose_extraction_contract.md). This is a
property of the local AVI timeline, not a PNG sampling rate or a new target
model FPS.

The immutable AVI `frame_index` is the authoritative coordinate for future
manual boundaries. It starts at zero and retains the Stage 2 sequential
decode order. `timestamp_ms` is derived deterministically as
`round(frame_index * 1000 / source_fps)` using Python's ties-to-even rounding.
It is **clip-relative**, not a Unix timestamp, and must not be entered
independently by an annotator. A future annotation record must be capable of
preserving source-video identity, subject, activity, frame index, derived
timestamp, protocol version, annotator identity or pseudonymous ID, and
annotation provenance. No annotation record or frame boundary is created here.

This route provides a verified physical timeline directly compatible with
Stage 2 raw poses and later time-aware preprocessing or real-time evaluation.
The exact label ontology, boundary definitions, handling of ambiguity, and
window supervision remain separate review decisions. The frozen
`configs/stage3_contract.json` retains its Stage 3.0 `UNDECIDED` values;
Stage 3.2d resolves only the subdecisions declared above and does not turn
other historical decisions into a build-ready policy.

## Meaning of existing labels and prohibited joins

A **fall activity video**, **fall transition/falling motion**, and
**grounded/post-fall state** are distinct concepts. The official PNG class 0
means `nofall`; class 1 means `grounded_fall_state`. These are the authors'
annotation-native semantics, not direct AVI frame-level ground truth. Class 1
does not automatically mean active falling motion, and a fall video's
activity label does not make all its frames or future windows positive. The
exact manual AVI ontology and operational boundary rules remain
**UNDECIDED until Stage 3.2e**.

The [Stage 3.2a audit](stage32a_annotation_alignment.md) found no unambiguous
Stage 2 AVI-frame-to-official-PNG/TXT correspondence. The
[Stage 3.2b audit](stage32b_annotation_timeline.md) verified all Train
PNG/TXT ordinals while leaving their physical timebase unresolved. These
conclusions prohibit treating AVI `frame_index` as a PNG ordinal or parsed
filename index, reconstructing alignment through image similarity, using
PNG class transitions as hidden AVI boundary coordinates, or converting PNG
ordinal positions into seconds with an assumed 20 or 23 FPS rate. PNG/TXT
evidence may inform understanding of the authors' grounded-fall state but
cannot supply direct AVI temporal boundaries. During annotation-protocol
design, official dataset documentation and already-reviewed semantic evidence
may inform the meaning of `grounded_fall_state`. During per-video manual AVI
boundary annotation, annotators must not inspect that video's official
PNG/TXT frame labels to determine AVI boundary positions.

The [Stage 3.2c PNG pose pilot](stage32c_png_pose_feasibility.md) remains
bounded feasibility evidence from a stateless IMAGE runtime. Its NPZs are a
non-primary experimental source. They are not promoted to the canonical pose
or training source; the pilot's availability cannot establish that PNG is
superior or inferior to AVI. No full Train PNG extraction is authorized.

## Split and annotation quality policy

Train subjects **8, 4, 3, 9, 1, 2** may be used to design and pilot the
manual AVI annotation protocol. Validation subjects **10, 5** must wait until
an initial Train protocol exists; later use requires a frozen or explicitly
versioned protocol and the Stage 3 validation-design policy. Held-out Test
subjects **6, 7** remain sealed: no Test video, pose, PNG or annotation
contents are inspected or annotated in Stage 3.2d. Future Test temporal
annotation should occur only after the protocol is frozen, preferably after
model, preprocessing and threshold decisions are frozen, with annotators
blinded to model output. This is a future policy, not authorization to open
Test now.

Manual temporal boundaries must be decided from **source AVI visual pixels
only**. Stage 2 `pose_raw_v1` is the canonical downstream model pose source,
aligned to the AVI timeline; it is **not annotation evidence**. Annotators
must not use `pose_raw_v1` landmarks or skeleton visualizations,
`pose_detected` state, MediaPipe missingness patterns or any other
MediaPipe-derived feature, model predictions or probabilities/confidence,
model errors or performance, validation performance, or future alert-system
outputs to determine boundaries. The same-video PNG/TXT frame-label exclusion
above applies as well. Future work must preserve original frame indices,
represent ambiguous boundaries explicitly, and store annotations separately
from unedited source videos.
A frozen subset of Train fall videos should be independently annotated by
more than one person before full Train annotation. Disagreement at temporal
boundaries must be measured at the **video/event level**, rather than
presenting thousands of neighboring frames as independent observations.
Adjudication must retain the original annotator records. No required
agreement threshold is selected here.

## Alternatives and remaining work

| Route | Status | Reason |
| --- | --- | --- |
| A: PNG-native pose with official PNG/TXT labels | Deferred; not primary | Order is verified, but physical PNG timing is unresolved. The pose pilot used stateless IMAGE mode rather than the canonical AVI temporal runtime. This route is not ruled out for future research. |
| B: AVI timeline with manually created temporal annotations | **Selected primary** | Verified local 20 FPS timeline, direct Stage 2 pose compatibility, and a basis for later time-aware processing. |
| C: Video-level weak supervision | Fallback | It avoids temporal annotation but does not locate active falling motion within a fall activity video. |

Stage 3.2e must **design, not presume**, the manual AVI ontology; exact
operational state and boundary definitions; annotator instructions;
ambiguity handling; frame-by-frame workflow and tooling; multi-annotator
agreement and adjudication; provenance and versioning; annotation schema;
pilot selection; and Gate criteria. The final window-label rule and
agreement threshold remain `UNDECIDED`. Stage 3.2d contains no source-video
inspection, annotation GUI, frame label, temporal window, pose transform or
model build. It stops for external Gate Review without a PASS claim.

## Verification

Tests use only standard-library reads of contracts and frozen metadata
configuration. From the repository root:

```sh
UV_CACHE_DIR=/private/tmp/caucafall-stage26-cache uv run --offline --python 3.13.15 --no-project python -B -m unittest tests.test_stage32d_temporal_supervision_strategy tests.test_stage3_contract -v
git diff --check
```

No source video, pose array, PNG or annotation TXT needs to be opened.
