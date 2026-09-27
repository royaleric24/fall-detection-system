# Stage 3.2e — manual AVI temporal annotation protocol and schema

Status: **pending external Gate Review**. Parent:
`5c420adc145767ca83a065da7e9bdbe0261bcc0d`. This is a protocol and
pure validation contract, not permission to inspect videos or create real
annotations. The [machine-readable protocol](../configs/stage32e_manual_annotation_protocol.json)
and [validator](../ml/preprocessing/manual_annotation_contract.py) describe
future records. No pilot clips are selected here.

Stage 3.2e is the latest explicit resolution only for the manual
event-presence vocabulary, minimal event-boundary ontology and status
semantics, boundary uncertainty, source-record schema, reason flags,
annotator blinding, descriptive agreement methodology, and separate
adjudication-record requirement. Stage 3.2d's Route B and source/evidence
decisions remain authoritative and unchanged. Under the layered rule, the
latest explicit resolution wins **only within its declared scope**; Stage
3.0 and Stage 3.2d remain historically intact. A full state ontology beyond
these recorded event boundaries, final model-label semantics, window rules,
pilot size/selection, adjudication decision procedure and acceptance
thresholds are outside Stage 3.2e's resolved scope.

## Evidence and decision layers

Stage 3.2d's Route B remains authoritative: manual boundaries come from
**source AVI visual pixels only**. Stage 2 `pose_raw_v1` is the canonical
downstream model pose source, aligned to the same AVI timeline, but is not
annotation evidence. Annotators must not use its landmarks or skeleton
visualizations, `pose_detected`, MediaPipe missingness or other derived
features, model predictions/confidence/errors/performance, Validation/Test
performance, or future alert outputs. Dataset activity names/filenames and
fall/non-fall metadata must not auto-fill the manual event-presence decision.

Official documentation and already-reviewed PNG/TXT semantics may inform
**protocol design** about the authors' `grounded_fall_state` concept. During
per-video AVI annotation, corresponding official PNG/TXT frame labels must
not be viewed to place boundaries. PNG ordinals, parsed filename indices,
image similarity and class transitions cannot be mapped to AVI coordinates.
The PNG physical timebase remains unresolved.

These are observable source **event boundaries**, not final training frame
classes or window targets. Nothing here defines `active_fall = 1`, a positive
span, a window overlap threshold, or how ambiguous boundaries affect model
labels. Later reviewed window-supervision design owns that conversion.

## Event presence and minimal boundary ontology

`event_presence` is an independent manual visual decision with exactly three
values: `fall_observed`, `no_fall_observed`, and `uncertain`. It must not be
copied from the activity filename, dataset binary label, or PNG/TXT class.
`no_fall_observed` makes all three fall boundaries `not_applicable`.
`uncertain` requires a reason flag, sets the two fall boundaries to
`unjudgeable`, and makes recovery `not_applicable`; it is never silently
converted to `no_fall_observed`. A `fall_observed` record requires applicable
transition and grounded boundaries, even when their exact coordinates are
censored or unjudgeable. The optional recovery event can be `not_observed`.

| Boundary | Operational meaning | Exclusion |
| --- | --- | --- |
| `fall_transition_start` | First visually defensible point in the continuous fall-related body transition that, considering the full event, proceeds into post-fall ground support. Base it on observable motion and loss of stable postural control. | Not clip start by default, first merely unusual frame, the whole fall-activity video, inferred intent, or PNG class-1 onset. |
| `grounded_start` | Start of the post-fall grounded state, once the main fall-related descent/impact transition has substantially completed and the body is clearly supported by floor/ground as a consequence of the fall. | The authors' grounded-state concept is semantic background, not an AVI coordinate source. |
| `recovery_start` | Optional start of a clear, sustained transition away from the post-fall grounded state toward recovery or leaving it. | Absence before clip end is valid; do not invent a recovery frame. |

No impact/contact boundary is introduced. The annotator must not infer medical
or psychological intent.

Each boundary has exactly one status:

| Status | Meaning | Coordinates |
| --- | --- | --- |
| `observed` | Visual evidence supports placing the boundary within the AVI. | Required earliest, preferred and latest plausible **source frame indices**. |
| `left_censored` | Relevant transition began before the first available AVI frame. | None; its unknown onset is not frame 0. |
| `right_censored` | Relevant transition/state onset has not been reached by the last available frame. | None; its unknown onset is not the last frame. |
| `not_observed` | Optional recovery does not occur in the available clip. | None; reserved for `recovery_start`. |
| `unjudgeable` | Boundary is conceptually relevant but visual evidence cannot place it reliably. A reason flag is required. | None. |
| `not_applicable` | Boundary does not apply under the event-presence decision. | None. |

For `observed`, require
`0 <= earliest_plausible_frame <= preferred_frame <= latest_plausible_frame < frame_count`.
The preferred frame is the annotator's best estimate; the interval contains
visually defensible alternatives. Observed preferred coordinates obey
`fall_transition_start <= grounded_start <= recovery_start` wherever both
members of an adjacent pair are observed. Adjacent uncertainty intervals may
overlap. Invalid order fails validation; nothing is auto-corrected. For a
non-observed status, no frame coordinate or timestamp may be fabricated.

Optional controlled reason flags are `occlusion`, `subject_out_of_frame`,
`motion_blur`, `ambiguous_transition_start`, `ambiguous_grounded_state`,
`ambiguous_recovery`, `insufficient_clip_context`, and `other`. They describe
difficulty but never alter a boundary automatically. An optional annotator
note can preserve context. No fixed uncertainty width is chosen.

## Record schema and timebase

A future source-annotation record contains:

- `annotation_record_id` and `protocol_version = stage32e_v1`;
- `source`: source-video identity, **neutral** `neutral_clip_id`, subject ID,
  split, optional source AVI SHA-256 (`null` when unavailable), positive source
  frame count, frozen `source_fps = 20`, and FPS provenance;
- `annotator`: pseudonymous annotator ID and session ID;
- `event_presence`, event-presence reason flags and one record for each of the
  three named boundaries, with controlled status and boundary reason flags;
- `creation`: UTC creation time, annotation run ID and tool version;
- optional `annotator_note`.

Stored provenance may include subject and source-video activity identity;
annotator-visible data is a separate, restricted view. The pure validator
checks only proposed Train records in memory. It does not read source files,
write annotations, select pilot clips, or adjudicate an event.

All boundary coordinates are zero-based AVI `frame_index` values. For an
observed boundary, `preferred_timestamp_ms` may be exported only as a
machine-derived convenience value, verified against
`round(preferred_frame * 1000 / 20)` (Python ties to even). Annotators enter
frames, never timestamps. Time is clip-relative, not Unix epoch. No PNG
timebase is introduced. Unknown boundary coordinates have no timestamp.

## Future blinded workflow and pilot

Stage 3.2f tooling should show original AVI visual frames, neutral clip ID,
`frame_index`, derived clip-relative time and playback controls. It should
hide fall/non-fall metadata, activity names where practical, revealing
filenames, same-video PNG/TXT labels, pose/skeleton data, MediaPipe missingness,
model predictions/scores/errors and other annotators' records. Provenance is
stored separately from this view. This is a requirement, not a UI build.

A later pilot must be frozen **before** annotation, use Train only, include
fall and non-fall videos plus hard-negative ADLs (intentional descent/posture
changes), cover multiple subjects and fall activities, and use neutral clip
identities. Selection must not use model predictions or pose missingness.
Exact clips and sample size remain `UNDECIDED`. Validation is unavailable for
protocol development. Test Subjects 6 and 7 remain sealed. At least two
annotators must independently use the same protocol version and subset,
blind to each other's records and to pose/model evidence. They annotate before
adjudication. Original records remain immutable; adjudication creates a
separate record referencing both original IDs rather than overwriting either.

## Descriptive agreement plan

The unit is a clip/event or a boundary jointly observed by two annotators,
never individual neighboring frames as independent samples. For
`event_presence`, report raw categorical agreement and Cohen's kappa when
defined; if kappa's chance-agreement denominator is zero, report it as
undefined rather than inventing a value.

For each boundary jointly `observed` by a pair, report:

1. `abs(preferred_frame_A - preferred_frame_B)` frames;
2. `abs(round(preferred_frame_A * 1000 / 20) - round(preferred_frame_B * 1000 / 20))` ms;
3. inclusive uncertainty-interval overlap, true when
   `max(earliest_A, earliest_B) <= min(latest_A, latest_B)`;
4. interval gap in frame-index units:
   `max(0, earliest_A - latest_B, earliest_B - latest_A)`. It is zero for
   overlap and the coordinate distance between nearest endpoints otherwise.

Report non-jointly-observed status pairs separately; do not assign them an
invented frame disagreement. These are descriptive metrics. Agreement and
adjudication acceptance thresholds remain `UNDECIDED` until real Train pilot
evidence is reviewed.

The full state ontology beyond these boundaries, final frame-class meanings,
final window-label rule, window overlap threshold, pilot size and selected
clips, agreement acceptance threshold, adjudication decision procedure and
adjudication acceptance threshold remain `UNDECIDED`. No real dataset content,
annotation, model-ready data or GUI is created in Stage 3.2e.
