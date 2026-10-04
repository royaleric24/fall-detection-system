# Stage 3.2j — Discrepancy Characterization and Pre-Adjudication Review Specification

**Project:** Real-Time Edge–Cloud Vision-Based Fall Detection System  
**Stage:** 3.2j — Discrepancy Characterization and Pre-Adjudication Review  
**Status:** Pre-review specification  
**Analysis type:** Blinded discrepancy-mechanism characterization  
**Purpose:** Characterize plausible mechanisms underlying Stage 3.2i annotation discrepancies and uncertainty differences without producing adjudicated labels or modifying any frozen annotation.

---

# 1. Frozen upstream provenance

Stage 3.2j MUST bind to the following frozen artifacts:

```text
Stage 3.2g execution protocol commit:
44d1543747f96fc8056adf2f71cdd6db3bdb07af

Stage 3.2h source annotation runtime commit:
725d6cd063c307fb50ddc2725ecfd1666c7f17b5

Stage 3.2h genuine annotation run:
stage32h-6747d0191488418d932955fda609ec0e

Raw-freeze manifest SHA-256:
750e46599836bafdf1644b48cd247a3a3ea16c3d99035f7c578682ea4d06fdb8

Stage 3.2i methodology freeze:
013418a92494b8350cce8856e339c7ad3f44514a

Stage 3.2i implementation freeze:
340ff0ca129d7829f85bc8fe2e5ae6173692bc05

Stage 3.2i analysis run:
stage32i-3d88c8a8f4ba4c188e93ab1864a1ba21

Stage 3.2i results freeze:
8f20d645becf3eb1603f450b47c57621352d57c2
```

The 24 original A01/A02 annotation records remain immutable.

Stage 3.2j MUST NOT modify:

- any original annotation record;
- Stage 3.2i agreement results;
- historical Stage 3.2e/3.2g/3.2h protocol definitions;
- Stage 3.2i metric definitions.

---

# 2. Research question

Stage 3.2j answers:

> What observable visual, semantic, procedural, censoring, coverage, or uncertainty-calibration mechanisms plausibly explain the annotation discrepancies and temporal uncertainty differences identified during Stage 3.2i?

Stage 3.2j explicitly does NOT answer:

> Which annotation is correct?

It does NOT produce:

- final event labels;
- final boundary statuses;
- final preferred frames;
- final plausible intervals;
- consensus annotations;
- adjudicated annotations;
- ML training targets.

---

# 3. Separation from adjudication

The following distinction is mandatory:

```text
discrepancy characterization
!=
adjudication
```

Allowed conclusion:

> Two visually distinct contact events plausibly explain why the candidates localized `grounded_start` differently.

Forbidden conclusion:

> Candidate X is correct and Candidate Y should be changed.

Stage 3.2j describes the structure and mechanism of disagreement.

Stage 3.2k, if later approved, will perform formal adjudication under a separately frozen protocol.

---

# 4. Review modes

Stage 3.2j supports exactly two review modes.

## 4.1 `STRICT_BLINDED_REVIEW`

Requirements:

- reviewer did not produce A01;
- reviewer did not produce A02;
- reviewer has not seen Stage 3.2i per-case results;
- reviewer has not been told annotator-specific aggregate patterns;
- reviewer has not participated in previous discrepancy discussion.

Preferred reviewer pseudonym:

```text
R01
```

This is the preferred mode.

---

## 4.2 `NONBLINDED_EXPLORATORY_REVIEW`

Use only if no eligible blinded reviewer is available.

This mode MUST be declared before review starts.

Results MUST be explicitly marked exploratory.

This mode MUST NOT be treated as equivalent to independent blind characterization.

A non-blinded review MUST NOT be the sole evidentiary basis for later adjudication.

---

# 5. Reviewer eligibility record

Before Stage 3.2j begins, record:

```text
reviewer_pseudonymous_id
review_mode

is_A01
is_A02

has_seen_stage32i_aggregate_results
has_seen_stage32i_per_case_results
has_seen_annotator_identity_patterns
has_participated_in_prior_discrepancy_discussion

eligibility_status
eligibility_notes
```

For `STRICT_BLINDED_REVIEW` all of the following MUST be false:

```text
is_A01
is_A02
has_seen_stage32i_aggregate_results
has_seen_stage32i_per_case_results
has_seen_annotator_identity_patterns
```

Otherwise strict blinding is invalid.

---

# 6. Canonical review unit

The physical review unit is:

```text
one source AVI clip
```

The logical review unit is:

```text
clip × review_target
```

where `review_target` is one of:

```text
fall_transition_start
grounded_start
recovery_coverage
event_presence
boundary_status
```

Multiple logical review units may reference the same AVI.

The reviewer should not reload identical video unnecessarily, but form outputs remain target-specific.

---

# 7. Reviewer-facing clip identity

Each source clip MUST receive an opaque Stage 3.2j alias:

```text
J001
J002
J003
...
```

The reviewer-facing UI MUST use this alias.

The mapping:

```text
Jxxx <-> record.source.neutral_clip_id
```

must be stored separately.

For strict blind review, the reviewer MUST NOT see `neutral_clip_id` unless technically unavoidable.

---

# 8. Review-case construction

The review set MUST be constructed automatically from the frozen Stage 3.2i artifacts before any Stage 3.2j video review begins.

The review set MUST then be frozen and hashed.

It MUST NOT be interactively changed because particular clips appear interesting.

---

# 9. Review groups

Every review target receives one of the following review groups.

## 9.1 `TEMPORAL_CALIBRATION`

Include ALL Stage 3.2i:

```text
observed <-> observed
```

pairs for:

```text
fall_transition_start
grounded_start
```

not only mismatch cases.

For the current frozen run, the expected counts are:

```text
fall_transition_start: 7
grounded_start: 7
```

This deliberately includes exact-match/context cases so the reviewer is not shown only selected disagreements.

---

## 9.2 `RECOVERY_COVERAGE`

Include all 12 pilot clips.

Purpose:

> Characterize why recovery temporal agreement could not be estimated.

No new recovery frame may be annotated.

---

## 9.3 `OBJECTIVE_DISCREPANCY_SUPPLEMENT`

Include any frozen Stage 3.2i discrepancy not already covered by the previous two groups, including if applicable:

```text
EVENT_PRESENCE_MISMATCH
BOUNDARY_STATUS_MISMATCH
```

For the current frozen run:

```text
event_presence mismatch count = 0
```

The single `recovery_start` status mismatch is already represented through the recovery review set and MUST NOT create a duplicate physical review item.

---

# 10. Duplicate consolidation

If a single `clip × boundary` carries several Stage 3.2i flags, such as:

```text
PREFERRED_FRAME_MISMATCH
INTERVAL_DISJOINT
```

it MUST remain one Stage 3.2j review case with multiple hidden source flags.

Do not create duplicated reviewer cases.

---

# 11. Reviewer-facing case list

Before Phase 1, the reviewer MAY see:

```text
review_case_id
clip_alias
review_target
```

The reviewer MUST NOT see:

```text
objective_stage32i_flags
A01 values
A02 values
candidate differences
IoU
interval widths
disagreement magnitude
```

---

# 12. Review state machine

Each review run MUST follow this irreversible state machine:

```text
CREATED
  ↓
CASE_SET_FROZEN
  ↓
PHASE1_OPEN
  ↓
PHASE1_LOCKED
  ↓
PHASE2_OPEN
  ↓
PHASE2_LOCKED
  ↓
PHASE3_UNBLINDED
  ↓
COMPLETE
```

Backward transitions are forbidden.

For example:

```text
PHASE2_OPEN -> PHASE1_OPEN
```

MUST NOT be possible.

---

# 13. Lock semantics

A phase lock MUST:

1. make the completed reviewer fields read-only;
2. calculate a deterministic content hash of the phase data;
3. record lock timestamp;
4. record reviewer ID;
5. record review run ID.

After Phase 1 lock, candidate annotations may be revealed.

After Phase 2 lock, A01/A02 identities may be unblinded.

Locked answers MUST NOT be overwritten.

---

# 14. Phase 1 — Blind visual characterization

Phase 1 occurs before any candidate annotation values are shown.

The reviewer inspects only the source AVI and frozen annotation definitions.

Goal:

> Characterize the visual structure and semantic clarity of the event without knowing how A01 or A02 annotated it.

---

# 15. Phase 1 video procedure

Every reviewed clip MUST first be viewed in full at normal speed.

Required:

```text
full_clip_viewed_at_1x = true
```

before the Phase 1 form can be submitted.

After the complete first pass, the reviewer MAY use:

```text
pause
replay
0.5x playback
frame forward
frame backward
```

Frame stepping MUST use canonical:

```text
0-based source AVI frame_index
```

---

# 16. Forbidden video processing

The review tool MUST NOT show:

- pose landmarks;
- model predictions;
- fall probabilities;
- bounding boxes;
- segmentation;
- optical flow;
- AI enhancement;
- generated/interpolated frames;
- post-hoc stabilization;
- Stage 3.2i discrepancy overlays during Phase 1.

The reviewer must inspect original source evidence.

---

# 17. Phase 1 visual-characteristic vocabulary

Field:

```text
visual_characteristics
```

Type:

```text
array[string]
```

Allowed values:

```text
ABRUPT_TRANSITION
GRADUAL_TRANSITION
MULTI_STAGE_TRANSITION
MULTIPLE_PLAUSIBLE_VISUAL_CUES

PARTIAL_OCCLUSION
SEVERE_OCCLUSION
SUBJECT_PARTLY_OUT_OF_FRAME

MOTION_BLUR
LOW_VISUAL_CLARITY

FURNITURE_OR_OBJECT_CONTACT
MULTIPLE_BODY_CONTACT_EVENTS
SLIDING_OR_PROGRESSIVE_SETTLING
BOUNCE_OR_RECONTACT

CLIP_START_TRUNCATION
CLIP_END_TRUNCATION

NO_OBVIOUS_VISUAL_AMBIGUITY
OTHER
```

Multiple values are permitted.

Contradictory combinations SHOULD be rejected where obvious.

For example:

```text
NO_OBVIOUS_VISUAL_AMBIGUITY
```

SHOULD NOT coexist with:

```text
SEVERE_OCCLUSION
```

without an explicit rationale.

---

# 18. Phase 1 semantic-clarity field

Field:

```text
boundary_semantic_clarity
```

Allowed values:

```text
clear
potentially_ambiguous
strongly_ambiguous
not_assessable
```

Meaning:

- `clear`: frozen boundary definition maps naturally to a visually identifiable event;
- `potentially_ambiguous`: more than one interpretation is plausible;
- `strongly_ambiguous`: operational definition does not map cleanly to a unique observable transition;
- `not_assessable`: available visual evidence is insufficient to make this characterization.

This is NOT an annotation-quality score.

---

# 19. Phase 1 notes

Field:

```text
phase1_visual_rationale
```

Type:

```text
string
```

Required unless:

```text
NO_OBVIOUS_VISUAL_AMBIGUITY
```

is the sole selected characteristic.

The rationale should describe visible event structure.

Allowed example:

> Initial lower-body contact is followed by continued torso descent and later stable settling.

Forbidden example:

> The correct grounded frame is 118.

Phase 1 free text MUST NOT provide:

- a preferred frame;
- an E/P/L interval;
- a final label;
- a candidate ranking.

This prevents Phase 1 from becoming an undocumented third annotation.

---

# 20. Phase 1 form schema

Each Phase 1 row MUST contain:

```text
review_run_id
review_case_id
clip_alias
review_target

full_clip_viewed_at_1x

slow_playback_used
frame_step_used

visual_characteristics

boundary_semantic_clarity
phase1_visual_rationale

phase1_completed
phase1_locked
phase1_lock_timestamp
phase1_content_sha256
```

Reviewer-facing Phase 1 data MUST NOT contain A01/A02/candidate values.

---

# 21. Recovery coverage review

`recovery_coverage` uses a specialized Phase 1 form.

It does NOT request a recovery frame.

Fields:

```text
post_fall_period_visible

recovery_opportunity_within_clip

recovery_behavior_visibility

clip_end_truncation_relevant

recovery_visibility_obstruction

recovery_coverage_characteristics

recovery_coverage_rationale
```

---

# 22. Recovery coverage allowed values

## `post_fall_period_visible`

```text
yes
no
unclear
```

## `recovery_opportunity_within_clip`

```text
yes
no
unclear
```

## `recovery_behavior_visibility`

```text
none_visible
partially_visible
clearly_visible
unclear
```

## `clip_end_truncation_relevant`

```text
yes
no
unclear
```

## `recovery_visibility_obstruction`

```text
none
partial
severe
unclear
```

---

# 23. Recovery coverage characteristics

Field:

```text
recovery_coverage_characteristics
```

Allowed multi-select values:

```text
SUBJECT_REMAINS_GROUNDED
POST_FALL_PERIOD_VISIBLE

RECOVERY_BEHAVIOR_PARTIALLY_VISIBLE
RECOVERY_BEHAVIOR_CLEARLY_VISIBLE

CLIP_ENDS_BEFORE_MEANINGFUL_RECOVERY_OPPORTUNITY
CLIP_ENDS_DURING_POSSIBLE_RECOVERY

RECOVERY_VISIBILITY_OBSCURED
RECOVERY_STATUS_SEMANTICALLY_AMBIGUOUS

COVERAGE_UNCLEAR
OTHER
```

These describe coverage only.

They MUST NOT create a new `recovery_start` annotation.

---

# 24. Phase 1 prohibition on candidate information

Before:

```text
PHASE1_LOCKED
```

the backend and UI MUST NOT expose:

```text
Candidate X
Candidate Y
A01
A02
preferred_frame
earliest_plausible_frame
latest_plausible_frame
stage32i flags
frame difference
IoU
interval widths
```

This must be enforced by the application logic, not merely hidden visually with CSS.

---

# 25. Candidate permutation

Candidate identities in Phase 2 MUST be anonymized as:

```text
Candidate X
Candidate Y
```

X/Y mapping must be fixed before Phase 2 begins.

To prevent manual selection of orientation, use the following deterministic randomized scheme.

At review-run creation generate:

```text
candidate_permutation_seed
```

as a random UUID.

For every `review_case_id`, compute:

```text
SHA256(candidate_permutation_seed + "|" + review_case_id)
```

Use the least-significant bit of the resulting digest to decide:

```text
0:
X = A01
Y = A02

1:
X = A02
Y = A01
```

The seed MUST be stored only in the administrative manifest.

The reviewer-facing UI MUST NOT reveal it.

Also store:

```text
candidate_permutation_mapping_sha256
```

so the frozen mapping can later be verified.

---

# 26. Phase 2 — Anonymous pairwise mechanism review

Phase 2 begins only after Phase 1 is locked.

The reviewer may now see Candidate X/Y annotations relevant to the target.

Annotator identities remain hidden.

---

# 27. Phase 2 display rules

For temporally comparable boundaries, show:

```text
Candidate X:
earliest_plausible_frame
preferred_frame
latest_plausible_frame

Candidate Y:
earliest_plausible_frame
preferred_frame
latest_plausible_frame
```

The candidates MAY be overlaid on the original AVI timeline.

Do NOT show:

```text
A01
A02
signed difference
absolute difference
aggregate annotator statistics
```

unless the difference is visually implicit from the candidate markers themselves.

---

# 28. Phase 2 for status discrepancies

Where Stage 3.2i contains a status discrepancy, show only:

```text
Candidate X status
Candidate Y status
```

plus source AVI.

Do not ask for a replacement status.

---

# 29. Phase 2 pairwise-relation field

Field:

```text
candidate_relation
```

Allowed values:

```text
SAME_PHYSICAL_SUBEVENT_DIFFERENT_POINT

DIFFERENT_PLAUSIBLE_PHYSICAL_SUBEVENTS

SIMILAR_EVENT_REGION_DIFFERENT_INTERVAL_WIDTH

STATUS_SEMANTIC_DIFFERENCE

CANDIDATES_BOTH_PLAUSIBLE_FROM_VISIBLE_EVIDENCE

CANDIDATE_RELATION_NOT_ASSESSABLE

OTHER
```

This characterizes how Candidate X and Y relate.

It does NOT rank them.

---

# 30. Discrepancy-mechanism taxonomy

Field:

```text
mechanisms
```

Type:

```text
array[object]
```

Each object contains:

```text
code
confidence
```

Allowed mechanism codes:

```text
SEMANTIC_BOUNDARY_AMBIGUITY

GRADUAL_TRANSITION
MULTI_STAGE_TRANSITION
MULTIPLE_PLAUSIBLE_VISUAL_CUES

VISUAL_OCCLUSION
OUT_OF_FRAME
MOTION_BLUR_OR_LOW_CLARITY

OBJECT_OR_FURNITURE_INTERACTION
MULTIPLE_CONTACT_EVENTS
PROGRESSIVE_SETTLING_OR_SLIDING

CLIP_TRUNCATION_OR_CENSORING

UNCERTAINTY_WIDTH_CALIBRATION_DIFFERENCE

STATUS_INTERPRETATION_DIFFERENCE

POSSIBLE_UI_OR_WORKFLOW_EFFECT

POSSIBLE_COORDINATE_OR_PLAYBACK_PROBLEM

NO_CLEAR_MECHANISM_IDENTIFIED

OTHER
```

Multiple mechanisms may be selected.

---

# 31. Mechanism confidence

Allowed:

```text
low
medium
high
```

Confidence refers only to:

> How strongly the visible evidence supports the proposed discrepancy mechanism.

It does NOT refer to:

> How confident the reviewer is that Candidate X or Candidate Y is correct.

---

# 32. Protocol clarification candidate

Field:

```text
protocol_clarification_candidate
```

Type:

```text
boolean
```

If true, require:

```text
protocol_clarification_area
```

Allowed values:

```text
BOUNDARY_DEFINITION
UNCERTAINTY_INTERVAL_SEMANTICS
CENSORING_RULE
OBSERVABILITY_RULE
RECOVERY_APPLICABILITY
ANNOTATION_UI_INSTRUCTION
OTHER
```

Also require:

```text
protocol_clarification_rationale
```

This records a candidate problem.

It MUST NOT modify the protocol during Stage 3.2j.

---

# 33. Phase 2 rationale

Field:

```text
phase2_mechanism_rationale
```

Required.

Good example:

> Candidate markers correspond to initial knee contact and later torso settling; both are visually distinct moments in a gradual multi-stage transition.

Forbidden example:

> Candidate X is correct.

---

# 34. Forbidden Phase 2 fields

The schema MUST NOT contain:

```text
winner

preferred_candidate

correct_candidate

correct_annotator

final_event_presence

final_boundary_status

final_preferred_frame

final_earliest_frame

final_latest_frame

consensus_frame

adjudicated_label

ground_truth

reviewer_preferred_frame
```

The review tool MUST NOT ask equivalent questions under different wording.

---

# 35. Phase 2 form schema

Each Phase 2 row MUST contain:

```text
review_run_id
review_case_id

candidate_relation

mechanisms

protocol_clarification_candidate
protocol_clarification_area
protocol_clarification_rationale

phase2_mechanism_rationale

phase2_completed
phase2_locked
phase2_lock_timestamp
phase2_content_sha256
```

Candidate values may be stored in a separate immutable candidate-display table rather than repeated in the reviewer response.

---

# 36. Phase 2 eligibility

For `TEMPORAL_CALIBRATION` cases:

```text
phase2_required = true
```

even if the preferred frames happen to match.

Reason:

> Interval-calibration differences may still exist in exact preferred-frame matches.

For `RECOVERY_COVERAGE` cases:

```text
phase2_required = false
```

unless the frozen Stage 3.2i outputs contain an objective recovery-related annotation discrepancy for that clip.

The current single recovery status mismatch therefore receives Phase 2 anonymous candidate review.

The remaining recovery cases remain coverage-characterization cases only.

---

# 37. Phase 2 lock

Before candidate identities are revealed:

```text
all required Phase 2 rows
```

must be completed and locked.

No partial identity unblinding is allowed.

---

# 38. Phase 3 — Identity unblinding

Phase 3 begins only after Phase 2 is completely locked.

Candidate X/Y may now be mapped back to:

```text
A01
A02
```

for methodology-lead analysis.

Phase 3 MUST NOT reopen Phase 1 or Phase 2 forms.

---

# 39. Phase 3 is not a new reviewer annotation phase

Stage 3.2j Phase 3 SHOULD primarily be machine-generated.

It combines:

```text
frozen Stage 3.2i metrics
+
frozen Phase 1 results
+
frozen Phase 2 mechanism classifications
+
candidate X/Y mapping
```

to describe annotator-specific patterns.

The reviewer SHOULD NOT receive editable fields that allow reinterpreting earlier judgments after identity revelation.

---

# 40. Permitted Phase 3 analyses

Examples:

```text
mechanism counts by boundary

mechanism counts by annotator orientation

association between
UNCERTAINTY_WIDTH_CALIBRATION_DIFFERENCE
and A01/A02 interval-width orientation

frequency of candidate_relation categories

semantic-ambiguity frequency by boundary

visual-characteristic frequency by boundary

recovery coverage pattern counts
```

These are descriptive only.

---

# 41. Forbidden Phase 3 conclusions

Forbidden:

> A02 is a worse annotator.

Forbidden:

> A01's narrow intervals are correct.

Forbidden:

> A02 should change all grounded labels.

Allowed:

> Candidate pairs involving wider A02 intervals were frequently characterized as uncertainty-width calibration differences.

---

# 42. Grounded-start targeted characterization

For `grounded_start`, Stage 3.2j must be capable of identifying whether disagreement plausibly relates to:

```text
initial body contact

later torso/pelvis contact

multiple body-part contacts

stable grounded state

sliding after initial contact

progressive settling

furniture/object interaction

gradual transition

different uncertainty-width interpretation
```

No new operational definition may be created yet.

---

# 43. Fall-transition targeted characterization

For `fall_transition_start`, characterize:

```text
pre-fall instability

committed falling motion

first clear departure from stable posture

gradual onset

multiple plausible onset cues

uncertainty-width interpretation
```

Again, no new definition is introduced.

---

# 44. Recovery targeted characterization

The recovery review must answer only coverage/observability questions such as:

```text
Was sufficient post-fall footage present?

Was meaningful recovery behavior visible?

Did the clip terminate before recovery could reasonably be evaluated?

Was visibility inadequate?

Does the frozen recovery definition appear difficult to apply?
```

It MUST NOT ask:

```text
What frame is recovery_start?
```

---

# 45. Coordinate-integrity anomaly

If the reviewer encounters evidence suggesting:

```text
frame coordinate mismatch
playback indexing mismatch
source-video mismatch
candidate marker synchronization error
```

they MUST select:

```text
POSSIBLE_COORDINATE_OR_PLAYBACK_PROBLEM
```

and the affected case enters:

```text
TECHNICAL_HOLD
```

The case MUST NOT proceed toward adjudication until separately investigated.

Stage 3.2j must not “interpret around” a potential provenance or coordinate problem.

---

# 46. Reviewer discussion restriction

During active Stage 3.2j review:

- R01 must not discuss candidate cases with A01;
- R01 must not discuss candidate cases with A02;
- A01 and A02 must not negotiate their original answers;
- methodology lead must not tell R01 what pattern Stage 3.2i found.

Such discussion would contaminate blind discrepancy characterization.

---

# 47. Original annotators

A01 and A02 may later participate in Stage 3.2k adjudication if the adjudication protocol permits it.

Their participation in Stage 3.2j review is not preferred.

If either serves as reviewer, review mode MUST be:

```text
NONBLINDED_EXPLORATORY_REVIEW
```

---

# 48. Required output artifacts

Stage 3.2j must eventually generate:

```text
stage32j_review_manifest.json

stage32j_reviewer_eligibility.json

stage32j_review_case_index.csv

stage32j_candidate_mapping.json
or equivalent protected mapping artifact

stage32j_phase1_visual_characterization.csv

stage32j_recovery_coverage_review.csv

stage32j_phase2_candidate_display.csv

stage32j_phase2_discrepancy_mechanisms.csv

stage32j_phase3_identity_pattern_summary.csv

stage32j_execution_audit.json

stage32j_report.md
```

The actual repository location may follow existing artifact conventions.

---

# 49. Protected candidate mapping

`stage32j_candidate_mapping.json` contains sensitive review-blinding information.

It MUST NOT be exposed through the reviewer UI before Phase 3.

At minimum it records:

```text
review_case_id
candidate_x_source
candidate_y_source
```

where sources are:

```text
A01
A02
```

Its content hash must be recorded in the main manifest.

---

# 50. Review manifest schema

At minimum:

```text
stage
specification_version

stage32j_review_run_id
review_mode

reviewer_pseudonymous_id

source_annotation_run_id
raw_freeze_sha256

stage32i_analysis_run_id
stage32i_results_commit

stage32j_protocol_commit
stage32j_tool_commit

review_case_set_sha256

candidate_permutation_seed
candidate_permutation_mapping_sha256

case_count

phase1_required_count
phase2_required_count

phase1_lock_timestamp
phase2_lock_timestamp
phase3_unblind_timestamp

source_artifact_paths
output_artifact_paths
```

If reviewer access to the manifest would reveal blinding information, use:

```text
admin manifest
+
reviewer-safe manifest
```

rather than exposing protected fields.

---

# 51. Review-case index schema

Canonical columns:

```text
review_case_id

clip_alias
neutral_clip_id

review_target
review_group

objective_stage32i_flags

phase1_required
phase2_required
phase3_eligible

source_video_reference

case_status
```

`neutral_clip_id` and `objective_stage32i_flags` MUST be omitted from the reviewer-facing representation before the appropriate phase.

---

# 52. Case status vocabulary

Allowed:

```text
NOT_STARTED

PHASE1_IN_PROGRESS
PHASE1_LOCKED

PHASE2_IN_PROGRESS
PHASE2_LOCKED

PHASE3_UNBLINDED

COMPLETE

TECHNICAL_HOLD
```

No other implicit status strings should be introduced.

---

# 53. Phase validation invariants

Before Phase 2 opens:

```text
all required Phase 1 records exist
all required Phase 1 records are locked
all Phase 1 content hashes verify
```

Before Phase 3 opens:

```text
all required Phase 2 records exist
all required Phase 2 records are locked
all Phase 2 content hashes verify
candidate mapping hash verifies
```

If any invariant fails:

```text
HOLD
```

---

# 54. Immutability requirements

The following MUST remain unchanged throughout Stage 3.2j:

```text
24 Stage 3.2h original annotation records

Stage 3.2i source outputs

Stage 3.2i results freeze commit

Stage 3.2j locked Phase 1 records

Stage 3.2j locked Phase 2 records
```

Stage 3.2j must reverify raw annotation integrity before and after the review execution.

---

# 55. Held-out data restriction

Stage 3.2j may access only source AVI clips belonging to the frozen Stage 3.2h pilot set.

It MUST NOT traverse or inspect:

```text
Validation subjects

Test Subject 6

Test Subject 7
```

No broader dataset discovery is permitted.

---

# 56. No new agreement metrics

Stage 3.2j MUST NOT add post-hoc metrics such as:

```text
new temporal tolerance accuracy

new IoU threshold

new ICC

new composite agreement score

new annotator quality score
```

Stage 3.2j interprets the frozen Stage 3.2i findings.

It does not redefine Stage 3.2i.

---

# 57. No discrepancy severity score

The review MUST NOT produce:

```text
minor disagreement
moderate disagreement
major disagreement
severity score
annotation quality score
annotator quality score
```

Different discrepancy mechanisms are not assumed to lie on one valid scalar scale.

---

# 58. Protocol-change firewall

If Stage 3.2j discovers a protocol ambiguity, record it as:

```text
protocol_clarification_candidate = true
```

Do not immediately modify the historical annotation protocol.

Any future protocol revision requires a separately frozen versioned stage.

The existing annotations remain interpreted under the protocol that generated them.

---

# 59. Adjudication firewall

Stage 3.2j MUST NOT create any field semantically equivalent to:

```text
winner
correct
final
consensus
ground_truth
adjudicated
preferred_candidate
replacement_label
```

Any such field constitutes a methodology violation.

---

# 60. Stage 3.2j Gate type

Stage 3.2j uses a:

```text
PROCESS / METHODOLOGY GATE
```

PASS does not mean the annotators now agree.

PASS means:

> The discrepancy mechanisms and recovery coverage limitations were characterized systematically under the frozen review protocol without informal adjudication.

---

# 61. Stage 3.2j PASS conditions

Stage 3.2j may be marked COMPLETE / PASS only when:

1. protocol was frozen before genuine review;
2. reviewer eligibility was recorded;
3. review mode was declared before genuine review;
4. review case set was frozen before video review;
5. candidate permutation was frozen before Phase 2;
6. all required Phase 1 cases were completed;
7. Phase 1 was locked before candidate annotations were shown;
8. all required Phase 2 cases were completed;
9. Phase 2 was locked before A01/A02 identities were revealed;
10. locked records remained immutable;
11. all required recovery coverage cases were reviewed;
12. no original annotation was modified;
13. no adjudicated value was created;
14. no protocol was rewritten during review;
15. no downstream ML supervision was designed;
16. raw annotation integrity passed before and after review;
17. no Validation/Test subject was accessed;
18. all required outputs were generated and frozen.

---

# 62. Stage 3.2j HOLD conditions

HOLD is mandatory if:

- candidate values are exposed before Phase 1 lock;
- A01/A02 identity is exposed before Phase 2 lock;
- review-case selection changes after clip inspection without a formally frozen amendment;
- locked Phase 1/2 values are edited;
- reviewer supplies a replacement frame/label;
- original annotators begin negotiating disagreements;
- source annotation records change;
- Stage 3.2i results change;
- technical coordinate/playback anomalies are detected and unresolved;
- held-out subjects are accessed;
- adjudication begins before Stage 3.2j freeze.

---

# 63. Required report structure

`stage32j_report.md` must include:

## A. Provenance

- Stage 3.2i result identity;
- Stage 3.2j specification identity;
- tool identity;
- review run identity;
- review mode;
- reviewer pseudonym.

## B. Review-set completeness

Counts for:

```text
TEMPORAL_CALIBRATION
RECOVERY_COVERAGE
OBJECTIVE_DISCREPANCY_SUPPLEMENT
```

## C. Visual characteristics

Boundary-specific counts and descriptions.

## D. Discrepancy mechanisms

Separate:

```text
fall_transition_start
grounded_start
recovery-related status/coverage
```

## E. Uncertainty calibration

Describe whether interval-width differences appear:

```text
systematic
case-specific
associated with identifiable visual mechanisms
unexplained
```

without declaring one annotator correct.

## F. Recovery coverage

Explain why jointly observed recovery boundaries were absent or rare.

## G. Protocol clarification candidates

List frozen review findings requiring possible future protocol discussion.

Do not revise protocol here.

## H. Technical holds

List any coordinate/playback/source anomalies.

## I. Adjudication readiness

State whether the discrepancy structure is sufficiently characterized to design Stage 3.2k.

---

# 64. Required limitation statement

The report MUST contain:

> Stage 3.2j characterizes plausible mechanisms underlying annotation discrepancies and temporal uncertainty differences. It does not determine which annotator is correct, does not produce consensus or adjudicated annotations, and does not define downstream ML supervision. All original Stage 3.2h annotations and frozen Stage 3.2i agreement results remain unchanged.

If the review mode is `NONBLINDED_EXPLORATORY_REVIEW`, also include:

> The reviewer had prior exposure to Stage 3.2i findings and therefore this review is considered exploratory rather than strictly blinded. Its mechanism classifications should not be treated as independent blinded evidence.

---

# 65. Pre-implementation checklist

Before Codex builds the Stage 3.2j review system:

```text
[ ] Specification complete
[ ] Reviewer eligibility rules frozen
[ ] Review-mode rules frozen
[ ] Review-case construction frozen
[ ] Phase 1 schema frozen
[ ] Recovery schema frozen
[ ] Candidate-permutation method frozen
[ ] Phase 2 schema frozen
[ ] Phase-lock semantics frozen
[ ] Phase 3 behavior frozen
[ ] Forbidden fields frozen
[ ] PASS/HOLD criteria frozen
```

No genuine Stage 3.2j review may occur before the implementation is separately tested and frozen.

---

# 66. Stage transition

After this specification is frozen:

```text
Stage 3.2j specification freeze
        ↓
review-tool implementation
        ↓
synthetic UI/workflow tests only
        ↓
review-tool implementation freeze
        ↓
genuine Phase 1 review
        ↓
Phase 1 freeze
        ↓
genuine Phase 2 review
        ↓
Phase 2 freeze
        ↓
Phase 3 identity unblinding
        ↓
Stage 3.2j results freeze
        ↓
methodological review
        ↓
only then consider Stage 3.2k formal adjudication
```

No genuine review data should be entered before both the specification and review implementation are frozen.