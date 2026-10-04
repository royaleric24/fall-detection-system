# Stage 3.2i — Inter-Annotator Agreement Analysis Specification

**Project:** Real-Time Edge–Cloud Vision-Based Fall Detection System  
**Stage:** 3.2i — Inter-Annotator Agreement Analysis  
**Status:** Pre-analysis specification  
**Analysis type:** Independent dual-annotation agreement analysis  
**Purpose:** Quantify and characterize A01–A02 agreement before adjudication or downstream supervision design.

---

## 1. Frozen provenance

Stage 3.2i MUST bind explicitly to the following completed Stage 3.2h artifacts:

```text
Stage 3.2g execution protocol commit:
44d1543747f96fc8056adf2f71cdd6db3bdb07af

Stage 3.2h readiness/runtime commit:
725d6cd063c307fb50ddc2725ecfd1666c7f17b5

Genuine run:
stage32h-6747d0191488418d932955fda609ec0e

A01 FINALIZED:
12 / 12

A02 FINALIZED:
12 / 12

Original annotation records:
24 / 24

Stage 3.2h completion gate:
PASS

Raw freeze:
EXECUTED and VERIFIED

Raw-freeze mechanism:
write_raw_freeze_new

Raw-freeze manifest SHA-256:
750e46599836bafdf1644b48cd247a3a3ea16c3d99035f7c578682ea4d06fdb8
```

The 24 original annotation records are immutable research records.

Stage 3.2i MAY read them only after verifying the existing raw-freeze manifest.

Stage 3.2i MUST NOT modify, normalize, rewrite, migrate, repair, rename, reorder, or regenerate any of the 24 frozen records.

The analysis implementation itself may exist at a later Git commit. Therefore the analysis manifest MUST distinguish:

```text
execution_protocol_commit
source_annotation_runtime_commit
raw_freeze_manifest_sha256
analysis_code_commit
analysis_run_id
```

`analysis_code_commit` MUST NOT be confused with the Stage 3.2h runtime commit.

---

# 2. Scope and research question

Stage 3.2i answers only:

> To what extent do A01 and A02 independently agree on event presence, boundary observability/status, preferred temporal location, and temporal uncertainty for the same 12 pilot clips?

Stage 3.2i does NOT answer:

- which annotator is correct;
- what the final annotation should be;
- what the final training label should be;
- how frames/windows should be supervised;
- how uncertain labels should enter an ML loss;
- which temporal tolerance is acceptable for the downstream model;
- whether disagreement should be corrected;
- whether one annotator's interpretation should override the other.

Stage 3.2i is a measurement stage, not an adjudication stage.

---

# 3. Experimental unit and pairing

## 3.1 Experimental unit

The fundamental paired experimental unit is:

```text
one pilot clip
```

There are exactly:

\[
N=12
\]

paired clips.

The 24 annotation files do NOT constitute 24 independent samples.

Similarly, the three temporal boundaries within a clip are repeated measurements associated with the same clip and MUST NOT be treated as independent samples to artificially increase sample size.

---

## 3.2 Pairing rule

For each canonical pilot clip identity, exactly:

```text
1 A01 FINAL record
+
1 A02 FINAL record
```

must exist.

The pairing key MUST be the canonical clip/sample identity already defined by the frozen Stage 3.2g/3.2h protocol.

The implementation MUST NOT infer pairing from:

- file order;
- directory listing order;
- timestamps;
- annotation completion time;
- approximate filename matching;
- frame counts;
- annotation contents.

If the actual schema uses a field other than `clip_id`, the implementation may map that existing canonical field internally to:

```text
annotation_unit_id
```

but MUST record the source field used.

The following invariant MUST hold:

```text
set(A01 annotation_unit_id)
==
set(A02 annotation_unit_id)
```

with exactly 12 unique IDs on each side.

Duplicate, missing, or ambiguous pairing is a hard Gate failure.

---

# 4. Allowed input domain

Stage 3.2i MAY access only:

1. the frozen Stage 3.2h annotation records;
2. their raw-freeze manifest;
3. metadata required to identify the corresponding 12 pilot clips;
4. existing frozen frame-count/FPS metadata if required for integrity checking;
5. code and documentation needed to execute Stage 3.2i.

It MUST NOT access:

- Validation subjects;
- Test Subject 6;
- Test Subject 7;
- any sealed held-out test data;
- annotations outside the genuine Stage 3.2h pilot run;
- later adjudicated annotations, if any exist in the repository.

Stage 3.2i SHOULD NOT decode or visually inspect video content.

This stage evaluates annotation agreement from the already-frozen annotations; it does not re-annotate the source video.

---

# 5. Canonical vocabularies

## 5.1 `event_presence`

Fixed nominal vocabulary and ordering:

```text
1. fall_observed
2. no_fall_observed
3. uncertain
```

These are nominal categories.

No ordinal relationship is assumed.

In particular:

```text
uncertain ≠ 0.5 × fall_observed
```

and `uncertain` MUST NOT receive partial agreement credit.

---

## 5.2 Boundary types

Fixed ordering:

```text
1. fall_transition_start
2. grounded_start
3. recovery_start
```

Each boundary type MUST be analyzed separately.

---

## 5.3 Boundary status

Fixed nominal vocabulary and ordering:

```text
1. observed
2. left_censored
3. right_censored
4. not_observed
5. unjudgeable
6. not_applicable
```

These six statuses MUST remain distinct.

The following collapses are forbidden:

```text
left_censored + right_censored → censored
not_observed + not_applicable → absent
unjudgeable + uncertain → uncertain
```

No such derived collapsing may be used in primary Stage 3.2i metrics.

---

# 6. Canonical temporal coordinate

The only canonical temporal coordinate is:

```text
0-based source AVI frame_index
```

AVI frame rate:

\[
20\ FPS
\]

Therefore:

\[
1\ frame = 50\ ms
\]

Temporal differences MUST first be calculated in integer frames.

Milliseconds are a derived reporting unit:

\[
difference_{ms}=difference_{frames}\times50
\]

No timestamp reconstructed from PNG order, TXT order, filesystem time, or any other source may replace AVI `frame_index`.

---

# 7. Pre-computation integrity Gate

No agreement metric may be calculated until all checks below PASS.

## 7.1 Provenance validation

MUST verify:

- genuine run ID matches exactly;
- raw-freeze manifest exists;
- raw-freeze manifest SHA-256 matches exactly;
- raw freeze verifies successfully using the existing mechanism;
- records belong to A01 or A02 as expected;
- all records are FINALIZED;
- record count is exactly 24;
- A01 count is exactly 12;
- A02 count is exactly 12.

---

## 7.2 Pairing validation

MUST verify:

```text
12 unique A01 annotation units
12 unique A02 annotation units
12 exact A01/A02 pairs
0 unmatched A01 records
0 unmatched A02 records
0 duplicate annotation units
```

---

## 7.3 Vocabulary validation

Every `event_presence` value MUST belong to the frozen three-category vocabulary.

Every boundary status MUST belong to the frozen six-category vocabulary.

Unknown or newly introduced values constitute a Gate failure.

Stage 3.2i MUST NOT silently map unknown values into existing categories.

---

## 7.4 Observed-boundary structural validation

Whenever:

```text
boundary.status == observed
```

all three values MUST exist as valid integer 0-based frame indices:

```text
earliest_plausible_frame
preferred_frame
latest_plausible_frame
```

and MUST satisfy:

\[
0\le E\le P\le L
\]

where:

```text
E = earliest_plausible_frame
P = preferred_frame
L = latest_plausible_frame
```

If frozen clip frame-count metadata is already available, MUST additionally validate:

\[
L < number\_of\_frames
\]

Stage 3.2i MUST NOT decode previously unseen video merely to obtain this value.

---

## 7.5 Non-observed boundary frame handling

For:

```text
left_censored
right_censored
not_observed
unjudgeable
not_applicable
```

temporal frame fields MUST NOT be used in temporal agreement calculations.

If the frozen Stage 3.2g schema requires these fields to be null, that existing rule MUST be validated.

If the frozen schema permits stored auxiliary frame values, Stage 3.2i MUST leave them untouched but MUST ignore them for temporal comparison.

Stage 3.2i MUST NOT invent a pseudo-frame such as:

```text
left_censored  → frame 0
right_censored → final frame
not_observed   → -1
unjudgeable    → midpoint
```

---

# 8. Analysis Layer A — `event_presence`

## 8.1 Denominator

All 12 paired clips MUST participate.

\[
N_{event}=12
\]

Missing values MUST NOT be listwise-deleted.

A missing frozen value is an integrity failure.

---

## 8.2 Confusion matrix

Generate a fixed 3 × 3 matrix.

Rows:

```text
A01
```

Columns:

```text
A02
```

Category order MUST be:

```text
fall_observed
no_fall_observed
uncertain
```

All nine cells MUST be reported, including zeros.

Invariant:

\[
\sum_{i=1}^{3}\sum_{j=1}^{3}n_{ij}=12
\]

---

## 8.3 Exact agreement

Define:

\[
P_o=
\frac{\sum_k n_{kk}}{12}
\]

Report both:

```text
exact_agreement_count
exact_agreement_rate
```

For example:

```text
10 / 12
0.8333
```

`uncertain ↔ uncertain` is exact agreement.

`uncertain ↔ fall_observed` is disagreement.

`uncertain ↔ no_fall_observed` is disagreement.

There is no partial credit.

---

## 8.4 Annotator marginal distributions

Separately report for A01 and A02:

```text
fall_observed count/rate
no_fall_observed count/rate
uncertain count/rate
```

Each rate uses denominator:

\[
12
\]

This is required because asymmetric use of `uncertain` or another category is itself methodologically important.

---

# 9. Cohen's κ for `event_presence`

Cohen's κ is a secondary descriptive statistic.

It MUST NOT replace the confusion matrix or exact agreement.

For category \(k\):

\[
p_{A01,k}=
\frac{n_{A01,k}}{12}
\]

\[
p_{A02,k}=
\frac{n_{A02,k}}{12}
\]

Expected chance agreement:

\[
P_e^{\kappa}
=
\sum_{k=1}^{3}
p_{A01,k}p_{A02,k}
\]

Then:

\[
\kappa=
\frac{P_o-P_e^{\kappa}}
{1-P_e^{\kappa}}
\]

If:

\[
1-P_e^{\kappa}=0
\]

the result MUST be:

```text
NA
```

with an explicit reason such as:

```text
undefined_due_to_degenerate_marginals
```

It MUST NOT be coerced to 0 or 1.

No weighted κ shall be calculated because `event_presence` is nominal rather than ordinal.

Because only 12 paired clips are available, κ MUST be described as an exploratory/descriptive coefficient rather than evidence that population-level reliability has been established.

No PASS/FAIL rule may be based on conventional κ interpretation labels such as:

```text
fair
moderate
substantial
almost perfect
```

---

# 10. Gwet's AC1 for `event_presence`

Gwet's AC1 is a secondary sensitivity statistic intended to complement Cohen's κ when marginal prevalence is asymmetric.

The number of possible categories is fixed:

\[
K=3
\]

even if one category happens not to occur in the 12-clip sample.

Define pooled category prevalence:

\[
\pi_k
=
\frac{
n_{A01,k}+n_{A02,k}
}{
2N
}
\]

where:

\[
N=12
\]

Chance agreement is:

\[
P_e^{AC1}
=
\frac{1}{K-1}
\sum_{k=1}^{K}
\pi_k(1-\pi_k)
\]

and:

\[
AC1=
\frac{P_o-P_e^{AC1}}
{1-P_e^{AC1}}
\]

This is the standard nominal multicategory AC1 formulation.

If the denominator is mathematically undefined, report `NA` plus reason.

AC1 MUST NOT be selected instead of κ merely because it produces a numerically larger value.

The report MUST present:

```text
observed agreement
Cohen's kappa
Gwet's AC1
```

together.

No inferential p-value is required.

No primary PASS/FAIL threshold shall be attached to AC1.

---

# 11. Analysis Layer B — boundary status agreement

The three boundaries MUST be analyzed independently:

```text
fall_transition_start
grounded_start
recovery_start
```

No pooled boundary-status κ or pooled overall status score is permitted.

---

## 11.1 Denominator

For every boundary type:

\[
N_{status}=12
\]

Each clip contributes exactly one paired status comparison.

---

## 11.2 Status confusion matrix

For each boundary, generate a fixed 6 × 6 matrix.

Rows:

```text
A01
```

Columns:

```text
A02
```

Fixed ordering:

```text
observed
left_censored
right_censored
not_observed
unjudgeable
not_applicable
```

Invariant for each boundary:

\[
\sum_{i=1}^{6}\sum_{j=1}^{6}n_{ij}=12
\]

---

## 11.3 Status exact agreement

For each boundary:

\[
P_{status}
=
\frac{\#(status_{A01}=status_{A02})}{12}
\]

Report:

```text
status_exact_agreement_count
status_exact_agreement_rate
```

All six categories require exact equality.

For example:

```text
left_censored ↔ left_censored
```

is status agreement.

But it is NOT temporal-location agreement.

---

## 11.4 Boundary availability decomposition

For every boundary classify each of the 12 pairs into exactly one of:

### A. `both_observed`

```text
A01 = observed
A02 = observed
```

### B. `same_nonobserved_status`

```text
A01 == A02
AND
A01 != observed
```

### C. `observed_vs_nonobserved`

Exactly one annotator has:

```text
observed
```

### D. `different_nonobserved_status`

Both are non-`observed`, but their statuses differ.

Required invariant:

\[
n_A+n_B+n_C+n_D=12
\]

These counts MUST be reported per boundary.

---

## 11.5 Chance-corrected status coefficients

Stage 3.2i primary analysis SHALL NOT calculate or interpret a six-category boundary-status κ as a headline reliability statistic.

Reason:

- only 12 observations exist per boundary;
- six possible categories create a highly sparse contingency table;
- the primary scientific information is the exact status disagreement pattern.

If a future extension calculates a chance-corrected status coefficient, that extension requires a separately frozen analysis amendment.

---

# 12. Temporal comparability rule

A boundary is temporally comparable if and only if:

\[
status_{A01}=observed
\]

AND

\[
status_{A02}=observed
\]

Therefore:

```text
temporal_comparable = true
```

only for:

```text
observed ↔ observed
```

All other status combinations are:

```text
temporal_comparable = false
```

This rule applies regardless of whether `event_presence` agrees.

`event_presence` agreement MUST be stored alongside the temporal result for interpretation, but MUST NOT silently change the temporal-comparability rule.

Any cross-field consistency requirement already present in the frozen annotation protocol remains valid. Stage 3.2i MUST NOT invent additional consistency rules after seeing the data.

---

# 13. Non-comparable boundary handling

For a non-comparable boundary:

```text
preferred_difference_frames = NA
preferred_difference_ms = NA
absolute_difference_frames = NA
absolute_difference_ms = NA

intersection_width = NA
union_width = NA
interval_iou = NA
overlap_coefficient = NA
inter_interval_blank_gap_frames = NA
```

Zero MUST NOT be used to represent missing/non-comparable temporal data.

A non-comparable reason MUST be stored as one of:

```text
A01_NONOBSERVED_ONLY
A02_NONOBSERVED_ONLY
BOTH_NONOBSERVED_SAME_STATUS
BOTH_NONOBSERVED_DIFFERENT_STATUS
```

The original A01 and A02 statuses MUST also remain available.

---

# 14. Analysis Layer C — preferred-frame agreement

Preferred-frame calculations apply only to:

```text
temporal_comparable == true
```

For one comparable boundary pair:

```text
P1 = A01 preferred_frame
P2 = A02 preferred_frame
```

---

## 14.1 Signed difference

Direction is frozen as:

\[
d=P2-P1
\]

That is:

\[
d=P_{A02}-P_{A01}
\]

Interpretation:

```text
d > 0 → A02 placed the boundary later
d < 0 → A02 placed the boundary earlier
d = 0 → exact preferred-frame agreement
```

The sign convention MUST NOT be reversed in individual reports or plots.

---

## 14.2 Absolute difference

\[
|d|=|P2-P1|
\]

Report:

```text
preferred_signed_difference_frames
preferred_absolute_difference_frames
preferred_signed_difference_ms
preferred_absolute_difference_ms
```

with:

\[
difference_{ms}=difference_{frames}\times50
\]

---

## 14.3 Exact preferred-frame match

Define:

```text
preferred_exact_match = (P1 == P2)
```

Per-boundary rate:

\[
P_{preferred-exact}
=
\frac{
\#preferred\_exact\_match
}{
n_{both-observed}
}
\]

The denominator is NOT 12.

If:

\[
n_{both-observed}=0
\]

report:

```text
NA
```

not zero.

---

## 14.4 Per-boundary descriptive summary

For each boundary, report:

```text
n_temporally_comparable

preferred_exact_match_count
preferred_exact_match_rate

median_signed_difference_frames
median_signed_difference_ms

median_absolute_difference_frames
median_absolute_difference_ms

Q1_absolute_difference_frames
Q3_absolute_difference_frames
IQR_absolute_difference_frames

min_signed_difference_frames
max_signed_difference_frames
max_absolute_difference_frames
```

If quartiles are calculated programmatically, they MUST use:

```text
Hyndman–Fan Type 7
```

equivalent to:

```text
numpy.quantile(..., method="linear")
```

This prevents implementation-dependent quartile definitions.

All underlying per-clip differences MUST also be retained in a derived table so the summary statistics are auditable.

---

# 15. Forbidden temporal metrics

Preferred-frame analysis MUST NOT use:

- Pearson correlation as an agreement metric;
- Spearman correlation as an agreement metric;
- \(R^2\);
- arbitrary tolerance accuracy such as `within ±3 frames`;
- frame rounding to a coarser grid;
- temporal binning;
- ML window overlap;
- label smoothing;
- pseudo-ground-truth averaging.

No statement such as:

```text
≤ 3 frames = acceptable
```

may be introduced during Stage 3.2i.

No scientifically justified downstream tolerance has yet been established.

---

# 16. Analysis Layer D — uncertainty interval agreement

Interval analysis applies only where:

```text
A01.status = observed
A02.status = observed
```

For A01:

\[
I_1=[E_1,L_1]
\]

with preferred point \(P_1\).

For A02:

\[
I_2=[E_2,L_2]
\]

with preferred point \(P_2\).

All frame intervals are discrete and inclusive.

---

## 16.1 Interval width

For annotator \(j\):

\[
W_j=L_j-E_j+1
\]

The `+1` is mandatory.

For example:

```text
[10, 10]
```

has width:

```text
1 frame
```

not zero.

Report:

```text
A01_interval_width_frames
A02_interval_width_frames
A01_interval_width_ms
A02_interval_width_ms
```

For interval duration reporting only:

\[
width_{ms}=width_{frames}\times50
\]

---

## 16.2 Intersection width

\[
I=
\max
\left(
0,
\min(L_1,L_2)
-
\max(E_1,E_2)
+1
\right)
\]

Define:

```text
interval_overlap_any = (I > 0)
```

---

## 16.3 Union width

\[
U=W_1+W_2-I
\]

For valid non-empty observed intervals:

\[
U>0
\]

---

## 16.4 Interval IoU

\[
IoU=\frac{I}{U}
\]

with:

\[
0\le IoU\le1
\]

Interpretation:

```text
1.0 → exact interval equality
0.0 → no shared frame
```

IoU MUST NOT receive an arbitrary good/bad threshold during Stage 3.2i.

---

## 16.5 Overlap coefficient

Define:

\[
OverlapCoefficient=
\frac{I}{\min(W_1,W_2)}
\]

with:

\[
0\le OverlapCoefficient\le1
\]

This statistic MUST accompany IoU because it identifies cases where the smaller uncertainty interval is fully contained within a larger interval.

It MUST NOT replace interval-width reporting.

---

# 17. Preferred-point containment

For every temporally comparable boundary compute:

```text
A01_preferred_in_A02_interval
A02_preferred_in_A01_interval
```

where:

\[
A01\_in\_A02
=
(E_2\le P_1\le L_2)
\]

and:

\[
A02\_in\_A01
=
(E_1\le P_2\le L_1)
\]

Also compute:

```text
mutual_preferred_containment
```

defined as:

```text
A01_preferred_in_A02_interval
AND
A02_preferred_in_A01_interval
```

The denominator for containment rates is:

\[
n_{both-observed}
\]

---

# 18. Inter-interval blank gap

For two discrete inclusive observed uncertainty intervals that do not overlap, `inter_interval_blank_gap_frames` is the number of source AVI frame indices strictly between the two intervals that are contained in neither interval.

When:

\[
I=0
\]

define:

```text
inter_interval_blank_gap_frames
=
max(E1, E2) - min(L1, L2) - 1
```

Examples:

```text
[10, 20] vs [25, 30]
inter_interval_blank_gap_frames = 4
```

because frames:

```text
21, 22, 23, 24
```

lie between them.

For:

```text
[10, 20] vs [21, 30]
```

the intervals do not overlap, but:

```text
inter_interval_blank_gap_frames = 0
```

because they are immediately adjacent.

When the intervals overlap:

```text
inter_interval_blank_gap_frames = NA
```

not zero.

Compatibility note: This Stage 3.2i derived metric is not the historical Stage 3.2e endpoint-distance metric. Stage 3.2e semantics remain frozen and unchanged. No cross-stage reinterpretation or migration is being performed.

---

# 19. Interval summary per boundary

For each of the three boundary types report:

```text
n_temporally_comparable

any_overlap_count
any_overlap_rate

interval_exact_match_count
interval_exact_match_rate

median_interval_iou
min_interval_iou
max_interval_iou

median_overlap_coefficient
min_overlap_coefficient
max_overlap_coefficient

A01_median_interval_width_frames
A02_median_interval_width_frames

mutual_preferred_containment_count
mutual_preferred_containment_rate

disjoint_interval_count

median_inter_interval_blank_gap_frames_among_disjoint
max_inter_interval_blank_gap_frames_among_disjoint
```

These `inter_interval_blank_gap_frames` summaries in `boundary_interval_summary.csv` use only disjoint interval pairs.

If:

```text
disjoint_interval_count = 0
```

`inter_interval_blank_gap_frames` summaries MUST be `NA`.

If:

```text
n_temporally_comparable = 0
```

all interval summary rates and continuous summaries MUST be `NA`.

---

# 20. Censored annotations

The following statuses represent categorical information:

```text
left_censored
right_censored
```

They MUST participate in boundary-status agreement analysis.

They MUST NOT participate in preferred-frame or uncertainty-interval analysis.

Examples:

```text
left_censored ↔ left_censored
```

means:

```text
status agreement = yes
temporal agreement = not measurable
```

Similarly:

```text
right_censored ↔ right_censored
```

does NOT establish that the two annotators agree about the actual boundary location.

The actual unobserved event position MUST NOT be imputed.

---

# 21. `not_observed`, `unjudgeable`, and `not_applicable`

These categories MUST remain semantically distinct.

For example:

```text
not_observed ↔ not_applicable
```

is a boundary-status disagreement.

Similarly:

```text
unjudgeable ↔ not_observed
```

is disagreement.

No attempt shall be made in Stage 3.2i to determine which interpretation is more appropriate.

That belongs to subsequent discrepancy review/adjudication.

---

# 22. Discrepancy inventory

Stage 3.2i MUST generate a derived discrepancy inventory.

This file is descriptive only.

It MUST NOT alter any source annotation.

Objective discrepancy flags may include:

```text
EVENT_PRESENCE_MISMATCH
BOUNDARY_STATUS_MISMATCH
PREFERRED_FRAME_MISMATCH
INTERVAL_DISJOINT
```

`PREFERRED_FRAME_MISMATCH` means strictly:

```text
both observed
AND
A01 preferred_frame != A02 preferred_frame
```

No threshold is applied.

Do NOT create subjective categories such as:

```text
minor disagreement
acceptable disagreement
major disagreement
bad annotation
annotator error
```

during Stage 3.2i.

Each discrepancy must retain sufficient provenance to recover:

```text
annotation_unit_id
boundary_type

A01 event_presence
A02 event_presence

A01 boundary status
A02 boundary status

A01 E/P/L when applicable
A02 E/P/L when applicable

derived difference metrics
derived overlap metrics
inter_interval_blank_gap_frames
```

---

# 23. Required overall reporting structure

The final report MUST contain separate sections for:

## 23.1 Provenance and integrity

Report:

```text
protocol commit
runtime/source commit
genuine run ID
raw-freeze manifest SHA
analysis code commit
record counts
pairing result
raw integrity result
```

---

## 23.2 Event presence

Report:

- 3 × 3 confusion matrix;
- exact agreement;
- A01 marginals;
- A02 marginals;
- Cohen's κ;
- Gwet's AC1;
- explicit note that n = 12.

---

## 23.3 `fall_transition_start`

Report independently:

- 6 × 6 status matrix;
- status exact agreement;
- availability decomposition;
- n both observed;
- preferred-frame differences;
- interval overlap;
- interval widths;
- preferred containment;
- disjoint `inter_interval_blank_gap_frames`.

---

## 23.4 `grounded_start`

Same structure.

---

## 23.5 `recovery_start`

Same structure.

---

## 23.6 Cross-boundary summary

A compact table MAY place the three boundary types in separate rows for visual comparison.

However, it MUST NOT calculate:

- pooled status agreement;
- pooled preferred-frame MAE;
- pooled IoU;
- pooled κ;
- macro-averaged agreement score;
- weighted composite agreement score.

The three boundary semantics remain separate.

---

# 24. Metrics that MUST NOT be collapsed into one scalar

Stage 3.2i MUST NOT construct any metric of the form:

\[
S=
w_1\kappa+
w_2IoU+
w_3FrameAgreement+
...
\]

The following concepts measure fundamentally different phenomena:

```text
event category agreement
boundary observability/status agreement
preferred temporal position agreement
annotator uncertainty width
interval overlap
censoring behavior
```

There is no scientifically justified common scale on which these may currently be added.

Therefore:

```text
overall_agreement_score
```

MUST NOT exist.

---

# 25. No artificial sample-size inflation

The report MUST NOT claim:

```text
N = 24
```

because there are 24 annotation records.

It MUST NOT claim:

```text
N = 36
```

because there are 12 clips × 3 boundaries.

It MUST NOT claim:

```text
N = 48
```

by combining event presence and boundaries.

The paired pilot contains:

\[
12\ clips
\]

with repeated annotation dimensions within each clip.

---

# 26. Statistical inference restrictions

Stage 3.2i is primarily descriptive.

No hypothesis testing is required.

Do NOT calculate or interpret:

```text
p-values for agreement > chance
```

as evidence that the protocol is valid.

No primary confidence interval for κ or AC1 is required in this stage.

With only 12 clips, chance-corrected coefficients may be highly unstable and MUST be interpreted together with the raw contingency table.

If inferential uncertainty for κ/AC1 is desired later, it requires an explicit analysis amendment rather than silently adding an estimator after inspecting the results.

---

# 27. Required derived outputs

Stage 3.2i MUST generate, at minimum:

```text
stage32i_analysis_manifest.json

stage32i_input_validation.json

paired_annotations.csv

event_presence_confusion_matrix.csv

event_presence_metrics.json

boundary_status_pairs.csv

boundary_status_confusion_fall_transition_start.csv

boundary_status_confusion_grounded_start.csv

boundary_status_confusion_recovery_start.csv

boundary_status_metrics.csv

boundary_temporal_comparable.csv

boundary_temporal_summary.csv

boundary_interval_summary.csv

discrepancy_inventory.csv

stage32i_report.md
```

File paths may follow existing repository conventions, but these logical artifacts and contents MUST exist.

---

# 28. `stage32i_analysis_manifest.json`

At minimum record:

```text
stage
analysis_spec_version
analysis_run_id
analysis_timestamp

execution_protocol_commit
source_annotation_runtime_commit
raw_freeze_manifest_sha256
analysis_code_commit

source_genuine_run_id

annotator_ids

expected_original_record_count
observed_original_record_count

expected_pair_count
observed_pair_count

fps
canonical_temporal_coordinate

analysis_input_paths
analysis_output_paths
```

The manifest MUST identify analysis outputs as derived artifacts rather than original annotation records.

---

# 29. `paired_annotations.csv`

Exactly:

```text
12 rows
```

One row per pilot clip.

Must provide sufficient raw-derived information to audit A01/A02 pairing.

It MUST NOT contain adjudicated fields.

---

# 30. `boundary_status_pairs.csv`

Exactly:

\[
12\times3=36
\]

rows.

One row per:

```text
annotation_unit_id × boundary_type
```

At minimum include:

```text
annotation_unit_id
boundary_type

A01_status
A02_status

status_exact_match
availability_class

temporal_comparable

event_presence_A01
event_presence_A02
event_presence_match
```

---

# 31. `boundary_temporal_comparable.csv`

Contains only rows satisfying:

```text
A01_status == observed
AND
A02_status == observed
```

Required columns include:

```text
annotation_unit_id
boundary_type

A01_earliest
A01_preferred
A01_latest

A02_earliest
A02_preferred
A02_latest

preferred_signed_difference_frames
preferred_absolute_difference_frames

preferred_signed_difference_ms
preferred_absolute_difference_ms

preferred_exact_match

A01_interval_width_frames
A02_interval_width_frames

intersection_width_frames
union_width_frames

interval_overlap_any
interval_iou
overlap_coefficient

A01_preferred_in_A02_interval
A02_preferred_in_A01_interval
mutual_preferred_containment

inter_interval_blank_gap_frames
```

---

# 32. Visual diagnostic outputs

Stage 3.2i SHOULD generate:

```text
event_presence_confusion_matrix.png

boundary_status_confusion_fall_transition_start.png
boundary_status_confusion_grounded_start.png
boundary_status_confusion_recovery_start.png
```

For any boundary with at least one temporally comparable pair:

```text
preferred_frame_difference_<boundary>.png
interval_agreement_<boundary>.png
```

Preferred-frame difference plots MUST preserve the sign convention:

\[
A02-A01
\]

The plots MUST show raw individual observations wherever practical.

No arbitrary green/yellow/red quality bands may be introduced.

No adjudicated interpretation may appear on the plots.

A Bland–Altman-style difference-vs-location diagnostic MAY be shown descriptively, but formal limits of agreement MUST NOT be treated as reliable population estimates from this very small pilot.

---

# 33. Denominator audit

Before declaring Stage 3.2i complete, the implementation MUST explicitly verify:

### Event presence

\[
sum(confusion\ matrix)=12
\]

### Each boundary status matrix

\[
sum(matrix)=12
\]

### Availability decomposition

For every boundary:

\[
both\_observed
+
same\_nonobserved
+
observed\_vs\_nonobserved
+
different\_nonobserved
=
12
\]

### Temporal rows

For every boundary:

\[
n_{temporal\ rows}=n_{both-observed}
\]

### Preferred exact-match denominator

\[
denominator=n_{both-observed}
\]

### Interval overlap denominator

\[
denominator=n_{both-observed}
\]

### Preferred-containment denominator

\[
denominator=n_{both-observed}
\]

No metric may silently use a denominator based on non-missing values unless that denominator is exactly the one specified above.

---

# 34. Raw immutability verification

Raw annotation integrity MUST be verified:

```text
before analysis
```

and again:

```text
after all analysis outputs have been generated
```

The post-analysis verification MUST confirm that all 24 original records remain unchanged relative to the raw freeze.

A clean Git diff alone is insufficient if the annotation records are ignored/untracked.

The frozen cryptographic identity is authoritative.

Any source mutation is an immediate Stage 3.2i Gate failure.

---

# 35. Stage 3.2i Gate semantics

Stage 3.2i uses a:

```text
PROCESS / METHODOLOGY GATE
```

not an:

```text
AGREEMENT PERFORMANCE GATE
```

Therefore high disagreement does NOT by itself cause Stage 3.2i to fail.

For example, the following is scientifically valid:

```text
Stage 3.2i completion gate: PASS

Finding:
substantial disagreement exists for recovery_start.

Interpretation:
requires discrepancy investigation before adjudication.
```

Agreement measurement succeeds even when it reveals poor agreement.

---

# 36. Hard Gate failure conditions

Stage 3.2i MUST STOP and report HOLD/FAIL if any of the following occurs:

- genuine run ID mismatch;
- raw-freeze manifest mismatch;
- raw-freeze verification failure;
- original record mutation;
- record count other than 24;
- annotator count other than 12 + 12;
- non-FINAL annotation record;
- missing A01/A02 pair;
- duplicate canonical annotation unit;
- ambiguous pairing;
- unknown `event_presence` category;
- unknown boundary status;
- invalid observed-boundary frame structure;
- violation of `E <= P <= L`;
- illegal frame index under available frozen metadata;
- access to prohibited Validation/Test subjects;
- use of a different annotation run;
- silent schema migration;
- silent repair of original annotations;
- denominator invariant failure;
- analysis outputs that cannot be traced to the frozen input records.

In such a case, do not continue with partial agreement calculations and do not attempt to “fix” the frozen records.

---

# 37. Conditions that are NOT Gate failures

The following findings are legitimate scientific results and MUST NOT automatically cause failure:

- low exact agreement;
- low Cohen's κ;
- disagreement between κ and AC1;
- frequent `uncertain`;
- different annotator uncertainty widths;
- poor interval IoU;
- large preferred-frame differences;
- frequent censoring;
- `observed` vs censored disagreements;
- `not_observed` vs `unjudgeable`;
- zero temporally comparable examples for one boundary;
- disjoint uncertainty intervals.

These findings must be reported, not corrected.

---

# 38. Forbidden actions

During Stage 3.2i, the implementation MUST NOT:

1. modify any of the 24 original records;
2. overwrite annotations;
3. create adjudicated labels;
4. choose a “winning” annotator;
5. ask annotators to revise annotations;
6. reopen clips for re-annotation;
7. silently alter the Stage 3.2e/3.2g annotation protocol;
8. collapse `uncertain`;
9. convert `uncertain` into numerical label smoothing;
10. convert censored boundaries to artificial frames;
11. average A01 and A02 preferred frames into ground truth;
12. define a final temporal label;
13. define final ML windows;
14. define fall/non-fall training supervision;
15. derive validation labels;
16. access Validation subjects;
17. access Test Subjects 6 or 7;
18. tune metrics after observing which metric looks more favorable;
19. delete disagreements as outliers;
20. exclude difficult clips from denominators;
21. pool the three boundary types into one temporal metric;
22. produce an overall agreement scalar;
23. classify disagreement using an arbitrary tolerance threshold;
24. interpret one annotator as more accurate without adjudication evidence.

---

# 39. Required limitations statement

`stage32i_report.md` MUST explicitly state:

> This analysis is based on an independent dual-annotation pilot containing 12 paired clips. Agreement coefficients are descriptive and are not treated as precise population-level reliability estimates. Event presence, boundary status, preferred temporal position, and uncertainty intervals represent different measurement dimensions and are therefore reported separately. No adjudication or downstream ML supervision decisions were performed during Stage 3.2i.

---

# 40. Required Stage 3.2i report conclusion format

The report conclusion SHOULD follow this structure:

```text
Stage 3.2i Analysis Execution:
PASS / HOLD

Input integrity:
PASS / FAIL

Raw immutability after analysis:
PASS / FAIL

Paired clips:
12 / 12

Event-presence agreement:
[descriptive results only]

fall_transition_start:
[status + temporal descriptive results]

grounded_start:
[status + temporal descriptive results]

recovery_start:
[status + temporal descriptive results]

Primary discrepancy patterns:
[objective observations only]

Adjudication:
NOT PERFORMED

Original-record modification:
NONE

Validation-subject access:
NONE

Test Subject 6/7 access:
NONE

Downstream supervision design:
NOT PERFORMED
```

No single numeric value may be presented as the overall reliability of Stage 3.2i.

---

# 41. Stage 3.2i Completion Gate Checklist

Before execution:

```text
[ ] Analysis specification frozen before viewing agreement results
[ ] Genuine Stage 3.2h run identity verified
[ ] Raw-freeze SHA-256 verified
[ ] 24/24 frozen records verified
[ ] A01 = 12/12 FINALIZED
[ ] A02 = 12/12 FINALIZED
[ ] Exactly 12 canonical pairs
[ ] Vocabulary validation PASS
[ ] Observed-boundary structural validation PASS
[ ] No prohibited held-out subjects accessed
```

After execution:

```text
[ ] Event-presence 3×3 matrix totals 12
[ ] Event exact-agreement denominator = 12
[ ] Cohen's κ uses fixed three-category nominal definition
[ ] Gwet's AC1 uses fixed K = 3
[ ] `uncertain` remained an independent nominal category

[ ] Three boundary types analyzed separately
[ ] Every boundary 6×6 status matrix totals 12
[ ] Availability decomposition totals 12 per boundary
[ ] Only observed↔observed pairs entered temporal analysis
[ ] Censored/non-observed states were never converted to pseudo-frames

[ ] Preferred-frame sign is consistently A02 − A01
[ ] Frame differences reported before ms conversion
[ ] 20 FPS / 50 ms per frame used
[ ] Inclusive interval widths use L − E + 1
[ ] IoU formula validated
[ ] Overlap coefficient validated
[ ] Preferred containment validated
[ ] `inter_interval_blank_gap_frames` validated

[ ] No arbitrary temporal tolerance introduced
[ ] No pooled boundary score produced
[ ] No overall agreement scalar produced
[ ] No adjudication performed
[ ] No label smoothing performed
[ ] No final ML supervision designed

[ ] All required derived artifacts generated
[ ] Denominator audit PASS
[ ] Analysis provenance recorded
[ ] All 24 raw records reverified after analysis
[ ] Original records remain byte/cryptographically unchanged
[ ] Validation/Test sealing preserved
```

---

# 42. Completion rule

Stage 3.2i may be marked:

```text
COMPLETE / PASS
```

if and only if:

1. all pre-analysis integrity checks pass;
2. all specified agreement analyses are executed without changing their definitions;
3. all denominator invariants pass;
4. all required outputs are generated;
5. the frozen annotations remain unchanged;
6. no prohibited data are accessed;
7. no adjudication is performed;
8. no downstream supervision design is introduced.

The magnitude of inter-annotator agreement is NOT part of the completion criterion.

The scientific output of Stage 3.2i is:

> a reproducible characterization of where A01 and A02 agree, where they disagree, how temporal differences are distributed, and how their explicitly annotated uncertainty intervals relate.

Only after these results have been frozen may a subsequent stage begin discrepancy interpretation and/or adjudication.