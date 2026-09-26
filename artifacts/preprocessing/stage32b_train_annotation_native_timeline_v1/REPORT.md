# Stage 3.2b annotation-native timeline feasibility

Pending external Gate Review; no PASS claim.

## Findings

- sequences: 60
- png_files: 11176
- txt_files: 11176
- exact_stem_pairs: 11172
- explicit_sequence_alias_pairs: 4
- order_status_counts: {'verified': 60}
- timebase_status_counts: {'unresolved': 60}
- sequences_with_pixel_duplicates: 14
- pixel_duplicate_groups: 18
- pixel_duplicate_excess: 66
- file_duplicate_groups: 18
- class0_count: 7927
- class1_count: 3249
- avi_frames_reference: 11115
- png_minus_avi_counts: {1: 59, 2: 1}
- leading_boundary_duplicate_sequences: 1
- trailing_boundary_duplicate_sequences: 2
- workbook_train_frames: 11177
- workbook_count_mismatches: 1
- png_ancillary_record_count: 0

Verified order means deterministic annotation-native ordinal identity only. It does not establish physical spacing, a PNG FPS, or correspondence to the Stage 2 AVI timeline. Every PNG timestamp and AVI frame-index field remains empty.

The four explicit filename aliases require unique identical prefix/numeric-token identity and no competing candidate. No source name is normalized or changed. See anomaly_cases.csv for names and neighbors.

## Exact duplicate groups

| Sequence | Kind | PNG ordinals | Leading pair | Trailing pair |
| --- | --- | --- | --- | --- |
| Subject.1/Fall forward/FallForwardS1.avi | file_sha256 | [12, 13] | False | False |
| Subject.1/Fall forward/FallForwardS1.avi | decoded_pixel_sha256 | [12, 13] | False | False |
| Subject.1/Fall left/FallLeftS1.avi | file_sha256 | [5, 6] | False | False |
| Subject.1/Fall left/FallLeftS1.avi | file_sha256 | [71, 72] | False | False |
| Subject.1/Fall left/FallLeftS1.avi | decoded_pixel_sha256 | [5, 6] | False | False |
| Subject.1/Fall left/FallLeftS1.avi | decoded_pixel_sha256 | [71, 72] | False | False |
| Subject.1/Fall sitting/FallSittingS1.avi | file_sha256 | [89, 90] | False | False |
| Subject.1/Fall sitting/FallSittingS1.avi | file_sha256 | [116, 117, 118] | False | False |
| Subject.1/Fall sitting/FallSittingS1.avi | file_sha256 | [119, 120, 121, 122] | False | False |
| Subject.1/Fall sitting/FallSittingS1.avi | decoded_pixel_sha256 | [89, 90] | False | False |
| Subject.1/Fall sitting/FallSittingS1.avi | decoded_pixel_sha256 | [116, 117, 118] | False | False |
| Subject.1/Fall sitting/FallSittingS1.avi | decoded_pixel_sha256 | [119, 120, 121, 122] | False | False |
| Subject.1/Kneel/KneelS1.avi | file_sha256 | [8, 9] | False | False |
| Subject.1/Kneel/KneelS1.avi | decoded_pixel_sha256 | [8, 9] | False | False |
| Subject.2/Fall backwards/FallBackwardsS2.avi | file_sha256 | [90, 91] | False | False |
| Subject.2/Fall backwards/FallBackwardsS2.avi | decoded_pixel_sha256 | [90, 91] | False | False |
| Subject.2/Hop/HopS2.avi | file_sha256 | [106, 107] | False | False |
| Subject.2/Hop/HopS2.avi | decoded_pixel_sha256 | [106, 107] | False | False |
| Subject.2/Pick up object/PickupobjectS2.avi | file_sha256 | [142, 143] | False | False |
| Subject.2/Pick up object/PickupobjectS2.avi | decoded_pixel_sha256 | [142, 143] | False | False |
| Subject.2/Walk/WalkS2.avi | file_sha256 | [191, 192] | False | False |
| Subject.2/Walk/WalkS2.avi | decoded_pixel_sha256 | [191, 192] | False | False |
| Subject.4/Fall forward/FallForwardS4.avi | file_sha256 | [0, 1] | True | False |
| Subject.4/Fall forward/FallForwardS4.avi | decoded_pixel_sha256 | [0, 1] | True | False |
| Subject.4/Pick up object/PickupobjectS4.avi | file_sha256 | [169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179] | False | True |
| Subject.4/Pick up object/PickupobjectS4.avi | decoded_pixel_sha256 | [169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179] | False | True |
| Subject.9/Fall backwards/FallBackwardsS9.avi | file_sha256 | [218, 219] | False | False |
| Subject.9/Fall backwards/FallBackwardsS9.avi | decoded_pixel_sha256 | [218, 219] | False | False |
| Subject.9/Fall sitting/FallSittingS9.avi | file_sha256 | [179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216] | False | True |
| Subject.9/Fall sitting/FallSittingS9.avi | decoded_pixel_sha256 | [179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216] | False | True |
| Subject.9/Hop/HopS9.avi | file_sha256 | [13, 14] | False | False |
| Subject.9/Hop/HopS9.avi | file_sha256 | [59, 60] | False | False |
| Subject.9/Hop/HopS9.avi | decoded_pixel_sha256 | [13, 14] | False | False |
| Subject.9/Hop/HopS9.avi | decoded_pixel_sha256 | [59, 60] | False | False |
| Subject.9/Kneel/KneelS9.avi | file_sha256 | [71, 72] | False | False |
| Subject.9/Kneel/KneelS9.avi | decoded_pixel_sha256 | [71, 72] | False | False |

Exact repetition is retained. Uneven duplication across sequences does not objectively establish a common +1 cause. Subject.2 / Fall backwards has its own +2 anomaly; duplicate counts are arithmetic diagnostics, not a repaired mapping.

## Local workbook discrepancies

- Subject.1/Sit down/SitDownS1.avi: workbook 189 at Hoja1!D14; PNGs 188.

## Temporal provenance

The [official V5 documentation](https://data.mendeley.com/datasets/7w7fccy7ky/5) describes a camera capable of 23 FPS. Frozen local AVI metadata says 20 FPS. Neither value establishes PNG sampling. The inspected local workbook contains frame counts without a documented count-generation method or timebase. See timeline_provenance.csv for local facts, external claims, derived observations and unresolved facts.

Timebase diagnostics separately use AVI T/f and (T-1)/f denominators, and PNG N or N-1 numerators. These ratios are not PNG FPS estimates accepted for preprocessing. All PNG timebases remain unresolved.

Grounded-fall transitions are ordinal positions only. No AVI-pose-label join, PNG pose extraction, MediaPipe call, resampling, window, or model training occurred. Validation/Test subject contents were not accessed. All downstream preprocessing decisions remain UNDECIDED.
