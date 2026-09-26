# Stage 3.2b — annotation-native timeline feasibility

This audit is pending external Gate Review. It establishes ordering and timing
provenance facts only. The frozen parent is
`98c34e553b3a75dddef6f28816db2227a6188aa7`; no frozen file is changed.
The official Stage 2 run remains `4df7dd5f-fb38-4908-8233-1a81deb8dc05`.

## Reproduction and scope

Run from the repository root. The output directory must not already exist.

```sh
UV_CACHE_DIR=/private/tmp/caucafall-stage26-cache uv run --offline --python 3.13.15 --no-project --with numpy==2.2.6 --with opencv-contrib-python==4.12.0.88 python -B -m unittest tests.test_stage32b_annotation_timeline tests.test_stage3_contract -v
UV_CACHE_DIR=/private/tmp/caucafall-stage26-cache uv run --offline --python 3.13.15 --no-project --with numpy==2.2.6 --with opencv-contrib-python==4.12.0.88 python -B -m ml.preprocessing.audit_annotation_timeline
git diff --check
```

The source selector is the frozen metadata-only Train selector. Subject IDs
must be exactly 8, 4, 3, 9, 1, 2. Validation and Test requests are rejected
before source I/O, including spoofed Train membership. Raw PNG/TXT checksums
must match the Stage 3.2a source ledger; that stage's artifact checksums are
verified before analysis. No AVI is opened, and no pose array is loaded.
AVI counts/FPS are references from frozen evidence, not new alignment results.

The read-only workbook parser selects the six authorized subject blocks in
`data/raw/caucafall_v5/Dataset details.xlsx`, sheet `Hoja1`, before resolving
activity/count cells. It records cell references and rejects an unexpected
layout or formula. ZIP/XML container parsing is necessary to select cells;
Validation/Test subject values are not resolved or analyzed. The local search
covers the dataset download root, CAUCAFall root, and selected Train subject and
activity directories. Supplemental text documents are searched for frame/FPS/
timestamp/extraction/transcoding terms. General JPEG figures are inventoried
and hashed, not interpreted as timebase evidence. No held-out directories are
opened to search for additional provenance.

## Identity and controlled vocabulary

- `png_filename_index` is the complete numeric token interpreted as an integer;
  the original token string, prefix, suffix and filename are also retained.
- `png_ordinal_index` is zero-based position after sorting by numeric index,
  then original filename. A filename gap never creates an imputed image.
- `avi_frame_index` remains null. It is not a synonym for either identifier.
- `annotation_order_status=verified` means a unique numeric sequence and
  bijective PNG/TXT identity. Gaps may exist; complete acquisition coverage,
  uniform spacing and physical chronology are not thereby certified.
- `annotation_order_status=unresolved` means unparsed/duplicate ordering keys,
  incompatible prefixes, or incomplete/non-bijective annotation identity.
- `annotation_timebase_status=unresolved` means no verified PNG sampling
  interval or per-image capture clock. This audit cannot promote it based on
  counts, ordering, an AVI FPS, or a camera specification.

Exact stems require unique prefix/numeric-token identities in both file sets.
Only the four explicitly declared aliases can bridge a stem difference:

| Sequence | Original PNG | Original TXT |
| --- | --- | --- |
| Subject.2 / Fall backwards | cas200091 - copia.png | cas200091.txt |
| Subject.8 / Hop | sals800096.png | sals800096a.txt |
| Subject.8 / Pick up object | res800090.png | res800090a.txt |
| Subject.9 / Walk | cams900140.png | cams900140w.txt |

Each alias must have the identical prefix and unchanged digit spelling,
exactly one occurrence of that identity in each sequence, and no competing
exact stem. Otherwise it remains unresolved. Accepted aliases receive
`verified_explicit_sequence_alias`; names and neighbors remain in the evidence.
This supports deterministic sequence identity, not independent certification
of the dataset author's intent. Source filenames are never renamed.

Class 0 remains `nofall`; class 1 remains `grounded_fall_state`, consistent
with the frozen annotation audit. First/last class-1 positions and transitions
are observed PNG ordinals only, not active-fall onset or physical-time labels.

## Exact structure and temporal provenance

All selected PNGs are decoded with OpenCV `IMREAD_UNCHANGED`; bit depth,
channels and size are preserved. File SHA-256 and decoded shape/dtype/pixel
SHA-256 are separate. Repeated decoded hashes are checked with `array_equal`.
There is no resizing, near-duplicate threshold, AVI comparison, DTW, or flow.
Every duplicate group is retained. Reports separate first/second and
penultimate/last pairs from internal adjacent and nonadjacent repetition.
The four boundary records are explicitly exported for every sequence.

The hypothetical count after removing identical copies is an arithmetic
diagnostic only. Matching an AVI count after such subtraction would not prove
which samples are redundant in time or justify deleting them. The systematic
+1 and Subject.2 Fall backwards +2 remain separate causal questions.

PNG chunk types and textual/time/EXIF ancillary records are recorded. A PNG
modification-time field is not assumed to describe capture time. No physical
PNG timestamp is generated; null timestamps serialize as empty CSV fields.

The [official V5 description](https://data.mendeley.com/datasets/7w7fccy7ky/5)
describes a camera capable of 23 FPS and motion-triggered DVR recording. This
is external documentation, not a local measurement of a PNG clock. The frozen
local AVI evidence records 20 FPS. Acquisition capability, AVI container
timing and PNG extraction cadence are separate provenance layers; their
relationship is not assumed. The workbook's `Frames` field is a documented
count, not documentation of how images were generated or timed.

The provenance table uses four categories:
`FACT_SUPPORTED_BY_LOCAL_EVIDENCE`,
`FACT_SUPPORTED_ONLY_BY_EXTERNAL_DATASET_DOCUMENTATION`,
`DERIVED_OBSERVATION`, and `UNRESOLVED`.

Let `T` be the frozen decoded AVI frame count, `f` its recorded FPS, and `N`
the PNG count. Two AVI reference conventions are kept separate:

| Quantity | Formula | Meaning |
| --- | --- | --- |
| AVI count duration | T/f | Count divided by nominal rate |
| AVI first-to-last sample span | (T−1)/f | Span under uniform AVI sampling |
| PNG sample/count-duration ratio | N/(T/f) | Diagnostic ratio |
| PNG sample/span ratio | N/((T−1)/f) | Diagnostic ratio |
| PNG interval/count-duration ratio | (N−1)/(T/f) | Diagnostic ratio |
| PNG interval/span ratio | (N−1)/((T−1)/f) | Diagnostic ratio |

Equal temporal coverage/endpoints and uniform PNG spacing are unverified.
None of these ratios is accepted as true PNG FPS. No target FPS, resampling,
interpolation, window, stride, pose threshold, label-overlap rule, active-fall
definition, feature set or model schema is selected.

## Artifacts and boundaries

The new namespace is
`artifacts/preprocessing/stage32b_train_annotation_native_timeline_v1/`.
It contains sequence inventory, per-PNG ordinal/annotation identity, exact
duplicate and boundary audits, local workbook evidence, local provenance
search, provenance classification, timing diagnostics, anomaly cases,
summary and report. A validation record may be added after the official run.
These are source-audit artifacts, not pose data or model-ready supervision.

Synthetic tests cover ordering, distinct indices, gaps, duplicate identities,
explicit aliases and competing candidates, exact byte/pixel duplication,
boundary/internal repetition, single-pixel differences, ordinal conventions,
unresolved timebase/null timestamps, ordinal class transitions, workbook
selection, and held-out rejection. A synthetic sequence integration test
forbids AVI and pose loaders. Stage 3 contract regression tests also run.

No PNG pose extraction, labeled pose arrays, temporal windows, preprocessing
or model training is authorized by this stage. No commit is made. Stop for
external Gate Review.
