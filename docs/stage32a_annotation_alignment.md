# Stage 3.2a — Train annotation source and frame alignment audit

This stage produces correspondence evidence for external Gate Review. It does
not authorize preprocessing, temporal supervision, window construction or a
pose-label dataset. The parent is Stage 3.1 commit
`46f4e299e943db24f386787e30ab05f484949262`; the official Stage 2 run remains
`4df7dd5f-fb38-4908-8233-1a81deb8dc05`.

## Reproduction

Run from the repository root. The output must not already exist.

```sh
UV_CACHE_DIR=/private/tmp/caucafall-stage26-cache uv run --offline --python 3.13.15 --no-project --with numpy==2.2.6 --with opencv-contrib-python==4.12.0.88 python -B -m unittest tests.test_stage32a_annotation_alignment tests.test_stage3_contract -v
UV_CACHE_DIR=/private/tmp/caucafall-stage26-cache uv run --offline --python 3.13.15 --no-project --with numpy==2.2.6 --with opencv-contrib-python==4.12.0.88 python -B -m ml.preprocessing.audit_annotation_alignment
UV_CACHE_DIR=/private/tmp/caucafall-stage26-cache uv run --offline --python 3.13.15 --no-project --with numpy==2.2.6 --with opencv-contrib-python==4.12.0.88 python -B -m ml.preprocessing.review_alignment_evidence
git diff --check
```

`--split` accepts only `train`, rejecting Validation and Test before source I/O.
The frozen `contract.select_sources(("train",), purpose="exploration")` supplies
membership and identity. Each selected source is checked again for canonical
paths and actual subject membership before opening its directory. No pose array
loader is imported or called. The selector's metadata/existence checks do not
load arrays or read Validation/Test image, annotation or frame contents.

The selected AVI hashes must match Stage 2's source checksum manifest. OpenCV
4.12.0 sequentially opens and reads the AVI to read termination, without seeking,
sampling or inference. FPS, dimensions and metadata count must agree with the
frozen inventory. A decoded count discrepancy disables all accepted mappings
for that video. OpenCV cannot distinguish EOF from decoder failure merely from
`read()` returning false; count agreement is reported explicitly.

## Annotation meaning and identity

The [dataset authors](https://pmc.ncbi.nlm.nih.gov/articles/PMC9508401/) describe
class 1 as a human body on the ground following a fall. We name that annotation
`grounded_fall_state`; class 0 is `nofall`. This is not an onset, transition or
window label. The locally supplied download's V5 identity is not independently
authenticated by this audit.

Each TXT is checked for exactly one nonempty row, five numeric fields, literal
class 0/1, finite values, normalized center x/y in [0,1], positive width/height
in (0,1], and agreement with the local `classes.txt`. Box edges outside the unit
square are separately reported. Box values are never used to align images,
correct poses, filter samples or choose supervision. Invalid files are retained.

Three identifiers are separate:

1. **Filename index:** the entire final numeric group in the filename stem. A
   subject digit may be part of it; no assumed subject prefix is removed.
2. **PNG ordinal:** zero-based position after numeric-token/name ordering.
3. **Stage 2 frame_index:** zero-based sequential AVI decode ordinal, with
   `timestamp_ms = round(frame_index * 1000 / source_fps)`.

The per-video table reports the actual index-minus-ordinal offsets, gaps,
duplicate numeric indices, prefixes and filename suffix variants. The script
does not equate displayed numbering with Stage 2 frame indices. A unique exact
PNG/TXT basename is required to accept annotation identity. A shared numeric
token with a different suffix is recorded as a candidate only; the raw files
are not renamed and the association is not silently accepted.

TXT sequence summaries use only original ordered annotations. Their transitions
refer to adjacent observed TXT rows, not inferred AVI frames. Invalid rows or
ambiguous numeric ordering suppress transition statistics. Filename gaps are
reported separately, with no missing annotation imputation. The class-1 interval
field refers to contiguity in observed annotation order.

## Image evidence and conservative mapping

Both AVI and PNG are compared as unresized uint8 BGR arrays. PNG loading uses
OpenCV `IMREAD_COLOR`; AVI BGR-to-RGB in Stage 2 is just a channel permutation,
so using BGR for both does not change pixel equality. SHA-256 includes shape,
dtype and C-order bytes; accepted pairs also undergo `array_equal`.

Hypothesis A tests every AVI frame against the PNG of the same ordinal when
present. Excess AVI frames and PNGs remain explicit. Hypothesis B is represented
by the filename-index-minus-ordinal offset alongside image evidence; a numbering
offset alone is not evidence of content correspondence.

Acceptance rules:

- An equal-length sequence with exact equality at every ordinal supports the
  full ordinal correspondence, including repeated frames within that sequence.
- Otherwise, retain only pixel identities that occur exactly once in each
  sequence and whose matched positions are strictly monotonic. Duplicated or
  crossed identities do not force an assignment.
- No non-exact pair is accepted automatically. For unresolved AVI frames,
  compare ordinal neighbors within ±3 positions and offsets supported by exact
  anchors within three AVI positions. Unmapped PNGs also receive bounded nearby
  AVI/boundary comparisons, including excess leading/trailing elements. This
  radius bounds investigation; it is not a match threshold. A search miss does
  not prove an image is absent from every other position.

Every comparison records dimensions, exact equality, mean absolute error,
maximum absolute error, MSE and PSNR. MAE/MSE average all original-resolution
pixels and BGR channels; subtraction is unsigned-safe, and squaring uses float64.
PSNR is `10 log10(255²/MSE)`; exact comparisons have infinite PSNR represented as
null with `exact=true`. Shape-incompatible pairs also have null metrics, with
`same_dimensions=false`. Descriptive distributions use NumPy linear quantiles.
No similarity threshold or preprocessing policy is selected.

## Artifacts

All evidence is published to a fresh directory:
`artifacts/preprocessing/stage32a_train_annotation_alignment_v1/`.

| Artifact | Evidence |
| --- | --- |
| `per_video_annotation_inventory.csv` | Counts, numbering, orphans, decoding properties, exact mapping counts, duplicate pixels |
| `annotation_schema.csv` | Every original TXT identity, class and schema result |
| `annotation_sequence_summary.csv` | Original TXT class counts, intervals and transitions |
| `frame_correspondence.csv` | Every AVI frame, every unmapped PNG and every orphan TXT; nullable identities for unresolved associations |
| `direct_ordinal_diagnostics.csv` | Hypothesis A at every AVI position, including explicit missing PNG positions |
| `local_alignment_diagnostics.csv` | Bounded candidate comparisons, without automatic non-exact acceptance |
| `mismatch_cases.csv` | Per-video count/identity/numbering/duplicate/unresolved cases |
| `source_files.csv` | Train-only source paths, roles and byte SHA-256 identities |
| `summary.json` | Totals, exact definitions, provenance and unresolved videos |
| `REPORT.md` | Human-readable evidence and limitations |

The artifact-only review command reads the saved diagnostics without another
media decode. It adds `candidate_rankings.csv`, `offset_similarity_summary.csv`,
`review_summary.json` and `REVIEW_NOTES.md`, refusing overwrite. Local MAE/MSE
minima and ties are descriptive candidates only; they do not update accepted
correspondence records. Offset summaries report compared and uncompared counts
explicitly. The review notes are tied to the observed v1 counts, with input hashes.

`frame_correspondence.csv` is an audit table. It does not contain skeletons or
pose features. For `avi_frame` rows, `exact` means both exact image correspondence
and valid exact-basename annotation identity; `image_correspondence_status`
distinguishes a proven image mapping with a missing or invalid annotation.
`unresolved`, `missing_annotation`, `missing_png` and `invalid_annotation` never
silently acquire a class for an AVI frame. Orphan annotation content remains in
the schema table; unmapped image/TXT rows have no Stage 2 index or timestamp.

No existing frozen file is changed. External review determines whether any
candidate rule is sufficient for a separately authorized later stage.
