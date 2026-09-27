# Stage 3.2f — annotation tool and frozen Train pilot

Status: **pending external Gate Review**. Parent commit:
`59052bab724fa501bc2938c822388356399046ea`. This is a protocol/tool
validation pilot, **not** a statistically representative sample. No human
annotation or model/window label was created.

## Freeze and selection

The pilot was frozen from the SHA-256-pinned official Stage 2 manifest before
any selected AVI was decoded. Only its 60 Train metadata rows were eligible.
The deterministic rule sorts the six Train subjects numerically
`1, 2, 3, 4, 8, 9`. Separately for fall and non-fall, it assigns the five
activities in canonical alphabetical order to the first five subjects and
repeats the first activity for the sixth. There is one video per
subject/activity. No filesystem enumeration, pixels, pose availability,
missingness, model output, PNG/TXT labels or annotation difficulty participates
in selection. The frozen [selection provenance](pilot_selection_provenance.json)
records the exact rule and metadata digest.

| Subject | Fall AVI | Non-fall AVI |
| --- | --- | --- |
| 1 | `Subject.1/Fall backwards/FallBackwardsS1.avi` | `Subject.1/Hop/HopS1.avi` |
| 2 | `Subject.2/Fall forward/FallForwardS2.avi` | `Subject.2/Kneel/KneelS2.avi` |
| 3 | `Subject.3/Fall left/FallLeftS3.avi` | `Subject.3/Pick up object/PickupobjectS3.avi` |
| 4 | `Subject.4/Fall right/FallRightS4.avi` | `Subject.4/Sit down/SitDownS4.avi` |
| 8 | `Subject.8/Fall sitting/FallSittingS8.avi` | `Subject.8/Walk/WalkS8.avi` |
| 9 | `Subject.9/Fall backwards/FallBackwardsS9.avi` | `Subject.9/Hop/HopS9.avi` |

There are six fall and six non-fall videos. All five activities in each class
appear at least once; `Kneel`, `Pick up object` and `Sit down` cover intentional
posture/descent hard negatives. The sample was not replaced after decoding.

Neutral IDs use the first 16 hexadecimal digits of a domain-separated SHA-256
digest of the frozen run, parent commit and source identity, prefixed `clip-`.
The presentation order is a second domain-separated digest order with a
deterministic swap only if it would otherwise alternate classes perfectly.
The [order](presentation_order.json) and [freeze record](freeze_record.json)
are hashed before decode. These IDs are procedurally blinded, not
cryptographically anonymous. The [private manifest](pilot_private_manifest.json)
stores source path/hash, subject, split, activity, dataset class, frame count,
FPS provenance, selection rule, parent and protocol version. The separate
[annotator manifest](pilot_annotator_manifest.json) has only neutral clip ID,
frame count and derived duration. It is the only manifest projected into the
UI. Reviewers may inspect private provenance; annotators must not.

## Frame source and local tool

The minimal local UI is a standard-library `HTTPServer` bound to
`127.0.0.1`, viewed in a browser. Tk was considered but its window creation
failed in the configured Python 3.13 runtime; the loopback UI needs no new
UI dependency. A random session token protects state-changing requests.
The HTTP response model is a strict projection of the neutral public clip
fields and current annotation controls. Source path, subject, activity,
dataset class, pose, PNG/TXT labels and model data never cross that boundary.
The synthetic loopback test checks both HTML and JSON responses.

On open, the source AVI hash is checked. `IndexedFrameSource.open_avi()` then
sequentially decodes from frame zero into memory, assigns each successful
read its zero-based ordinal, requires exactly the frozen frame count, and
rejects invalid dimensions or extra frames. The UI navigates this indexed
cache; it never seeks with `CAP_PROP_POS_FRAMES`. The current `frame_index`
is authoritative; displayed milliseconds are `round(frame_index*1000/20)`.
Transient PNG encoding is only for browser display and does not re-encode
or modify the source AVI.

The annotation display route is now
`GET /frame?clip=<neutral_clip_id>&index=<zero_based_frame_index>`. It requires
both fields, rejects an unknown neutral ID or out-of-range index, and uses a
non-mutating `IndexedFrameSource.peek(index)` to retrieve exactly the
requested cached decoded frame. The response is `image/png`, produced by
OpenCV `imencode(".png", frame)`, and carries `X-Frame-Index` and
`X-Neutral-Clip-Id` headers. PNG transport is **lossless relative to that
decoded OpenCV BGR frame**: OpenCV applies its ordinary reversible BGR/RGB
format convention when encoding; `imdecode(..., IMREAD_UNCHANGED)` returns
the exact cached BGR pixel array. This does not claim that the source AVI
codec itself is lossless.

`/state` and `/frame` responses use `Cache-Control: no-store`, and the browser
requests both with `cache: 'no-store'`. The UI requests the image using the
same state snapshot's neutral ID and `frame_index` used for its label. It
checks the image response media type and both identity headers, awaits image
decode, and then replaces the image and label in one JavaScript task. A
monotonic render epoch discards older asynchronous responses; actions are
serialized so rapid navigation cannot let an earlier response replace a
newer frame. Until the matching image is ready, the previous image and label
remain together. No source path, activity, subject, class, pose, PNG label or
model field appears in the browser frame route or its headers.

The synthetic HTTP test uses nine different deterministic pixel patterns.
For indices **0, 1, 4, 7 and 8**, it requests the actual `/frame` route,
checks `image/png` and no-store headers, decodes the returned PNG, and
requires exact array equality with the cached synthetic frame. An injected
off-by-one frame response is detected. It also checks next, previous, jump,
first/last clamping, 20 rapid step/state checks, and a state change between
the state and frame requests: the older explicit frame request still returns
its requested pixels without changing the navigation cursor. All checks
use synthetic frames only; none displays a real pilot clip.

The UI supports previous/next, ten-frame steps, exact frame jump,
event-presence and boundary-status selection, marking earliest/preferred/latest
at the current frame, controlled reason flags, note entry, draft save/resume,
and Stage 3.2e validation on finalization. Drafts can be replaced; a final
record uses exclusive-create storage and cannot be overwritten by the tool.
Corrections would require a new record or a later adjudication workflow.
No adjudication was implemented. Stage 3.2e owns schema validation; the UI
does not derive model labels. Tool interaction tests use synthetic frames and
temporary synthetic records only. The tool was **not** opened on a real pilot
clip or shown to a human annotator in this stage.

After a separate authorization to begin annotation, an operator can launch
`python -B -m ml.preprocessing.annotation_tool` with a frozen neutral clip ID,
pseudonymous annotator/session/run IDs, and an explicit private `--draft-root`
under Git-ignored `data/`. No such launch or real draft/final record occurred
in Stage 3.2f.

## Automated real-AVI integrity evidence

Only after the manifest freeze, the automated checker hashed and decoded the
12 selected Train AVIs sequentially. All 12 opened, matched their frozen
source hashes and measured 20 FPS/dimensions, and yielded exactly their
expected counts. The total is **2,163 expected and 2,163 decoded frames**;
each clip starts at index 0 and ends at `T-1` with contiguous indices.
There were **zero count mismatches**. The per-clip
[decode results](decode_integrity.json), [runtime provenance](tool_runtime_provenance.json)
and [validation record](validation_record.json) are machine-readable.
OpenCV `read()` termination cannot by itself distinguish a normal EOF from a
decoder failure when the count happens to match; count/hash/schema agreement
does not certify the visual integrity of every pixel.

## Verification

```sh
UV_CACHE_DIR=/private/tmp/caucafall-stage26-cache uv run --offline --python 3.13.15 --no-project python -B -m ml.preprocessing.annotation_pilot freeze
UV_CACHE_DIR=/private/tmp/caucafall-stage26-cache uv run --offline --python 3.13.15 --no-project --with opencv-contrib-python==4.12.0.88 python -B -m ml.preprocessing.annotation_pilot verify
UV_CACHE_DIR=/private/tmp/caucafall-stage26-cache uv run --offline --python 3.13.15 --no-project python -B -m ml.preprocessing.annotation_pilot validate
UV_CACHE_DIR=/private/tmp/caucafall-stage26-cache uv run --offline --python 3.13.15 --no-project --with opencv-contrib-python==4.12.0.88 python -B -m unittest tests.test_stage32f_annotation_tool tests.test_stage32e_manual_annotation_protocol tests.test_stage32d_temporal_supervision_strategy tests.test_stage3_contract -v
git diff --check
```

The full unittest run contained **84 tests, OK**. The loopback socket test
required an unsandboxed local bind because the execution sandbox denies even
`127.0.0.1` binding. No network service for real pilot clips was launched.
No Validation/Test video contents, pose arrays, official same-video PNG/TXT
labels, model outputs or annotations were opened. Stage 3.2g did not start.
