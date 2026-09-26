# Stage 3.2c PNG-native pose feasibility pilot

Pending external Gate Review; no PASS claim.

Frozen full Train sequences: 10; PNGs: 1788; pose detected: 1151; missing: 637; availability: 0.643736.

## Per-sequence availability

| Sequence | PNGs | Detected | Missing | Availability |
| --- | ---: | ---: | ---: | ---: |
| Subject.2/Fall backwards/FallBackwardsS2.avi | 121 | 99 | 22 | 0.818182 |
| Subject.8/Hop/HopS8.avi | 167 | 106 | 61 | 0.634731 |
| Subject.8/Pick up object/PickupobjectS8.avi | 267 | 28 | 239 | 0.104869 |
| Subject.9/Walk/WalkS9.avi | 235 | 183 | 52 | 0.778723 |
| Subject.4/Fall forward/FallForwardS4.avi | 169 | 110 | 59 | 0.650888 |
| Subject.4/Pick up object/PickupobjectS4.avi | 180 | 125 | 55 | 0.694444 |
| Subject.9/Fall sitting/FallSittingS9.avi | 217 | 205 | 12 | 0.944700 |
| Subject.8/Fall forward/FallForwardS8.avi | 123 | 0 | 123 | 0.000000 |
| Subject.1/Fall backwards/FallBackwardsS1.avi | 126 | 114 | 12 | 0.904762 |
| Subject.1/Hop/HopS1.avi | 183 | 181 | 2 | 0.989071 |

## By original annotation state

| State | PNGs | Detected | Missing | Availability |
| --- | ---: | ---: | ---: | ---: |
| nofall | 1307 | 822 | 485 | 0.628921 |
| grounded_fall_state | 481 | 329 | 152 | 0.683992 |

Subject.8 / Fall forward: 0/123 PNG poses; frozen Stage 2 AVI had 0/122. This is a source-level diagnostic only, with no frame mapping or causal conclusion.

Exact duplicate diagnostic: 49/49 comparison pairs exactly equal. Repeatability diagnostic: 8/8 exactly equal. See CSVs for per-pair state, structure and exact float32 differences; no tolerance is used.

Inference/encoding exception records: 0. Such frames remain explicit all-NaN missing observations; no frame is skipped.

The sample and repeatability subset were frozen before inference. This is a selected feasibility stress pilot, not an estimate for all Train videos or a classifier score.

MediaPipe Full float16 v1, CPU, IMAGE mode processed original-resolution OpenCV BGR PNG decodes converted to RGB. IMAGE mode needs no timestamp. Annotation identity and labels are stored only in the separate manifest. Each NPZ has exactly ordinal, detection and raw landmark arrays. No physical PNG FPS or timestamp was created.

No AVI/PNG frame alignment, preprocessing, windows or training occurred. Validation/Test contents were not accessed. All downstream decisions remain UNDECIDED.

## Runtime execution note

The first sandboxed `run` invocation verified the bytes of all frozen pilot PNGs and TXTs, then aborted with exit code 134 during IMAGE graph initialization before decoding images for inference or producing an NPZ. MediaPipe logged `DrishtiMetalHelper ... Service is unavailable` after macOS graphics-context creation failed. A separate, unsandboxed graph-initialization diagnostic succeeded without reading PNGs or running inference. The unchanged frozen pilot then completed once with macOS graphics access. The successful run logged the MediaPipe feedback-manager and NORM_RECT warnings; no per-image inference/encoding exception occurred. This environment dependency limits portability and is recorded in `runtime_execution_notes.json`.
