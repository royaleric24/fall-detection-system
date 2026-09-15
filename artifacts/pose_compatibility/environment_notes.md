# Environment checks (2026-09-14)

No repository `pyproject.toml` or `.venv` existed at inspection. Existing uv-managed
Python is 3.13.15 on macOS arm64. System Python 3.9 is not used for the experiment.

1. Current official MediaPipe 1.0.1 was selected after checking the official Tasks
   Python guide and current PyPI macOS arm64 wheel. Installation initially timed
   out fetching dependencies; a retry installed successfully.
2. The real CPU PoseLandmarker creation with the official Full model aborted
   (exit 134), before processing any frames:
   `graph_service.h:139 Check failed: service_ Service is unavailable.`
   Stack included `DrishtiMetalHelper` and `TensorsToDetectionsCalculator::Open`.
   The signature is consistent with [upstream issue 6356](https://github.com/google-ai-edge/mediapipe/issues/6356).
   That report used Python 3.14.5; this experiment used Python 3.13.15.
   Similar signatures do not prove an identical root cause.
   It is not evidence that Python 3.13 cannot import MediaPipe.
3. A separate 1.0.1 GPU initialization check printed Python 3.13.15 and MediaPipe
   1.0.1, then failed with `Could not create an NSOpenGLPixelFormat` in the
   restricted execution environment. No GPU pose results were produced.
4. A targeted fallback selects MediaPipe 0.10.35 with the same modern Tasks API,
   official Full float16 v1 model, and CPU delegate. Python stays at 3.13.15.
   The experiment output records the actual runtime and outcome.

The pose environment pins opencv-contrib-python 4.12.0.88, required by MediaPipe,
and numpy 2.2.6. It excludes opencv-python-headless to avoid two packages providing
`cv2`. The existing isolated full-media inspector still uses headless 4.12.0.88.
No global pip install or architecture/model-family substitution was performed.

Official references:
- https://developers.google.com/edge/mediapipe/solutions/setup_python
- https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python
- https://pypi.org/project/mediapipe/1.0.1/

## Repository audit clarification (2026-09-15)

The scripts and recorded experiment commands read raw source files without
writing to them; all diagnostics are outside `data/raw`. Git tracks/stages no
raw files, and the current AVI/PNG/TXT counts match the inspection report.
There is no pre-experiment checksum manifest, so this audit cannot independently
prove historical byte-for-byte immutability. An ignored Finder `.DS_Store` is
present at `data/raw/caucafall_v5/CAUCAFall/Subject.5/.DS_Store`; it was left
untouched and is not a dataset annotation or a recommended commit artifact.
