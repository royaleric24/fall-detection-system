# Stage 3.2h instructions — identical for A01 and A02

Readiness preparation only. Genuine annotation has not been authorized.
Do not run a genuine launch command until the external Readiness Gate has
passed and the operator authorizes the human pilot. Use only your assigned
pseudonym, A01 or A02; do not enter a real name.

Read the frozen [Stage 3.2e protocol](stage32e_manual_annotation_protocol.md)
before practice. It defines event presence, boundaries, uncertainty, reason
flags and frame coordinates. These instructions add no annotation semantics.

Practice only with the generated synthetic onboarding images. They are
labeled **SYNTHETIC PRACTICE ONLY**, contain geometric shapes rather than
pilot videos, and have no research meaning. Use the same tool controls to
practice stepping forward/backward, jumping to a frame, selecting the
existing event/status controls, marking coordinates, saving and resuming a
draft, and validating/finalizing a practice record. No practice record counts
toward the genuine pilot.

From `/Users/eric/Developer/fall-detection-system`, use your assigned command:

```sh
uv run --python 3.13.15 --no-project --with opencv-contrib-python==4.12.0.88 python -B -m ml.preprocessing.annotation_readiness onboarding --annotator-id A01 --practice-id practice-01 --port 8765
uv run --python 3.13.15 --no-project --with opencv-contrib-python==4.12.0.88 python -B -m ml.preprocessing.annotation_readiness onboarding --annotator-id A02 --practice-id practice-01 --port 8766
```

The canonical environment specification is Python **3.13.15** and the
`opencv-contrib-python` distribution **4.12.0.88** (module **4.12.0**), using
the repository's established `uv run --no-project --with` mechanism. The
launcher rejects a different runtime. A temporary cache interpreter path is
not the environment specification. If offline package resolution is unavailable,
the operator may use an already-verified interpreter satisfying those exact
requirements; record it as the interpreter used for that specific run.
The operator can check it with `python -B -m
ml.preprocessing.annotation_readiness environment-preflight`, which opens no
video. Do not change dependency versions to work around resolution failure.
Practice storage is
`data/annotation_onboarding_synthetic/synthetic_onboarding/<your_id>/`.
After saving, ask the operator for your practice record ID and restart the
same practice command with `--resume-record-id <your_record_id>`. A finalized
practice original cannot be resumed; use a new `--practice-id practice-02`
for further practice. Practice never uses the genuine 12 clips or workspace.

When genuine annotation is later authorized, the launcher supplies the next
unfinished neutral clip in the one frozen presentation order. Do not skip,
reshuffle or substitute clips. Use the operator-prepared command for your
own pseudonym and workspace. The operator first binds this existing run to
the commit containing the reviewed Stage 3.2h readiness/launcher code.
Genuine launch rejects a missing binding, a different repository HEAD,
changed tracked files/index, or an incorrect Python/OpenCV runtime before
opening AVI. Passing this technical check is not human-launch authorization.
Inspect source AVI visual pixels only. The
zero-based AVI `frame_index` is authoritative; clip-relative milliseconds
are machine-derived. Never use activity/class metadata, revealing filenames,
pose landmarks or skeletons, MediaPipe missingness/features, model
predictions/confidence/errors/performance, Validation/Test evidence, future
alert outputs, or same-video official PNG/TXT labels to decide event presence
or boundaries. Do not transfer PNG/TXT coordinates to AVI.

You may rewatch and revise your own draft before finalization. Save a draft
to pause, then have the operator resume your own record in the same run.
Finalization validates the frozen schema and creates an immutable original;
there is no unlock or in-place correction. Do not finalize merely to test
the genuine tool. If a correction is needed after finalization, report it
without editing the original; later reviewed handling uses separate records.

Do not discuss pilot clips, event decisions, or boundary frames with the
other annotator. Do not view their screen, drafts, finals, answers or hints.
Do not request interim agreement results. Both streams must be complete and
raw originals frozen before any later authorized agreement/adjudication.

Report a semantic/protocol question to the operator with your pseudonym,
neutral clip ID (or no clip for a general question), protocol section and
description. The operator records it in `_control/protocol_questions` as
`semantic_protocol`, `OPEN`. It remains unresolved during the active pilot;
do not obtain the other annotator's interpretation or change definitions.
Procedural/tool questions may be answered without changing semantics.

Report technical incidents separately with your pseudonym, neutral clip ID,
failure category and description. The operator records them in
`_control/technical_incidents`; correctness-relevant failures block affected
clip finalization until resolved. Do not encode a tool failure as semantic
uncertainty or silently skip the clip. The tool provides application and
procedural isolation; the operator must keep workspace access and screens
separate. No shared operating-system account is claimed to provide secrecy.
