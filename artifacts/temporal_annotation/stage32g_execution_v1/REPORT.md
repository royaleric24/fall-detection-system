# Stage 3.2g — dual-annotator execution and workspace isolation

Status: **HOLD — integration addendum submitted for external Gate Review**. Parent commit:
`6f0af44710121afd43a8f9e077a5bdd09bd8dec5`. This stage prepares the
future controlled pilot. It does not authorize real annotation, real-clip
display, agreement analysis, adjudication, or Stage 3.2h execution.

The frozen Stage 3.2f pilot remains exactly 12 Train AVIs with the same
neutral IDs and presentation order. Both future annotators use that one order,
the `stage32e_v1` protocol, and the frozen Stage 3.2f tool baseline. The two
research identities are pseudonyms `A01` and `A02`; real names are neither
required nor stored. Each annotator creates one independent annotation stream.
Neither may view or discuss the other's clip decisions or specific boundary
frames before both raw streams are complete and frozen. No interim agreement
or adjudication is allowed. Pose, model and same-video PNG/TXT evidence remain
excluded from manual annotation.

The controlled future launcher is `python -B -m
ml.preprocessing.annotation_execution` with an explicit approved annotator
ID, neutral clip ID and session ID. It accepts no arbitrary draft path, checks
the pinned manifests and source AVI hash, refuses a clip other than the next
unfinished item in the common frozen presentation order, and routes the session into
`data/annotation_runs/stage32f_train_pilot_v1/<annotator_id>/`. This root is
Git-ignored. The old Stage 3.2f direct launcher is a frozen historical tool,
not the authorized dual-annotator pilot entry point. The Stage 3.2g store
binds one pseudonym and clip, separates `drafts/`, `finalized/`, and `logs/`,
and refuses cross-owner draft reads via its application API. The browser is
still restricted to the current session's neutral state. This is
**application/procedural isolation**, not OS permissions or cryptographic
secrecy; operators must also keep workspaces and screens separate.

Drafts can be saved, resumed, revised, and rewound within the owner's
workspace. Drafts never count as completed research records. Finalization
uses the Stage 3.2e validator, checks unresolved blocking incidents, rejects a
second final original for the same clip/annotator, and uses exclusive-create
storage. There is no unlock operation. Future correction/adjudication uses a
separate record, preserving both originals.

The completion checker accepts byte snapshots of finalized files and checks
structure and provenance only. It requires exactly `A01` and `A02`, each
with one valid `FINALIZED` original for every one of the 12 frozen neutral
IDs, matching source/protocol/pilot/tool identity, no extra or duplicate
clip, and SHA-256 of each final file. Drafts cannot substitute. Its result
contains counts, hashes and integrity errors; it never compares annotation
values across annotators. The future agreement guard rejects any incomplete
set or absent/mismatched raw freeze. It exposes no partial agreement.

After future dual completion and before any analysis, an operator must write
one exclusive-create raw freeze with protocol/pilot/tool versions, both
pseudonyms, full ordered clip set and order hash, source-manifest hash, all
final file paths and SHA-256 hashes, UTC completion time, and code commit.
Future analysis must consume only files matching that freeze. Exclusive-create
prevents tool overwrite; a current snapshot alone cannot prove earlier file
history, while a post-freeze hash check catches subsequent mutation.

Protocol questions live separately in `_control/protocol_questions`. A
procedural/tool question may be answered if semantics do not change. A
semantic question remains open throughout the active pilot and is reviewed
only after raw freeze; the frozen Stage 3.2e definitions are not edited
mid-pilot. Technical failures live separately in
`_control/technical_incidents`, with category, severity/blocking state,
description and resolution metadata. They are not encoded as annotation
`uncertain` or `unjudgeable`; an unresolved blocking incident prevents the
affected clip's finalization. The operator must resolve it, not silently skip
the clip.

Synthetic onboarding uses mock frames and
`data/annotation_onboarding_synthetic`, never a real pilot AVI or the real
annotation namespace. Real annotation output is not automatically committed
to Git. No genuine annotation, agreement calculation or genuine raw freeze
was produced in Stage 3.2g. The synthetic rehearsal below is test evidence only.

Machine-readable rules are in [execution_protocol.json](execution_protocol.json),
[workspace_policy.json](workspace_policy.json), and
[completion_gate_spec.json](completion_gate_spec.json).

## Verification

```sh
UV_CACHE_DIR=/private/tmp/caucafall-stage26-cache uv run --offline --python 3.13.15 --no-project --with opencv-contrib-python==4.12.0.88 python -B -m unittest tests.test_stage32g_execution_protocol tests.test_stage32f_annotation_tool tests.test_stage32e_manual_annotation_protocol tests.test_stage32d_temporal_supervision_strategy tests.test_stage3_contract -v
git diff --check
```

The initial combined suite ran **99 tests, OK**. Its pre-existing synthetic loopback
HTTP test required execution outside the sandbox because that environment
blocks even `127.0.0.1` binding. Stage 3.2g tests used synthetic metadata,
frames, records, questions and incidents in temporary synthetic workspaces.
No real pilot clip was shown, no real annotation was created, no agreement
was calculated, and no Validation/Test contents were opened.

## Integration addendum: synthetic dual-annotator rehearsal

The [test-only rehearsal helper](../../../tests/stage32g_synthetic_rehearsal.py)
uses `PilotSpec.from_frozen()` metadata and the reviewed
`AnnotationSession`/`IsolatedStore` finalization path in a temporary
`synthetic_rehearsal/` namespace. It does not invoke the genuine AVI launcher
or change its workspace policy. All 24 fixture records have deterministic
`synthetic-rehearsal-*` record/session IDs and an explicit synthetic/test-only
note. Frame sources contain nonvisual placeholders; AVI-open/decode routes
and data-content reads are blocked during the rehearsal. The fixture uses
the exact frozen 12 neutral IDs, pilot version and common order.

Both streams finalize 12 schema-valid originals through storage. Their
deliberately distinct fixture decisions do not affect structural completion;
no agreement measure or disagreement magnitude is calculated. The actual
completion checker rejects 11/12 + 12/12, 12/12 + 0/12, and 12/12 + 11/12,
and accepts structural completeness for 12/12 + 12/12. The guard is exercised
chronologically before dual completion, after completion before freeze, and
after the matching exclusive-create raw freeze: DENY, DENY, then
ALLOW_PREREQUISITES_ONLY. That final state authorizes no analysis in this stage.

The test-only raw freeze captures both owners, the exact ordered clip set,
versions/order identity, all 24 finalized file identities and SHA-256 hashes,
fixture timestamp and code baseline. The evidence separately records the
uncommitted execution-module and rehearsal-helper hashes. A second freeze is
rejected without changing the first. In a disposable copy, a schema-preserving
byte change makes the stored hash mismatch, invalidates the freeze and closes
the guard. Ownership, duplicate-original, protocol/pilot version, missing/extra
clip, wrong neutral ID, and order/order-hash checks all reject their invalid
fixtures. Separate question/incident files leave annotation semantics and the
frozen protocol unchanged. Temporary records and the tamper copy are discarded;
only [synthetic_rehearsal_validation.json](synthetic_rehearsal_validation.json)
is retained as compact machine-readable test evidence.

The latest required suite ran **100 tests in 0.382s, OK**. The offline `uv`
resolver could not find OpenCV package metadata, so the existing cached
environment was verified as Python **3.13.15**, OpenCV **4.12.0**, distribution
**4.12.0.88**, and executed directly:

```sh
/private/tmp/caucafall-stage26-cache/archive-v0/j01PHKlaO3SeMqsD/bin/python -B -m unittest tests.test_stage32g_execution_protocol tests.test_stage32f_annotation_tool tests.test_stage32e_manual_annotation_protocol tests.test_stage32d_temporal_supervision_strategy tests.test_stage3_contract -v
```

The prescribed Stage 3.2f regressions retain their synthetic codec/localhost
tests; the localhost test runs outside the sandbox. The integration rehearsal
itself decodes no AVI. No real video/data content, pose, MediaPipe/model,
PNG/TXT or Validation/Test content was accessed. No genuine annotation was
created. Nothing was staged or committed, and Stage 3.2h was not started.
