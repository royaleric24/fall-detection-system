# Stage 3.2h — readiness provenance correction and freeze

This tracked document is the pre-freeze verification snapshot. The external
Readiness Gate remains HOLD pending this freeze/provenance correction and
subsequent review; genuine human annotation is not authorized.

The preserved run is `stage32h-6747d0191488418d932955fda609ec0e`. The frozen
parent before readiness work is `44d1543747f96fc8056adf2f71cdd6db3bdb07af`.
That parent contains the prior-stage tool/execution baseline, **not** the
Stage 3.2h readiness/launcher implementation added by this freeze.

## Non-circular runtime provenance

The freeze commit contains exactly the seven reviewed Stage 3.2h files.
Its hash is not guessed or embedded in its own tracked artifacts.
[readiness.json](readiness.json) preserves the parent and records
`runtime_commit_bound_after_freeze: null` as this pre-freeze snapshot.
The existing initialization marker is preserved byte-for-byte at:

`data/annotation_runs/stage32f_train_pilot_v1/_control/run_initialization/active_run.json`

After the commit exists, the operator uses the metadata-only `bind-runtime
--runtime-commit <actual_freeze_commit>` entry. It exclusive-creates:

`data/annotation_runs/stage32f_train_pilot_v1/_control/run_initialization/runtime_binding.json`

This Git-ignored binding records the existing run ID, exact runtime commit,
parent, original active-marker SHA-256, runtime requirements and binding time.
It does not create another run or annotation. Rebinding cannot replace the
original runtime commit. `runtime-preflight` opens no video.

Every genuine `launch`, including `--prepare-only`, rejects missing or
mismatched binding, wrong HEAD, a runtime commit without the readiness files,
a different freeze parent, changed committed readiness bytes, dirty tracked
working tree/index, or wrong Python/OpenCV identity **before AVI access**.
The check compares current readiness files to the Git tree of the bound
commit; this is not a circular self-hash stored within that commit.
Post-commit acceptance and deliberately mismatched-HEAD rejection are checked
without executing either genuine launch command. Passing provenance checks
is not permission for human annotation; external Readiness Gate Review and
later explicit human-launch authorization remain required.

## Preserved genuine state

| Annotator | Workspace | Finalized |
| --- | --- | --- |
| A01 | `data/annotation_runs/stage32f_train_pilot_v1/A01` | 0/12 |
| A02 | `data/annotation_runs/stage32f_train_pilot_v1/A02` | 0/12 |

Both have empty drafts/finalized/logs, zero genuine records and zero genuine
sessions. Agreement is **LOCKED**. There is no raw annotation freeze.
The frozen 12 neutral IDs/order, `stage32e_v1` protocol,
`stage32f_train_pilot_v1` pilot and `stage32g_execution_v1` execution rules
remain unchanged. The 21 prior-stage hashes remain pinned to the parent.
Frozen record `creation.annotation_run_id` still equals the pilot version;
the distinct Stage 3.2h run binds through the original marker, new runtime
binding and run-prefixed session IDs. No record semantics changed.

## Frozen tool audit — Case A

`stage32f_local_tk_v1` is the authoritative **historical frozen identifier**,
not an identifier introduced by Stage 3.2h. It is present at line 20 of
[annotation_tool.py](../../../ml/preprocessing/annotation_tool.py), including
its bytes in frozen Stage 3.2f commit
`6f0af44710121afd43a8f9e077a5bdd09bd8dec5`.
The [frozen Stage 3.2f report](../stage32f_train_pilot_v1/REPORT.md), lines
49–51, describes the actual implementation: standard-library loopback
HTTPServer viewed in a browser after the Tk window probe failed.
[annotation_web.py](../../../ml/preprocessing/annotation_web.py), lines
249–256, implements the server bound to `127.0.0.1`.
There is no active Tk UI. The historical identifier is preserved; no frozen
Stage 3.2e/f/g file is modified.

## Stable environment and human launch specification

Canonical requirements are Python **3.13.15**, distribution
`opencv-contrib-python` **4.12.0.88**, module **4.12.0**. Canonical commands
use the repository's established `uv run --python 3.13.15 --no-project --with
opencv-contrib-python==4.12.0.88 python -B -m ...` mechanism. The launcher
checks the actual runtime, independent of executable location. No dependency
was changed to make offline resolution succeed.

[launch_commands.json](launch_commands.json) has separate identity-bound
A01/A02 commands, prepared only and never executed. The common
[annotator instructions](../../../docs/stage32h_annotator_instructions.md)
use the reproducibility specification, not a permanent temporary-cache path.
An already-verified interpreter can be a specific-run fallback when resolution
is unavailable; it must pass `environment-preflight` and bound-runtime checks.
The cached interpreter recorded below was used for these tests only and is
not a canonical long-term launcher identity.

Synthetic onboarding remains in the separate empty
`data/annotation_onboarding_synthetic/synthetic_onboarding` namespace.
It generates geometric images in memory and uses the unchanged controls.
No real pilot clip is used for practice. Existing synthetic tests verify
navigation, boundary controls, draft resume, immutable finals, blinding and
cross-owner isolation; no genuine record is produced.

## Executed checks

The readiness suite passed **18 tests in 8.146s, OK**. The full required
readiness plus Stage 3.2g/f/e/d and Stage 3 regressions passed:

```text
Ran 118 tests in 10.767s
OK
```

```sh
/private/tmp/caucafall-stage26-cache/archive-v0/j01PHKlaO3SeMqsD/bin/python -B -m unittest tests.test_stage32h_readiness tests.test_stage32g_execution_protocol tests.test_stage32f_annotation_tool tests.test_stage32e_manual_annotation_protocol tests.test_stage32d_temporal_supervision_strategy tests.test_stage3_contract -v
```

Offline `uv` failed dependency resolution before executing tests. The fallback
was verified as Python 3.13.15 / OpenCV distribution 4.12.0.88, module 4.12.0.
The existing synthetic loopback regression required sandbox escalation to
bind localhost. Tests use temporary namespaces; mock commit identities are
explicitly unit-test-only. Wrong-HEAD checks block before any real decoder
or child launcher, and no test binds the actual genuine run.

`git diff --check` and `git diff --cached --check` are required before freeze;
new-file whitespace is also checked before staging. The staged path set is
restricted to the seven reviewed files. [validation.json](validation.json)
records this pre-commit evidence and hashes of the corrected implementation,
instructions and artifacts. The exact runtime commit is subsequently recorded
only in Git-ignored binding state and the final Codex report.

No new genuine run, pilot selection, order or protocol was created. No real
pilot pixels, Validation/Test contents, pose/model or official PNG/TXT evidence
were opened. No genuine annotation/session, raw freeze, agreement, adjudication,
preprocessing/window/model-ready dataset or training was produced.
After the single freeze commit and metadata-only runtime binding/preflight,
stop for external Readiness Gate Review without launching A01 or A02.
