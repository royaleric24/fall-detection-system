"""Stage 3.2h metadata-only initialization and prepared annotation launch entry."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml.preprocessing.annotation_execution import (
    ANNOTATORS, PILOT_VERSION, PROTOCOL_VERSION, REAL_ROOT, REPO, ControlLogs,
    ExecutionError, ExecutionWorkspace, PilotSpec, collect_finalized, next_required_clip,
)
from ml.preprocessing.annotation_pilot import IndexedFrameSource, load_frozen, sha256_file
from ml.preprocessing.annotation_tool import TOOL_VERSION, AnnotationSession, annotator_payload


PARENT = "44d1543747f96fc8056adf2f71cdd6db3bdb07af"
VERSION = "stage32h_readiness_v1"
EXECUTION_VERSION = "stage32g_execution_v1"
ONBOARDING_VERSION = "stage32h_synthetic_onboarding_v1"
ONBOARDING_ROOT = REPO / "data/annotation_onboarding_synthetic/synthetic_onboarding"
G_ARTIFACTS = "artifacts/temporal_annotation/stage32g_execution_v1"
F_ARTIFACTS = "artifacts/temporal_annotation/stage32f_train_pilot_v1"
H_ARTIFACTS = "artifacts/temporal_annotation/stage32h_readiness_v1"
READINESS_FILES = (
    "ml/preprocessing/annotation_readiness.py", "tests/test_stage32h_readiness.py",
    "docs/stage32h_annotator_instructions.md",
    *(f"{H_ARTIFACTS}/{name}" for name in (
        "readiness.json", "launch_commands.json", "validation.json", "REPORT.md")),
)
RUNTIME_REQUIREMENTS = {
    "python_version": "3.13.15", "opencv_distribution": "opencv-contrib-python",
    "opencv_distribution_version": "4.12.0.88", "opencv_module_version": "4.12.0",
}
FROZEN_FILES = (
    "configs/stage32e_manual_annotation_protocol.json",
    "docs/stage32e_manual_annotation_protocol.md",
    "ml/preprocessing/manual_annotation_contract.py",
    "ml/preprocessing/annotation_pilot.py",
    "ml/preprocessing/annotation_tool.py",
    "ml/preprocessing/annotation_web.py",
    "ml/preprocessing/annotation_execution.py",
    *(f"{F_ARTIFACTS}/{name}" for name in (
        "freeze_record.json", "pilot_private_manifest.json", "pilot_annotator_manifest.json",
        "presentation_order.json", "pilot_selection_provenance.json", "decode_integrity.json",
        "tool_runtime_provenance.json", "validation_record.json", "REPORT.md")),
    *(f"{G_ARTIFACTS}/{name}" for name in (
        "execution_protocol.json", "workspace_policy.json", "completion_gate_spec.json",
        "synthetic_rehearsal_validation.json", "REPORT.md")),
)


def frozen_identities() -> dict[str, Any]:
    """Verify only committed contracts, code and metadata; never open dataset bytes."""
    hashes = {}
    for relative in FROZEN_FILES:
        baseline = subprocess.check_output(["git", "show", f"{PARENT}:{relative}"], cwd=REPO)
        expected = hashlib.sha256(baseline).hexdigest()
        if sha256_file(REPO / relative) != expected:
            raise ExecutionError(f"Frozen file changed: {relative}")
        hashes[relative] = expected
    spec = PilotSpec.from_frozen()
    if (spec.version, spec.protocol_version, spec.tool_version) != (
            PILOT_VERSION, PROTOCOL_VERSION, TOOL_VERSION):
        raise ExecutionError("Frozen version mismatch")
    if any(row["split"] != "train" or row["subject_id"] not in {1, 2, 3, 4, 8, 9}
           for row in spec.private_by_id.values()):
        raise ExecutionError("Only frozen Train identities are available")
    execution = json.loads((REPO / G_ARTIFACTS / "execution_protocol.json").read_text())
    if execution["version"] != EXECUTION_VERSION:
        raise ExecutionError("Frozen execution version mismatch")
    _, public, _ = load_frozen()
    for row in public:
        annotator_payload(row)
    return {
        "protocol_version": PROTOCOL_VERSION, "pilot_version": PILOT_VERSION,
        "tool_version": TOOL_VERSION, "tool_code_commit": PARENT,
        "execution_protocol_version": EXECUTION_VERSION,
        "expected_annotators": list(ANNOTATORS),
        "expected_neutral_clip_ids_in_presentation_order": list(spec.presentation_order),
        "presentation_order_sha256": spec.presentation_order_sha256,
        "file_sha256": hashes,
    }


def _safe_path(path: Path) -> None:
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ExecutionError("Symlinked readiness workspace rejected")


def _workspaces() -> dict[str, str]:
    _safe_path(REAL_ROOT)
    _safe_path(ONBOARDING_ROOT)
    if (REAL_ROOT.resolve() == ONBOARDING_ROOT.resolve()
            or REAL_ROOT.resolve() in ONBOARDING_ROOT.resolve().parents
            or ONBOARDING_ROOT.resolve() in REAL_ROOT.resolve().parents):
        raise ExecutionError("Genuine and synthetic namespaces must be separate")
    return {owner: str((REAL_ROOT / owner).resolve()) for owner in ANNOTATORS}


def _active_path() -> Path:
    return REAL_ROOT / "_control/run_initialization/active_run.json"


def _binding_path() -> Path:
    return REAL_ROOT / "_control/run_initialization/runtime_binding.json"


def _head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_active() -> dict[str, Any]:
    """Load the sole run binding; no annotation values are accessed."""
    workspaces = _workspaces()
    _safe_path(_active_path())
    value = json.loads(_active_path().read_text())
    if (value.get("version") != VERSION or value.get("parent_commit") != PARENT
            or re.fullmatch(r"stage32h-[0-9a-f]{32}", value.get("run_id", "")) is None
            or value.get("frozen") != frozen_identities()
            or value.get("genuine_workspaces") != workspaces
            or value.get("synthetic_onboarding_root") != str(ONBOARDING_ROOT.resolve())
            or value.get("agreement_state") != "LOCKED"
            or value.get("record_annotation_run_id") != PILOT_VERSION
            or value.get("session_id_rule") != "<run_id>-<annotator_id>-<uuid_hex>"):
        raise ExecutionError("Active run identity mismatch")
    entries = {path.name for path in _active_path().parent.iterdir()}
    if "active_run.json" not in entries or entries - {"active_run.json", "runtime_binding.json"}:
        raise ExecutionError("Unexpected additional run initialization")
    return value


def verify_runtime() -> dict[str, str]:
    """Check the reproducibility specification, independent of interpreter location."""
    if platform.python_version() != RUNTIME_REQUIREMENTS["python_version"]:
        raise ExecutionError("Required Python runtime is 3.13.15")
    try:
        distribution = importlib.metadata.version(RUNTIME_REQUIREMENTS["opencv_distribution"])
        import cv2
    except (importlib.metadata.PackageNotFoundError, ImportError) as exc:
        raise ExecutionError("Required OpenCV distribution is unavailable") from exc
    if (distribution != RUNTIME_REQUIREMENTS["opencv_distribution_version"]
            or cv2.__version__ != RUNTIME_REQUIREMENTS["opencv_module_version"]):
        raise ExecutionError("Required OpenCV runtime is opencv-contrib-python 4.12.0.88")
    return dict(RUNTIME_REQUIREMENTS)


def _verify_readiness_commit(commit: str) -> None:
    """Require the actual readiness freeze commit and unchanged committed files."""
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None or _head() != commit:
        raise ExecutionError("Repository HEAD does not match bound runtime commit")
    parent = subprocess.check_output(["git", "rev-parse", f"{commit}^"], cwd=REPO, text=True).strip()
    if parent != PARENT:
        raise ExecutionError("Runtime commit is not the Stage 3.2h freeze child")
    for relative in READINESS_FILES:
        try:
            committed = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=REPO)
        except subprocess.CalledProcessError as exc:
            raise ExecutionError("Runtime commit does not contain readiness implementation") from exc
        if sha256_file(REPO / relative) != hashlib.sha256(committed).hexdigest():
            raise ExecutionError("Readiness file differs from bound runtime commit")
    for options in (("diff", "--quiet", "HEAD", "--"), ("diff", "--cached", "--quiet", "HEAD", "--")):
        if subprocess.run(["git", *options], cwd=REPO, check=False).returncode != 0:
            raise ExecutionError("Tracked working tree/index differs from runtime commit")


def require_runtime_binding(active: dict[str, Any] | None = None) -> dict[str, Any]:
    """Non-video preflight used by every genuine launch, including prepare-only."""
    active = read_active() if active is None else active
    _safe_path(_binding_path())
    if not _binding_path().is_file():
        raise ExecutionError("Runtime commit must be bound after readiness freeze")
    binding = json.loads(_binding_path().read_text())
    if (binding.get("version") != "stage32h_runtime_binding_v1"
            or binding.get("run_id") != active["run_id"]
            or binding.get("parent_commit") != PARENT
            or binding.get("active_run_sha256") != sha256_file(_active_path())
            or binding.get("runtime_requirements") != RUNTIME_REQUIREMENTS):
        raise ExecutionError("Runtime binding identity mismatch")
    _verify_readiness_commit(binding.get("runtime_commit", ""))
    verify_runtime()
    return binding


def bind_runtime(commit: str) -> dict[str, Any]:
    """Bind the existing empty run after freeze; never initialize or replace a run."""
    active = empty_readiness()
    _verify_readiness_commit(commit)
    verify_runtime()
    _safe_path(_binding_path())
    if _binding_path().exists():
        binding = require_runtime_binding(active)
        if binding["runtime_commit"] != commit:
            raise ExecutionError("Existing runtime binding cannot be replaced")
        return binding
    binding = {
        "version": "stage32h_runtime_binding_v1", "run_id": active["run_id"],
        "parent_commit": PARENT, "runtime_commit": commit,
        "active_run_sha256": sha256_file(_active_path()),
        "runtime_requirements": dict(RUNTIME_REQUIREMENTS),
        "bound_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    _write_new(_binding_path(), binding)
    return require_runtime_binding(active)


def empty_readiness() -> dict[str, Any]:
    """Verify an initialized run has no sessions, drafts, finals or raw freeze."""
    value = read_active()
    if {entry.name for entry in REAL_ROOT.iterdir()} != {*ANNOTATORS, "_control"}:
        raise ExecutionError("Unexpected genuine workspace entry")
    for owner in ANNOTATORS:
        workspace = REAL_ROOT / owner
        _safe_path(workspace)
        if {path.name for path in workspace.iterdir()} != {"drafts", "finalized", "logs"}:
            raise ExecutionError("Unexpected annotator workspace layout")
        for name in ("drafts", "finalized", "logs"):
            folder = workspace / name
            _safe_path(folder)
            if not folder.is_dir() or any(folder.iterdir()):
                raise ExecutionError("Readiness requires empty genuine annotator workspaces")
    control = REAL_ROOT / "_control"
    if {path.name for path in control.iterdir()} != {
            "run_initialization", "protocol_questions", "technical_incidents", "raw_freezes"}:
        raise ExecutionError("Unexpected readiness control entry")
    for name in ("protocol_questions", "technical_incidents", "raw_freezes"):
        _safe_path(control / name)
        if any((control / name).iterdir()):
            raise ExecutionError("Readiness contains control records or a raw freeze")
    if any(collect_finalized(REAL_ROOT).values()):
        raise ExecutionError("Readiness cannot contain finalized annotations")
    return dict(value, finalized_counts={"A01": 0, "A02": 0}, expected_per_annotator=12,
                annotation_record_count=0, genuine_session_count=0,
                readiness_checks={
                    "frozen_metadata_and_code": "PASS", "train_only_identities": "PASS",
                    "public_manifest_strict_blinding": "PASS", "namespaces_separate": "PASS",
                    "genuine_workspaces_empty": "PASS", "agreement_locked": "PASS",
                    "dataset_open_required": False,
                })


def initialize_run() -> dict[str, Any]:
    """Exclusive-create one genuine identity; repeat initialization reuses it."""
    head = _head()
    if head != PARENT:
        raise ExecutionError("Initialization requires the frozen Stage 3.2g HEAD")
    frozen = frozen_identities()
    workspaces = _workspaces()
    if not _active_path().exists():
        if REAL_ROOT.exists():
            raise ExecutionError("Unbound existing genuine workspace; refusing to adopt it")
        value = {
            "version": VERSION, "run_id": "stage32h-" + uuid.uuid4().hex,
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "parent_commit": PARENT, "frozen": frozen,
            "genuine_workspaces": workspaces,
            "synthetic_onboarding_root": str(ONBOARDING_ROOT.resolve()),
            "agreement_state": "LOCKED", "record_annotation_run_id": PILOT_VERSION,
            "session_id_rule": "<run_id>-<annotator_id>-<uuid_hex>",
        }
        _write_new(_active_path(), value)
        for root in (REAL_ROOT, ONBOARDING_ROOT):
            for owner in ANNOTATORS:
                for name in ("drafts", "finalized", "logs"):
                    (root / owner / name).mkdir(parents=True, exist_ok=True)
            for name in ("protocol_questions", "technical_incidents", "raw_freezes"):
                (root / "_control" / name).mkdir(parents=True, exist_ok=True)
    return empty_readiness()


def prepare_launch(*, run_id: str, annotator_id: str, workspace: Path, pilot_version: str,
                   protocol_version: str, tool_version: str, execution_version: str,
                   resume_record_id: str | None = None, port: int = 8765) -> list[str]:
    """Validate binding and return frozen launcher argv, without launching/decoding."""
    if annotator_id not in ANNOTATORS:
        raise ExecutionError("Unapproved pseudonymous annotator ID")
    value = read_active()
    _safe_path(workspace)
    if (run_id != value["run_id"] or workspace.resolve() != (REAL_ROOT / annotator_id).resolve()
            or (pilot_version, protocol_version, tool_version, execution_version) != (
                PILOT_VERSION, PROTOCOL_VERSION, TOOL_VERSION, EXECUTION_VERSION)):
        raise ExecutionError("Prepared launch identity mismatch")
    require_runtime_binding(value)
    spec = PilotSpec.from_frozen()
    clip = next_required_clip(spec, annotator_id, collect_finalized(REAL_ROOT)[annotator_id])
    if clip is None:
        raise ExecutionError("Annotator has completed the frozen pilot")
    session_id = f"{run_id}-{annotator_id}-{uuid.uuid4().hex}"
    command = [sys.executable, "-B", "-m", "ml.preprocessing.annotation_execution",
               "--annotator-id", annotator_id, "--neutral-clip-id", clip,
               "--session-id", session_id, "--port", str(port)]
    if resume_record_id is not None:
        # Fail before AVI access if a resume request is outside this active run.
        from ml.preprocessing.annotation_execution import IsolatedStore

        store = IsolatedStore(REAL_ROOT, annotator_id, clip, PILOT_VERSION,
                              ControlLogs(REAL_ROOT), spec.source_fields(clip))
        draft = store.load_draft(resume_record_id)
        if not draft["annotator"]["session_id"].startswith(f"{run_id}-{annotator_id}-"):
            raise ExecutionError("Draft belongs to a different active run")
        command.extend(["--resume-record-id", resume_record_id])
    return command


def synthetic_session(annotator_id: str, *, practice_id: str = "practice-01",
                      resume_record_id: str | None = None) -> AnnotationSession:
    """Generated geometric images only; reuse frozen controls in separate storage."""
    if annotator_id not in ANNOTATORS or re.fullmatch(r"[A-Za-z0-9_-]+", practice_id) is None:
        raise ExecutionError("Invalid onboarding identity")
    _workspaces()
    import cv2
    import numpy as np

    clip = "clip-synthetic-onboarding-" + practice_id
    row = {"neutral_clip_id": clip, "source_video": "Subject.1/Synthetic/generated.avi",
           "subject_id": 1, "split": "train", "source_avi_sha256": "0" * 64,
           "expected_frame_count": 12, "source_fps": 20,
           "source_fps_provenance": "frozen_Stage_2_AVI_timeline"}
    spec = PilotSpec({clip: row}, (clip,), "0" * 64, "0" * 64, version=ONBOARDING_VERSION)
    frames = []
    for index in range(12):
        frame = np.zeros((180, 480, 3), dtype=np.uint8)
        cv2.putText(frame, "SYNTHETIC PRACTICE ONLY", (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 1)
        cv2.putText(frame, f"Generated frame {index}", (8, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 1)
        cv2.rectangle(frame, (index * 30 + 10, 90), (index * 30 + 40, 140), (80, 180, 255), -1)
        frames.append(frame)
    return ExecutionWorkspace(ONBOARDING_ROOT, synthetic=True).session(
        spec, annotator_id, clip, IndexedFrameSource(frames),
        {"neutral_clip_id": clip, "frame_count": 12, "duration_ms": 600},
        session_id="synthetic-" + uuid.uuid4().hex, resume_record_id=resume_record_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("initialize")
    commands.add_parser("check")
    bind = commands.add_parser("bind-runtime", help="Post-freeze metadata binding; opens no video")
    bind.add_argument("--runtime-commit", required=True)
    commands.add_parser("runtime-preflight", help="Check bound commit/environment; opens no video")
    commands.add_parser("environment-preflight", help="Check Python/OpenCV only; opens no video")
    launch = commands.add_parser("launch", help="Future authorized genuine session; do not run at readiness")
    launch.add_argument("--run-id", required=True)
    launch.add_argument("--annotator-id", required=True, choices=ANNOTATORS)
    launch.add_argument("--workspace", type=Path, required=True)
    launch.add_argument("--pilot-version", required=True)
    launch.add_argument("--protocol-version", required=True)
    launch.add_argument("--tool-version", required=True)
    launch.add_argument("--execution-version", required=True)
    launch.add_argument("--resume-record-id")
    launch.add_argument("--port", type=int, default=8765)
    launch.add_argument("--prepare-only", action="store_true")
    onboarding = commands.add_parser("onboarding")
    onboarding.add_argument("--annotator-id", required=True, choices=ANNOTATORS)
    onboarding.add_argument("--practice-id", default="practice-01")
    onboarding.add_argument("--resume-record-id")
    onboarding.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.command in {"initialize", "check"}:
        result = initialize_run() if args.command == "initialize" else empty_readiness()
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "bind-runtime":
        print(json.dumps(bind_runtime(args.runtime_commit), indent=2, sort_keys=True))
    elif args.command == "runtime-preflight":
        print(json.dumps(require_runtime_binding(), indent=2, sort_keys=True))
    elif args.command == "environment-preflight":
        print(json.dumps(verify_runtime(), indent=2, sort_keys=True))
    elif args.command == "launch":
        values = vars(args).copy()
        values.pop("command")
        prepare_only = values.pop("prepare_only")
        command = prepare_launch(**values)
        if prepare_only:
            print(json.dumps(command))
        else:
            subprocess.run(command, cwd=REPO, check=True)
    else:
        from ml.preprocessing.annotation_web import serve_web

        verify_runtime()
        session = synthetic_session(args.annotator_id, practice_id=args.practice_id,
                                    resume_record_id=args.resume_record_id)
        serve_web(session, args.port)


if __name__ == "__main__":
    main()
