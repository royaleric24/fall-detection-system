# /// script
# requires-python = "==3.13.15"
# dependencies = ["mediapipe==0.10.35", "opencv-contrib-python==4.12.0.88", "numpy==2.2.6"]
# ///
"""Stage 2.6 preflight only: validate inputs and prepare a fresh run; never infer."""

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import tempfile
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from . import extract_pose_video as core
    from .assign_dataset_split import assign_rows
except ImportError:
    import extract_pose_video as core
    from assign_dataset_split import assign_rows

CHECKPOINT = "dd30d02100d3d77eb73ceb8b2e5ee8b90a6564e0"
FROZEN_FILES = (
    "ml/datasets/extract_pose_video.py", "ml/datasets/pose_raw.py",
    "ml/datasets/pose_runtime.py", "ml/datasets/assign_dataset_split.py",
    "ml/datasets/inspect_caucafall.py", "docs/pose_extraction_contract.md",
    "docs/dataset_protocol.md", "configs/dataset_split.json",
    "artifacts/dataset_inspection/inventory.csv",
    "artifacts/dataset_inspection/split_inventory.csv",
)
FIELDS = (
    "schema_version run_id dataset_name dataset_version subject_id original_activity "
    "binary_label split source_relative_video_path source_sha256 output_relative_path "
    "source_fps width height expected_frame_count extracted_frame_count "
    "pose_detected_count pose_missing_count timestamp_method status error_type "
    "error_message error_frame_index"
).split()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(core.REPO), *args], text=True).strip()


def verify_checkpoint() -> dict[str, str]:
    """Require checkpoint ancestry and byte-identical frozen production files."""
    git("merge-base", "--is-ancestor", CHECKPOINT, "HEAD")
    hashes = {}
    for name in FROZEN_FILES:
        path = core.REPO / name
        recorded = subprocess.check_output(
            ["git", "-C", str(core.REPO), "show", f"{CHECKPOINT}:{name}"])
        if path.read_bytes() != recorded:
            raise ValueError(f"Frozen checkpoint mismatch: {name}")
        hashes[name] = core.sha256(path)
    return hashes


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"Invalid CSV header: {path}")
        return list(reader)


def inventory_metadata() -> list[dict[str, Any]]:
    """Adapt validated inventory metadata for the explicitly authorized full scope.

    source_metadata retains its development-only test-subject guard. For train
    and validation rows it is also cross-checked; held-out rows use the same
    frozen assign_rows validation, without opening or decoding a video.
    """
    config = json.loads(core.SPLIT_CONFIG.read_text())
    rows = assign_rows(read_csv(core.INVENTORY), config)
    split_rows = read_csv(core.REPO / "artifacts/dataset_inspection/split_inventory.csv")
    if rows != split_rows:
        raise ValueError("Split inventory differs from validated frozen inventory")
    result = []
    for row in rows:
        dimensions = [float(row[k]) for k in ("width", "height", "frame_count")]
        if any(not v.is_integer() for v in dimensions):
            raise ValueError("Nonintegral inventory dimensions/count")
        meta = dict(subject_id=int(row["subject"].split(".")[1]),
                    original_activity=row["activity"], binary_label=row["label"],
                    split=row["split"], source_relative_video_path=row["relative_video_path"],
                    source_fps=float(row["fps"]), width=int(dimensions[0]),
                    height=int(dimensions[1]), expected_frame_count=int(dimensions[2]))
        if meta["split"] != "test" and core.source_metadata(
                core.SOURCE_ROOT / row["relative_video_path"]) != meta:
            raise ValueError("Development/full-run metadata disagreement")
        result.append(meta)
    if (len(result) != 100 or sum(r["expected_frame_count"] for r in result) != 19877
            or Counter(r["split"] for r in result) != {"train": 60, "validation": 20, "test": 20}):
        raise ValueError("Unexpected full-run scope; do not force counts")
    return result


def source_checksums(metadata: list[dict[str, Any]], root: Path) -> list[dict[str, str]]:
    """Hash actual AVI bytes; reject extras, omissions and symlink redirection."""
    expected = [r["source_relative_video_path"] for r in metadata]
    found = {p.relative_to(root).as_posix() for p in root.rglob("*")
             if p.suffix.lower() == ".avi"}
    if len(set(expected)) != len(expected) or found != set(expected):
        raise ValueError("Actual AVI set differs from declared inventory")
    result = []
    for relative in expected:
        path = root / relative
        if path.resolve() != path or not path.is_file() or not path.is_relative_to(root):
            raise ValueError(f"Noncanonical source path: {path}")
        result.append(dict(relative_video_path=relative, sha256=core.sha256(path)))
    return result


def manifest_rows(metadata: list[dict[str, Any]], checksums: list[dict[str, str]],
                  run_id: str) -> list[dict[str, Any]]:
    if [m["source_relative_video_path"] for m in metadata] != [c["relative_video_path"] for c in checksums]:
        raise ValueError("Checksum scope/order mismatch")
    return [dict.fromkeys(FIELDS, "") | meta | dict(
        schema_version=core.SCHEMA_VERSION, run_id=run_id, dataset_name="CAUCAFall",
        dataset_version="V5", source_sha256=checksum["sha256"],
        output_relative_path=Path(meta["source_relative_video_path"]).with_suffix(".npz").as_posix(),
        timestamp_method=core.TIMESTAMP_METHOD, status="pending")
        for meta, checksum in zip(metadata, checksums)]


def csv_text(rows: list[dict[str, Any]], fields: list[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def runtime() -> dict[str, Any]:
    """Inspect installed distributions/model; never import or initialize MediaPipe."""
    p = core.runtime_provenance(core.MODEL)
    expected = {"mediapipe": "0.10.35", "opencv-contrib-python": "4.12.0.88", "numpy": "2.2.6"}
    if p["python"] != "3.13.15" or any(p["packages"].get(k) != v for k, v in expected.items()):
        raise ValueError("Runtime differs from frozen pins")
    if p["model_sha256"] != core.MODEL_SHA256:
        raise ValueError("Model checksum differs from frozen model")
    import cv2
    if cv2.__version__ != "4.12.0":
        raise ValueError("Unexpected OpenCV runtime")
    p.update(opencv_runtime=cv2.__version__, os=platform.system(), architecture=platform.machine())
    return p


def identity_sha256(provenance: dict[str, Any]) -> str:
    """Detect accidental edits to immutable run identity, excluding lifecycle state."""
    identity = {k: v for k, v in provenance.items()
                if k not in ("identity_sha256", "status", "execution")}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()


def implementation_hashes() -> dict[str, str]:
    """Include the executor and all tests before any official run is declared."""
    names = set(FROZEN_FILES) | {"ml/datasets/preflight_pose_run.py",
                               "ml/datasets/execute_pose_run.py"}
    names.update(p.relative_to(core.REPO).as_posix()
                 for p in (core.REPO / "tests").glob("test_*.py"))
    return {name: core.sha256(core.REPO / name) for name in sorted(names)}


def run_tests() -> dict[str, Any]:
    command = [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"]
    result = subprocess.run(command, cwd=core.REPO, text=True, capture_output=True)
    if result.returncode:
        raise ValueError("Lightweight tests failed:\n" + result.stdout + result.stderr)
    return dict(command=command, exit_code=result.returncode, output=result.stdout + result.stderr)


def prepare(check_only: bool = False) -> dict[str, Any]:
    """Validate first; publish preflight artifacts only from clean committed code.

    This command does not implement extraction or resume. A completed preflight
    is not permission to infer without rechecking its source/code/runtime hashes.
    """
    initial_status = git("status", "--porcelain", "--untracked-files=all")
    if initial_status and not check_only:
        raise ValueError("Commit/review orchestration first: clean worktree required")
    initial_commit = git("rev-parse", "HEAD")
    verify_checkpoint()
    metadata = inventory_metadata()
    checksums = source_checksums(metadata, core.SOURCE_ROOT)
    provenance = runtime()
    implementation = implementation_hashes()
    tests = run_tests()
    if any(core.sha256(core.REPO / p) != h for p, h in implementation.items()):
        raise ValueError("Implementation changed during preflight")
    if (source_checksums(metadata, core.SOURCE_ROOT) != checksums
            or core.sha256(core.MODEL) != provenance["model_sha256"]):
        raise ValueError("Source/model changed during preflight")
    if git("rev-parse", "HEAD") != initial_commit or git("status", "--porcelain", "--untracked-files=all") != initial_status:
        raise ValueError("Git state changed during preflight")
    summary = dict(checks="passed", clean_worktree=not bool(initial_status),
                   git_commit=initial_commit, videos=100, expected_frames=19877,
                   split_counts=dict(Counter(r["split"] for r in metadata)),
                   inference_performed=False, tests=tests)
    if check_only:
        return summary | dict(status="check_only", run_id=None, artifacts_created=False)
    run_id = str(uuid.uuid4())
    run_dir = core.REPO / "artifacts/pose_extraction/runs" / run_id
    npz_root = core.REPO / "data/interim/caucafall_v5/pose_raw_v1_runs" / run_id
    core.output_path(run_dir.parent, Path(run_id))
    core.output_path(npz_root.parent, Path(run_id))
    # Atomic directory publication: a failed preflight never exposes a partial run.
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=run_dir.parent, prefix=".preflight-") as tmp:
        stage = Path(tmp) / run_id
        stage.mkdir()
        (stage / "results").mkdir()
        (stage / "source_checksums.csv").write_text(csv_text(checksums, ["relative_video_path", "sha256"]), encoding="utf-8")
        (stage / "manifest.csv").write_text(csv_text(manifest_rows(metadata, checksums, run_id), FIELDS), encoding="utf-8")
        checksum_path = (run_dir / "source_checksums.csv").relative_to(core.REPO).as_posix()
        (stage / "tests_before.txt").write_text(tests["output"], encoding="utf-8")
        test_path = (run_dir / "tests_before.txt").relative_to(core.REPO).as_posix()
        provenance.update(schema_version=core.SCHEMA_VERSION, run_id=run_id,
                          dataset_name="CAUCAFall", dataset_version="V5", status="pending",
                          created_at_utc=datetime.now(timezone.utc).isoformat(),
                          checkpoint_commit=CHECKPOINT, git_commit=initial_commit, git_dirty=False,
                          output_root=str(npz_root), result_root=str(run_dir / "results"),
                          scope=[m["source_relative_video_path"] for m in metadata],
                          source_checksum_manifest_path=checksum_path,
                          file_sha256=implementation | {checksum_path: core.sha256(stage / "source_checksums.csv"),
                                                       test_path: core.sha256(stage / "tests_before.txt")},
                          source_checksum_baseline="Current bytes only; historical immutability not certified",
                          historical_evidence_policy="No reuse; all 100 outputs must be newly extracted",
                          fresh_tracker_per_video=True, preflight=summary)
        provenance["identity_sha256"] = identity_sha256(provenance)
        (stage / "extraction_run.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        # mkdir claims the UUID without overwriting another run; rename only into
        # our empty reservation. On failure, preserve any unexpectedly nonempty dir.
        run_dir.mkdir()
        try:
            os.rename(stage, run_dir)
        except BaseException:
            if run_dir.exists() and not any(run_dir.iterdir()):
                run_dir.rmdir()
            raise
    return summary | dict(status="preflight_complete_extraction_pending", run_id=run_id,
                          run_directory=str(run_dir), output_root=str(npz_root))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Read-only checks, allowed before committing; no run ID or artifacts")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.check), indent=2))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Preflight failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
