# /// script
# requires-python = "==3.13.15"
# dependencies = ["mediapipe==0.10.35", "opencv-contrib-python==4.12.0.88", "numpy==2.2.6"]
# ///
"""Execute one reviewed, preflight-created Stage 2.6 run; never create a run ID."""

import argparse
import json
import os
import re
import subprocess
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from . import preflight_pose_run as pre
    from .pose_raw import load_validated
except ImportError:
    import preflight_pose_run as pre
    from pose_raw import load_validated

core = pre.core
STATES = {"pending", "running", "complete", "failed", "incomplete"}
COUNTERS = ("extracted_frame_count", "pose_detected_count", "pose_missing_count")
ERRORS = {"invalid_source_metadata", "malformed_pose_result", "video_decode_failure",
          "inference_runtime_failure", "output_persistence_failure", "provenance_mismatch"}
MUTABLE = {"status", "error_type", "error_message", "error_frame_index", *COUNTERS}


class RunBlocked(ValueError):
    """Stop for review: do not repair, overwrite or infer through inconsistent evidence."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RunBlocked(message)


def canonical(path: Path) -> Path:
    path = path.absolute()
    require(path.resolve() == path, f"Symlink or noncanonical path: {path}")
    return path


def atomic_text(path: Path, text: str) -> None:
    """Same-directory replace; an interrupted write preserves the previous ledger."""
    canonical(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".ledger-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def transition(row: dict[str, str], status: str, result: dict[str, Any] | None = None) -> None:
    allowed = {"pending": {"running"}, "running": {"complete", "failed", "incomplete"}}
    require(status in allowed.get(row["status"], set()), f"Invalid transition {row['status']} -> {status}")
    if status == "running":
        row.update({key: "0" for key in COUNTERS})
    elif result is not None:
        row.update({key: str(result[key]) for key in COUNTERS})
        row.update({key: "" if result.get(key) is None else str(result[key])
                    for key in ("error_type", "error_message", "error_frame_index")})
    row["status"] = status


def metrics(arrays: dict[str, Any]) -> dict[str, Any]:
    """Recompute missing intervals independently from the authoritative boolean mask."""
    intervals = []
    for index, present in enumerate(arrays["pose_detected"]):
        if not present:
            if intervals and intervals[-1][1] == index - 1:
                intervals[-1][1] = index
            else:
                intervals.append([index, index])
    count = len(arrays["frame_index"])
    detected = int(arrays["pose_detected"].sum())
    return dict(extracted_frame_count=count, pose_detected_count=detected,
                pose_missing_count=count - detected, missing_frame_intervals=intervals,
                longest_missing_run=max((end - start + 1 for start, end in intervals), default=0))


class Executor:
    def __init__(self, run_dir: Path, reviewed_commit: str):
        self.run_dir = canonical(run_dir)
        self.reviewed_commit = reviewed_commit
        require(re.fullmatch(r"[0-9a-f]{40}", reviewed_commit) is not None,
                "Supply the full reviewed execution commit SHA")
        run_id = self.run_dir.name
        require(str(uuid.UUID(run_id)) == run_id, "Run ID must be a canonical UUID")
        require(self.run_dir.parent == core.REPO / "artifacts/pose_extraction/runs", "Unexpected run namespace")
        for name in ("extraction_run.json", "manifest.csv", "source_checksums.csv", "tests_before.txt", "results"):
            path = canonical(self.run_dir / name)
            require(path.is_dir() if name == "results" else path.is_file(), f"Missing run input: {name}")
        self.provenance = json.loads((self.run_dir / "extraction_run.json").read_text())
        self.run_id = run_id
        p = self.provenance
        require(p.get("run_id") == run_id, "Run ID mismatch")
        self.output_root = canonical(core.REPO / "data/interim/caucafall_v5/pose_raw_v1_runs" / run_id)
        require(p.get("output_root") == str(self.output_root), "NPZ namespace/run ID mismatch")
        require(p.get("result_root") == str(self.run_dir / "results"), "Result namespace mismatch")
        require(p.get("source_root") == str(core.SOURCE_ROOT), "Source namespace mismatch")
        self.rows = pre.read_csv(self.run_dir / "manifest.csv")
        self.metadata: list[dict[str, Any]] = []
        self.checksums: list[dict[str, str]] = []

    def allowed_artifacts(self) -> set[str]:
        root = self.run_dir.relative_to(core.REPO)
        names = {"extraction_run.json", "manifest.csv", "source_checksums.csv", "tests_before.txt",
                 "tests_after.txt", "summary.json", "FULL_EXTRACTION_REPORT.md", "executor.lock"}
        names.update("results/" + str(Path(r["source_relative_video_path"]).with_suffix(".json"))
                     for r in self.metadata)
        return {(root / name).as_posix() for name in names}

    def verify_git(self) -> None:
        require(pre.git("rev-parse", "HEAD") == self.reviewed_commit == self.provenance.get("git_commit"),
                "HEAD/run provenance must equal the explicitly reviewed execution commit")
        require(self.provenance.get("git_dirty") is False, "Run did not begin from clean code")
        data = subprocess.check_output(["git", "-C", str(core.REPO), "status", "--porcelain=v1", "-z", "--untracked-files=all"])
        allowed = self.allowed_artifacts()
        for entry in filter(None, data.decode().split("\0")):
            require(len(entry) >= 4 and entry[:2] not in ("R ", " C", "C ", " R"), "Unexpected rename/copy")
            require(entry[3:] in allowed, f"Unrelated dirty path: {entry}")

    def verify_identity(self) -> None:
        p = self.provenance
        require(p.get("identity_sha256") == pre.identity_sha256(p), "Altered run provenance")
        require(p.get("dataset_name") == "CAUCAFall" and p.get("dataset_version") == "V5", "Dataset identity mismatch")
        require(p.get("checkpoint_commit") == pre.CHECKPOINT and p.get("schema_version") == core.SCHEMA_VERSION,
                "Checkpoint/schema mismatch")
        pre.verify_checkpoint()
        self.verify_git()
        checksum_path = (self.run_dir / "source_checksums.csv").relative_to(core.REPO).as_posix()
        test_path = (self.run_dir / "tests_before.txt").relative_to(core.REPO).as_posix()
        expected = pre.implementation_hashes()
        expected.update({checksum_path: core.sha256(self.run_dir / "source_checksums.csv"),
                         test_path: core.sha256(self.run_dir / "tests_before.txt")})
        require(p.get("file_sha256") == expected, "Implementation/configuration/evidence hash mismatch")
        require(p.get("source_checksum_manifest_path") == checksum_path, "Checksum manifest path mismatch")
        actual = pre.runtime()
        # git_dirty changes only because owned run artifacts are written; no other
        # runtime/config/source provenance field may differ from preflight.
        for key, value in actual.items():
            if key not in ("git_dirty", "file_sha256"):
                require(p.get(key) == value, f"Runtime/provenance mismatch: {key}")
        require(p.get("fresh_tracker_per_video") is True, "Tracker policy mismatch")
        require(p.get("scope") == [m["source_relative_video_path"] for m in self.metadata], "Scope mismatch")
        require(pre.source_checksums(self.metadata, core.SOURCE_ROOT) == self.checksums,
                "Source checksum differs from preflight baseline")
        require(p.get("preflight", {}).get("checks") == "passed"
                and p["preflight"].get("clean_worktree") is True
                and p["preflight"].get("tests", {}).get("exit_code") == 0, "Preflight tests/clean-code evidence missing")
        require((self.run_dir / "tests_before.txt").read_text() == p["preflight"]["tests"]["output"],
                "Preflight test evidence mismatch")

    def paths(self, meta: dict[str, Any]) -> tuple[Path, Path]:
        relative = Path(meta["source_relative_video_path"])
        return canonical(self.output_root / relative.with_suffix(".npz")), canonical(
            self.run_dir / "results" / relative.with_suffix(".json"))

    def validate_pair(self, row: dict[str, str], meta: dict[str, Any], check_ledger: bool = True) -> dict[str, Any]:
        output, report = self.paths(meta)
        require(output.is_file() and report.is_file(), "Complete row requires both final artifacts")
        r = json.loads(report.read_text())
        identity = dict(meta, run_id=self.run_id, schema_version=core.SCHEMA_VERSION,
                        dataset_name="CAUCAFall", dataset_version="V5", status="complete",
                        source_sha256=row["source_sha256"], output_root=str(self.output_root),
                        output_relative_path=row["output_relative_path"], timestamp_method=core.TIMESTAMP_METHOD,
                        provenance_identity_sha256=self.provenance["identity_sha256"])
        for key, value in identity.items():
            require(r.get(key) == value, f"Result identity mismatch: {key}")
        require(r.get("provenance") == self.result_provenance(), "Result provenance mismatch")
        require(all(r.get(k) is None for k in ("error_type", "error_message", "error_frame_index")), "Complete result contains error")
        arrays = load_validated(output, meta["source_fps"], meta["expected_frame_count"])
        measured = metrics(arrays)
        for key in COUNTERS:
            require(r.get(key) == measured[key], f"Result/NPZ count mismatch: {key}")
            if check_ledger:
                require(row[key] == str(measured[key]), f"Manifest/NPZ count mismatch: {key}")
        require(r.get("decoded_frame_count") == meta["expected_frame_count"], "Decoded count mismatch")
        require(r.get("missing_frame_intervals") == measured["missing_frame_intervals"], "Missing intervals mismatch")
        require(r.get("arrays") == {k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in arrays.items()},
                "Result array metadata mismatch")
        return measured

    def result_provenance(self) -> dict[str, Any]:
        return {k: v for k, v in self.provenance.items() if k not in ("status", "execution")}

    def validate(self) -> None:
        self.metadata = pre.inventory_metadata()
        self.checksums = pre.read_csv(self.run_dir / "source_checksums.csv")
        require(len(self.metadata) == len(self.rows) == len(self.checksums) == 100, "Expected 100 manifest rows")
        require(all(set(c) == {"relative_video_path", "sha256"}
                    and isinstance(c["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", c["sha256"])
                    for c in self.checksums), "Invalid checksum manifest schema")
        initial = pre.manifest_rows(self.metadata, self.checksums, self.run_id)
        for row, baseline in zip(self.rows, initial):
            require(set(row) == set(pre.FIELDS), "Manifest fields mismatch")
            for key in set(pre.FIELDS) - MUTABLE:
                require(row[key] == str(baseline[key]), f"Manifest identity/order mismatch: {key}")
            require(row["status"] in STATES, "Unknown manifest status")
            if row["status"] == "pending":
                require(all(row[k] == "" for k in MUTABLE - {"status"}), "Pending row contains execution data")
            else:
                require(all(re.fullmatch(r"0|[1-9][0-9]*", row[k]) for k in COUNTERS), "Invalid counters")
                require(int(row["pose_detected_count"]) + int(row["pose_missing_count"]) == int(row["extracted_frame_count"])
                        <= int(row["expected_frame_count"]), "Counter invariant mismatch")
                if row["status"] == "complete":
                    require(all(not row[k] for k in ("error_type", "error_message", "error_frame_index")), "Complete row error")
                elif row["status"] == "failed":
                    require(row["error_type"] in ERRORS and bool(row["error_message"]), "Failed row error missing")
                if row["error_frame_index"]:
                    require(re.fullmatch(r"0|[1-9][0-9]*", row["error_frame_index"]) is not None, "Invalid error index")
        self.verify_identity()
        require(self.provenance.get("status") in ("pending", "running", "incomplete", "complete"), "Invalid run state")
        # Stale running work is unfinished/incomplete, never inferred complete.
        require(not any(r["status"] in ("running", "incomplete") for r in self.rows),
                "Unfinished/incomplete video requires review; no automatic restart or cleanup")
        if self.provenance["status"] == "complete":
            require(all(r["status"] == "complete" for r in self.rows), "Complete run has unfinished rows")
            execution = self.provenance.get("execution", {})
            tests = execution.get("tests_after", {})
            require(tests.get("exit_code") == 0 and execution.get("final_audits")
                    and all(execution["final_audits"].values()), "Complete run missing final audit")
            require((self.run_dir / "tests_after.txt").read_text() == tests.get("output"), "Final test log mismatch")
            summary = json.loads((self.run_dir / "summary.json").read_text())
            require(summary.get("run_id") == self.run_id and summary.get("status") == "complete"
                    and (self.run_dir / "FULL_EXTRACTION_REPORT.md").is_file(), "Complete run missing final reports")
        if self.provenance["status"] == "pending":
            require(all(r["status"] == "pending" for r in self.rows), "Pending run has started rows")
        allowed_npzs = set()
        for row, meta in zip(self.rows, self.metadata):
            output, report = self.paths(meta)
            allowed_npzs.add(output)
            if row["status"] == "complete":
                self.validate_pair(row, meta)
            else:
                require(not output.exists() and not report.exists(), "Unexpected artifact for non-complete row")
        for root, allowed in ((self.run_dir, {core.REPO / p for p in self.allowed_artifacts()}),
                              (self.output_root, allowed_npzs)):
            for path in root.rglob("*"):
                canonical(path)
                if path.is_file():
                    require(path in allowed, f"Unexpected artifact/temporary file: {path}")

    def persist_manifest(self) -> None:
        atomic_text(self.run_dir / "manifest.csv", pre.csv_text(self.rows, pre.FIELDS))

    def persist_run(self) -> None:
        write_json(self.run_dir / "extraction_run.json", self.provenance)

    def process(self, row: dict[str, str], meta: dict[str, Any], cv2: Any, mp: Any) -> None:
        # Recheck the code/config/model before each call; full source rehash is
        # done at entry/final audit, and this video's bytes are checked twice.
        self.verify_git()
        for path, digest in self.provenance["file_sha256"].items():
            require(core.sha256(canonical(core.REPO / path)) == digest, "Implementation/evidence changed during run")
        require(core.sha256(core.MODEL) == self.provenance["model_sha256"], "Model changed during run")
        source = canonical(core.SOURCE_ROOT / meta["source_relative_video_path"])
        require(core.sha256(source) == row["source_sha256"], "Source checksum changed before extraction")
        output, report = self.paths(meta)
        core.output_path(output.parent, Path(output.name))
        core.output_path(report.parent, Path(report.name))
        transition(row, "running")
        self.persist_manifest()
        result = dict(meta, schema_version=core.SCHEMA_VERSION, run_id=self.run_id,
                      dataset_name="CAUCAFall", dataset_version="V5", status="running",
                      decoded_frame_count=0, extracted_frame_count=0, pose_detected_count=0,
                      pose_missing_count=0, timestamp_method=core.TIMESTAMP_METHOD,
                      source_sha256=row["source_sha256"], output_root=str(self.output_root),
                      output_relative_path=row["output_relative_path"],
                      provenance_identity_sha256=self.provenance["identity_sha256"],
                      provenance=self.result_provenance())
        phase = "inference_runtime_failure"
        try:
            arrays = core.extract_arrays(source, core.MODEL, meta, result, cv2, mp)
            require(core.sha256(source) == row["source_sha256"] and
                    core.sha256(core.MODEL) == self.provenance["model_sha256"], "Source/model changed during extraction")
            # A mutation while inference ran is also a blocker before publication.
            for path, digest in self.provenance["file_sha256"].items():
                require(core.sha256(core.REPO / path) == digest, "Implementation/evidence changed during extraction")
            phase = "output_persistence_failure"
            result = core.publish_video_pair(output, report, arrays, meta, result)
            self.validate_pair(row, meta, check_ledger=False)
            transition(row, "complete", result)
        except KeyboardInterrupt:
            result.update(error_type="interrupted_run", error_message="Interrupted", error_frame_index=None)
            transition(row, "incomplete", result)
            raise
        except core.ExtractionError as exc:
            result.update(error_type=exc.error_type, error_message=str(exc), error_frame_index=exc.frame_index)
            transition(row, "failed", result)
            if exc.error_type == "provenance_mismatch":
                raise RunBlocked(str(exc)) from exc
        except Exception as exc:
            result.update(error_type="provenance_mismatch" if isinstance(exc, RunBlocked) else phase,
                          error_message=str(exc), error_frame_index=None)
            transition(row, "failed", result)
            raise  # Unexpected defects and validation/persistence errors stop the run.
        finally:
            self.persist_manifest()

    def final_report(self, audits: dict[str, Any]) -> bool:
        videos = []
        for row, meta in zip(self.rows, self.metadata):
            item = dict(row)
            if row["status"] == "complete":
                item.update(self.validate_pair(row, meta))
                item["pose_availability"] = item["pose_detected_count"] / meta["expected_frame_count"]
            videos.append(item)
        complete = len(videos) == 100 and all(v["status"] == "complete" for v in videos) and all(audits.values())
        validated = [v for v in videos if v["status"] == "complete"]
        def aggregate(items):
            stored = [v for v in items if v["status"] == "complete"]
            return dict(videos=len(items), statuses=dict(Counter(v["status"] for v in items)),
                        expected_frames=sum(int(v["expected_frame_count"]) for v in items),
                        validated_extracted_frames=sum(int(v["extracted_frame_count"]) for v in stored),
                        detected=sum(int(v["pose_detected_count"]) for v in stored),
                        missing=sum(int(v["pose_missing_count"]) for v in stored))
        totals = aggregate(videos)
        summary = dict(run_id=self.run_id, status="complete" if complete else "incomplete", totals=totals,
                       pose_availability=totals["detected"] / totals["expected_frames"] if complete else None,
                       audits=audits, videos=videos,
                       zero_pose_videos=[v["source_relative_video_path"] for v in validated if v["pose_detected_count"] == 0],
                       per_split={s: aggregate([v for v in videos if v["split"] == s]) for s in ("train", "validation", "test")},
                       per_activity={s: aggregate([v for v in videos if v["original_activity"] == s]) for s in sorted({v["original_activity"] for v in videos})})
        write_json(self.run_dir / "summary.json", summary)
        lines = ["# Stage 2.6 full extraction report", "", f"Run: `{self.run_id}`. Status: **{summary['status']}**.", "",
                 "Pose availability is diagnostic only, not classifier accuracy or robustness. No coverage threshold is applied.",
                 "Only validated complete pairs contribute to stored-frame totals; failed-prefix counters remain in the ledger.",
                 "All missing runs are reported without defining a long-run cutoff. Zero-pose videos are retained.",
                 "Subjects 6/7 use only the frozen mechanical extraction and validation path; no policy tuning or qualitative analysis.", "",
                 "## Totals and audits", "", "```json", json.dumps({k: v for k, v in summary.items() if k != "videos"}, indent=2), "```", "",
                 "## Per-video results", "", "| Source | Status | Expected | Extracted | Detected | Missing | Availability | Longest missing run | Error |",
                 "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for v in videos:
            cells = [v["source_relative_video_path"], v["status"], v["expected_frame_count"],
                     v["extracted_frame_count"], v["pose_detected_count"], v["pose_missing_count"],
                     v.get("pose_availability", "unavailable"), v.get("longest_missing_run", "unavailable"),
                     v["error_type"] + ": " + v["error_message"] if v["error_type"] else ""]
            lines.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in cells) + " |")
        atomic_text(self.run_dir / "FULL_EXTRACTION_REPORT.md", "\n".join(lines) + "\n")
        return complete

    def execute(self) -> int:
        # A small exclusive sentinel prevents concurrent writers. A native crash
        # leaves it in place: stop for review, never automatically delete a lock.
        lock = canonical(self.run_dir / "executor.lock")
        with lock.open("x") as handle:
            handle.write(f"pid={os.getpid()}\n")
        try:
            self.validate()  # Entire ledger, all complete pairs, identity before MP import.
            if self.provenance["status"] == "complete":
                return 0
            self.provenance["status"] = "running"
            self.persist_run()
            try:
                if any(r["status"] == "pending" for r in self.rows):
                    cv2, mp = runtime_modules()
                    for row, meta in zip(self.rows, self.metadata):
                        if row["status"] == "pending":
                            self.process(row, meta, cv2, mp)
                tests = pre.run_tests()
                atomic_text(self.run_dir / "tests_after.txt", tests["output"])
                self.verify_identity()
                audits = dict(source_checksums=True, implementation_hashes=True,
                              runtime_model_config=True, post_tests=tests["exit_code"] == 0,
                              held_out_policy=True)
                complete = self.final_report(audits)
                self.provenance["status"] = "complete" if complete else "incomplete"
                self.provenance["execution"] = dict(tests_after=tests, final_audits=audits,
                                                     final_git_status=pre.git("status", "--short"))
                self.persist_run()  # Run completion marker LAST, after reports/audits.
                return 0 if complete else 1
            except BaseException as exc:
                self.provenance["status"] = "incomplete"
                self.provenance["execution"] = dict(stop_reason=f"{type(exc).__name__}: {exc}")
                self.persist_run()
                raise
        finally:
            lock.unlink()


def runtime_modules() -> tuple[Any, Any]:
    import cv2
    import mediapipe as mp
    return cv2, mp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--execution-commit", required=True, help="Full explicitly reviewed execution commit SHA")
    args = parser.parse_args()
    try:
        return Executor(args.run_directory, args.execution_commit).execute()
    except (Exception, KeyboardInterrupt) as exc:
        parser.exit(1, f"Execution stopped for review: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
