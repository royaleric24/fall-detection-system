"""Stage 3.0 metadata-only access and build guards. No pose loading or transforms."""

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from ml.datasets.assign_dataset_split import assign_rows, subject_assignments

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/stage3_contract.json"
RUN_ID = "4df7dd5f-fb38-4908-8233-1a81deb8dc05"
EXECUTION_COMMIT = "4b7557a5b3a8a4b40906b57a01a264bdeef9edb6"
EVIDENCE_COMMIT = "a28e0289c53463b3676b6c9275d4ee18a840fdae"
FROZEN_SUBJECTS = {"train": (8, 4, 3, 9, 1, 2), "validation": (10, 5), "test": (6, 7)}
DECISIONS = (
    "missing_pose_policy", "visibility_threshold", "interpolation_policy",
    "maximum_interpolation_gap_ms", "normalization_denominator",
    "invalid_normalization_scale_policy", "temporal_resampling_method",
    "target_model_fps", "window_duration_ms", "window_stride_ms",
    "minimum_pose_coverage", "temporal_supervision_protocol",
    "positive_window_overlap_threshold", "ambiguous_window_policy",
    "optional_motion_features", "transform_order", "derived_schema_version",
)


class ContractError(ValueError):
    """Stop on invalid identity, forbidden access or an unresolved build contract."""


@dataclass(frozen=True)
class PoseSource:
    """Identity only; deliberately excludes availability/visibility diagnostics."""

    dataset_name: str
    dataset_version: str
    stage2_run_id: str
    source_video: str
    source_npz: Path
    subject_id: int
    activity: str
    split: str
    video_label: str  # Never a frame or window label.


def read_contract(path: Path = CONFIG) -> dict[str, Any]:
    """Read the Stage 3.0 specification, with no parameter defaults."""
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ContractError("Expected a contract object")
    if (type(config.get("contract_version")) is not int
            or config["contract_version"] != 1 or config.get("stage") != "3.0"):
        raise ContractError("Unsupported Stage 3 contract version")
    if (config.get("build_enabled") is not False
            or config.get("review_status") != "pending_external_gate_review"):
        raise ContractError("Stage 3.0 remains build-disabled and pending external Gate Review")
    source = config.get("source")
    expected = {
        "dataset_name": "CAUCAFall", "dataset_version": "V5", "schema_version": "pose_raw_v1",
        "run_id": RUN_ID, "execution_commit": EXECUTION_COMMIT, "evidence_commit": EVIDENCE_COMMIT,
        "pose_root": f"data/interim/caucafall_v5/pose_raw_v1_runs/{RUN_ID}",
        "evidence_root": f"artifacts/pose_extraction/runs/{RUN_ID}",
        "split_config": "configs/dataset_split.json",
    }
    if not isinstance(source, dict) or any(source.get(k) != v for k, v in expected.items()):
        raise ContractError("Official Stage 2 source identity mismatch")
    for key in ("run_sha256", "manifest_sha256", "split_sha256"):
        value = source.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ContractError(f"Missing/invalid source checksum: {key}")
    decisions = config.get("decisions")
    if not isinstance(decisions, dict) or set(decisions) != set(DECISIONS):
        raise ContractError("Missing or unknown preprocessing decisions; no defaults are allowed")
    return config


def require_build_ready(config: dict[str, Any]) -> None:
    """Always block construction in 3.0; report unresolved choices before the stage lock."""
    decisions = config.get("decisions", {})
    if not isinstance(decisions, dict):
        raise ContractError("Missing preprocessing decisions; no defaults are allowed")
    unresolved = [name for name in DECISIONS
                  if decisions.get(name) is None or decisions.get(name) == "UNDECIDED"
                  or decisions.get(name) == ""]
    if unresolved:
        raise ContractError("UNDECIDED preprocessing decisions: " + ", ".join(unresolved))
    raise ContractError("Stage 3.0 has no authorized builder or window supervision; "
                        "a later reviewed contract and implementation are required")


def require_subject_access(subject_id: int, split: str, *, purpose: str = "exploration") -> None:
    """Check actual subject membership, not only a caller-supplied split label."""
    if purpose not in ("exploration", "fit_statistics"):
        raise ContractError("Unsupported access mode; final/frozen application is not implemented")
    if type(subject_id) is not int or subject_id not in range(1, 11):
        raise ContractError("Unknown subject")
    actual = next(name for name, subjects in FROZEN_SUBJECTS.items() if subject_id in subjects)
    if split != actual:
        raise ContractError("Subject/split mismatch")
    if actual == "test":
        raise ContractError("Held-out Test subjects are forbidden for exploration and fitting")
    if purpose == "fit_statistics" and actual != "train":
        raise ContractError("Learned statistics must be fitted on Train only")


def _checked_bytes(path: Path, digest: str) -> bytes:
    if path.is_symlink() or path.resolve() != path.absolute():
        raise ContractError(f"Noncanonical metadata path: {path}")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ContractError(f"Frozen metadata checksum mismatch: {path}")
    return content


def select_sources(splits: Iterable[str] = ("train",), *, purpose: str = "exploration",
                   repo: Path = REPO, config_path: Path = CONFIG) -> tuple[PoseSource, ...]:
    """Select official identities without opening NPZs, videos, summaries or result JSONs.

    Future analysis must use this entry point before loading poses. Validation
    must be explicitly requested; Test and unimplemented future modes fail before I/O.
    """
    requested = tuple(splits)
    if not requested or len(set(requested)) != len(requested):
        raise ContractError("Request a nonempty set of unique splits")
    for split in requested:
        if split not in FROZEN_SUBJECTS:
            raise ContractError(f"Unknown split: {split}")
        for subject in FROZEN_SUBJECTS[split]:
            require_subject_access(subject, split, purpose=purpose)
    config = read_contract(config_path)
    source = config["source"]
    split_config = json.loads(_checked_bytes(repo / source["split_config"], source["split_sha256"]))
    subject_assignments(split_config)
    if any(split_config[name] != list(ids) for name, ids in FROZEN_SUBJECTS.items()):
        raise ContractError("Frozen subject assignment changed")
    evidence = repo / source["evidence_root"]
    run = json.loads(_checked_bytes(evidence / "extraction_run.json", source["run_sha256"]))
    if (run.get("run_id") != RUN_ID or run.get("git_commit") != EXECUTION_COMMIT
            or run.get("status") != "complete" or run.get("schema_version") != "pose_raw_v1"):
        raise ContractError("Official Stage 2 run is not the expected complete source")
    # Validate canonical identities against the original inventory, not pose diagnostics.
    inventory_path = repo / "artifacts/dataset_inspection/inventory.csv"
    inventory_text = _checked_bytes(inventory_path, run["file_sha256"][
        "artifacts/dataset_inspection/inventory.csv"]).decode("utf-8")
    inventory = assign_rows(list(csv.DictReader(inventory_text.splitlines())), split_config)
    manifest_text = _checked_bytes(evidence / "manifest.csv", source["manifest_sha256"]).decode("utf-8")
    manifest = list(csv.DictReader(manifest_text.splitlines()))
    if len(manifest) != len(inventory):
        raise ContractError("Stage 2 manifest scope mismatch")
    selected = []
    for row, original in zip(manifest, inventory):
        subject = int(original["subject"].split(".")[1])
        relative_video = original["relative_video_path"]
        relative_npz = PurePosixPath(relative_video).with_suffix(".npz").as_posix()
        expected = dict(run_id=RUN_ID, schema_version="pose_raw_v1", dataset_name="CAUCAFall",
                        dataset_version="V5", subject_id=str(subject), split=original["split"],
                        original_activity=original["activity"], binary_label=original["label"],
                        source_relative_video_path=relative_video, output_relative_path=relative_npz,
                        status="complete")
        if any(row.get(k) != v for k, v in expected.items()):
            raise ContractError("Stage 2 manifest identity/split mismatch")
        # Do not expose or interpret any per-video missingness/visibility counters.
        if row["split"] not in requested:
            continue
        require_subject_access(subject, row["split"], purpose=purpose)
        npz = repo / source["pose_root"] / relative_npz
        if npz.resolve() != npz.absolute() or not npz.is_file():
            raise ContractError(f"Missing or redirected source NPZ: {npz}")
        selected.append(PoseSource("CAUCAFall", "V5", RUN_ID, relative_video, npz,
                                   subject, row["original_activity"], row["split"], row["binary_label"]))
    return tuple(selected)


def validate_output_path(path: Path, *, repo: Path = REPO) -> Path:
    """Validate a future destination without creating it; refuse overwrite and aliases."""
    repo = repo.resolve()
    candidate = path if path.is_absolute() else repo / path
    resolved = candidate.resolve()
    protected = (repo / "data/raw", repo / "data/interim", repo / "artifacts/pose_extraction")
    if any(resolved == root or resolved.is_relative_to(root) or root.is_relative_to(resolved)
           for root in protected):
        raise ContractError("Output overlaps immutable Stage 2 inputs/evidence")
    allowed = (repo / "data/processed/caucafall_v5", repo / "artifacts/preprocessing")
    if not any(resolved.is_relative_to(root) and resolved != root for root in allowed):
        raise ContractError("Output must be inside a separate Stage 3 data/evidence namespace")
    if candidate.absolute() != resolved or candidate.is_symlink():
        raise ContractError("Output path must not traverse aliases or symlinks")
    if candidate.exists():
        raise ContractError("Refusing existing Stage 3 output")
    return resolved
