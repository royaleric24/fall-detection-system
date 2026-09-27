"""Pure Stage 3.2e annotation-record checks; no dataset or annotation I/O."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


PROTOCOL_PATH = Path(__file__).resolve().parents[2] / "configs/stage32e_manual_annotation_protocol.json"
TRAIN_SUBJECTS = frozenset({8, 4, 3, 9, 1, 2})


class AnnotationContractError(ValueError):
    """A proposed record violates the Stage 3.2e source-annotation contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AnnotationContractError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def _keys(value: Mapping[str, Any], required: set[str], optional: set[str], name: str) -> None:
    _require(required <= value.keys(), f"{name} missing {sorted(required - value.keys())}")
    _require(value.keys() <= required | optional, f"{name} has unknown fields {sorted(value.keys() - required - optional)}")


def _nonempty(value: Any, name: str) -> None:
    _require(isinstance(value, str) and bool(value.strip()), f"{name} must be nonempty text")


def _utc(value: Any, name: str) -> None:
    _nonempty(value, name)
    _require(value.endswith("Z"), f"{name} must use UTC Z notation")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AnnotationContractError(f"{name} must be an ISO 8601 UTC timestamp") from exc


def _flags(value: Any, vocabulary: set[str], name: str) -> None:
    _require(isinstance(value, list), f"{name} must be a list")
    _require(all(isinstance(flag, str) and flag in vocabulary for flag in value),
             f"{name} contains an unknown reason flag")
    _require(len(value) == len(set(value)), f"{name} contains duplicate flags")


def _protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def validate_record(record: Mapping[str, Any]) -> None:
    """Validate a proposed Train source-annotation record without changing it."""
    protocol = _protocol()
    record = _mapping(record, "record")
    schema = protocol["record_schema"]
    _keys(record, set(schema["required_top_level"]), set(schema["optional_top_level"]), "record")
    _nonempty(record["annotation_record_id"], "annotation_record_id")
    _require(record["protocol_version"] == schema["protocol_version"], "protocol_version mismatch")
    if "annotator_note" in record:
        _require(isinstance(record["annotator_note"], str), "annotator_note must be text")

    source = _mapping(record["source"], "source")
    _keys(source, set(schema["source_required"]), set(), "source")
    for name in ("source_video", "neutral_clip_id"):
        _nonempty(source[name], name)
    _require(source["split"] == "train" and type(source["subject_id"]) is int
             and source["subject_id"] in TRAIN_SUBJECTS,
             "Stage 3.2e protocol records require a frozen Train subject")
    digest = source["source_avi_sha256"]
    _require(digest is None or (isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None),
             "source_avi_sha256 must be null or a lowercase SHA-256 digest")
    frame_count = source["frame_count"]
    _require(type(frame_count) is int and frame_count > 0, "frame_count must be a positive integer")
    _require(type(source["source_fps"]) is int and source["source_fps"] == schema["source_fps"],
             "source_fps must be the frozen 20 FPS")
    _require(source["source_fps_provenance"] == schema["source_fps_provenance"],
             "source_fps_provenance mismatch")

    annotator = _mapping(record["annotator"], "annotator")
    _keys(annotator, set(schema["annotator_required"]), set(), "annotator")
    for name in schema["annotator_required"]:
        _nonempty(annotator[name], f"annotator.{name}")
    creation = _mapping(record["creation"], "creation")
    _keys(creation, set(schema["creation_required"]), set(), "creation")
    _utc(creation["created_at_utc"], "creation.created_at_utc")
    for name in ("annotation_run_id", "tool_version"):
        _nonempty(creation[name], f"creation.{name}")

    presence = record["event_presence"]
    _require(presence in protocol["event_presence"]["vocabulary"], "invalid event_presence")
    vocabulary = set(protocol["reason_flags"])
    _flags(record["event_presence_reason_flags"], vocabulary, "event_presence_reason_flags")
    if presence == "uncertain":
        _require(bool(record["event_presence_reason_flags"]), "uncertain event_presence requires a reason flag")

    boundaries = _mapping(record["boundaries"], "boundaries")
    names = protocol["boundaries"]["names"]
    _keys(boundaries, set(names), set(), "boundaries")
    preferred: dict[str, int] = {}
    coordinate_fields = set(protocol["boundaries"]["observed_fields"])
    statuses = set(protocol["boundaries"]["boundary_statuses"])
    for name in names:
        boundary = _mapping(boundaries[name], name)
        _keys(boundary, {"status", "reason_flags"}, coordinate_fields | {"preferred_timestamp_ms"}, name)
        status = boundary["status"]
        _require(isinstance(status, str) and status in statuses, f"{name} has invalid status")
        _flags(boundary["reason_flags"], vocabulary, f"{name}.reason_flags")
        if status == "observed":
            _require(coordinate_fields <= boundary.keys(), f"{name} observed requires earliest/preferred/latest frames")
            earliest, middle, latest = (boundary[field] for field in protocol["boundaries"]["observed_fields"])
            _require(all(type(index) is int and 0 <= index < frame_count for index in (earliest, middle, latest)),
                     f"{name} coordinates must be source AVI frame indices")
            _require(earliest <= middle <= latest, f"{name} earliest <= preferred <= latest required")
            preferred[name] = middle
            if "preferred_timestamp_ms" in boundary:
                _require(type(boundary["preferred_timestamp_ms"]) is int
                         and boundary["preferred_timestamp_ms"] == round(middle * 1000 / 20),
                         f"{name} preferred_timestamp_ms must be machine-derived")
        else:
            _require(not (coordinate_fields | {"preferred_timestamp_ms"}) & boundary.keys(),
                     f"{name} {status} cannot fabricate unknown coordinates")
            if status == "unjudgeable":
                _require(bool(boundary["reason_flags"]), f"{name} unjudgeable requires a reason flag")
            if status == "not_observed":
                _require(name == "recovery_start", "not_observed is only for optional recovery_start")

    if presence == "no_fall_observed":
        _require(all(boundaries[name]["status"] == "not_applicable" for name in names),
                 "no_fall_observed cannot have fall boundaries")
    elif presence == "uncertain":
        _require(all(boundaries[name]["status"] == "unjudgeable" for name in names[:2])
                 and boundaries["recovery_start"]["status"] == "not_applicable",
                 "uncertain event requires unjudgeable fall boundaries and inapplicable recovery")
    else:
        _require(all(boundaries[name]["status"] not in ("not_applicable", "not_observed")
                     for name in names[:2]),
                 "fall_observed requires applicable fall-transition and grounded boundaries")
        _require(boundaries["recovery_start"]["status"] != "not_applicable",
                 "fall_observed recovery must be observed, censored, unjudgeable, or not_observed")

    for earlier, later in zip(names, names[1:]):
        if earlier in preferred and later in preferred:
            _require(preferred[earlier] <= preferred[later],
                     f"preferred event order violated: {earlier} must precede {later}")


def validate_adjudication_link(link: Mapping[str, Any], original_record_ids: tuple[str, str]) -> None:
    """Check a separate adjudication reference; never edit original records."""
    link = _mapping(link, "adjudication link")
    _keys(link, {"adjudication_record_id", "protocol_version", "original_annotation_record_ids",
                 "adjudicator_pseudonymous_id", "created_at_utc"}, set(), "adjudication link")
    _nonempty(link["adjudication_record_id"], "adjudication_record_id")
    _require(link["protocol_version"] == _protocol()["record_schema"]["protocol_version"],
             "adjudication protocol_version mismatch")
    _require(len(original_record_ids) == 2 and len(set(original_record_ids)) == 2,
             "adjudication requires two distinct original records")
    _require(link["original_annotation_record_ids"] == list(original_record_ids),
             "adjudication must reference both immutable original records")
    _require(link["adjudication_record_id"] not in original_record_ids,
             "adjudication record must have a separate identity")
    _nonempty(link["adjudicator_pseudonymous_id"], "adjudicator_pseudonymous_id")
    _utc(link["created_at_utc"], "created_at_utc")
