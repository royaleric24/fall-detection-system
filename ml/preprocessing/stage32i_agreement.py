"""Frozen Stage 3.2i agreement analysis; ``preflight`` never computes agreement.

Pure metrics accept synthetic sample sizes. The genuine loader admits only the
pinned 12-clip run. JSON null / CSV NA mean unavailable, never numeric zero.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from ml.preprocessing.annotation_execution import (
    ANNOTATORS, PILOT_VERSION, REAL_ROOT, REPO, FinalizedBlob, PilotSpec,
    check_completion, collect_finalized, require_agreement_ready,
)
from ml.preprocessing.annotation_readiness import READINESS_FILES, RUNTIME_REQUIREMENTS, read_active
from ml.preprocessing.manual_annotation_contract import validate_record

SPEC_COMMIT = "013418a92494b8350cce8856e339c7ad3f44514a"
EXECUTION_COMMIT = "44d1543747f96fc8056adf2f71cdd6db3bdb07af"
SOURCE_RUNTIME_COMMIT = "725d6cd063c307fb50ddc2725ecfd1666c7f17b5"
GENUINE_RUN = "stage32h-6747d0191488418d932955fda609ec0e"
RAW_FREEZE_SHA256 = "750e46599836bafdf1644b48cd247a3a3ea16c3d99035f7c578682ea4d06fdb8"
SPEC_PATH = "docs/stage3/stage32i_analysis_spec.md"
EVENTS = ("fall_observed", "no_fall_observed", "uncertain")
BOUNDARIES = ("fall_transition_start", "grounded_start", "recovery_start")
STATUSES = ("observed", "left_censored", "right_censored", "not_observed", "unjudgeable", "not_applicable")
COORDINATES = ("earliest_plausible_frame", "preferred_frame", "latest_plausible_frame")
AVAILABILITY = ("both_observed", "same_nonobserved_status", "observed_vs_nonobserved", "different_nonobserved_status")
GAP = "inter_interval_blank_gap_frames"
FPS = 20
MS_PER_FRAME = 50
LIMITATIONS = (
    "This analysis is based on an independent dual-annotation pilot containing 12 paired clips. "
    "Agreement coefficients are descriptive and are not treated as precise population-level "
    "reliability estimates. Event presence, boundary status, preferred temporal position, and "
    "uncertainty intervals represent different measurement dimensions and are therefore reported "
    "separately. No adjudication or downstream ML supervision decisions were performed during Stage 3.2i."
)
TEMPORAL_FIELDS = (
    "preferred_signed_difference_frames", "preferred_absolute_difference_frames",
    "preferred_signed_difference_ms", "preferred_absolute_difference_ms", "preferred_exact_match",
    "A01_interval_width_frames", "A02_interval_width_frames", "A01_interval_width_ms", "A02_interval_width_ms",
    "intersection_width_frames", "union_width_frames", "interval_overlap_any", "interval_exact_match",
    "interval_iou", "overlap_coefficient", "A01_preferred_in_A02_interval", "A02_preferred_in_A01_interval",
    "mutual_preferred_containment", GAP,
)
RAW_BOUNDARY_FIELDS = tuple(f"{owner}_{part}" for owner in ANNOTATORS for part in ("earliest", "preferred", "latest"))
ROW_FIELDS = (
    "annotation_unit_id", "boundary_type", "A01_record_id", "A02_record_id", "A01_source_file", "A02_source_file",
    "A01_status", "A02_status", "status_exact_match", "availability_class", "temporal_comparable",
    "non_comparable_reason", "event_presence_A01", "event_presence_A02", "event_presence_match",
    *RAW_BOUNDARY_FIELDS, *TEMPORAL_FIELDS,
)
OUTPUT_ROOT = REPO / "artifacts/temporal_annotation/stage32i_agreement"


class AgreementError(ValueError):
    """Fail closed without repairing inputs or emitting their decisions."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AgreementError(message)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def confusion_matrix(pairs: Sequence[tuple[str, str]], categories: Sequence[str]) -> list[list[int]]:
    """Rows A01, columns A02; retain every category including empty cells."""
    order = {category: i for i, category in enumerate(categories)}
    _require(len(order) == len(categories) and bool(order), "Invalid category ordering")
    matrix = [[0 for _ in categories] for _ in categories]
    for a01, a02 in pairs:
        _require(a01 in order and a02 in order, "Unknown categorical value")
        matrix[order[a01]][order[a02]] += 1
    return matrix


def event_metrics(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Nominal three-category exact agreement, kappa and AC1 (fixed K=3)."""
    matrix = confusion_matrix(pairs, EVENTS)
    n = len(pairs)
    rows = [sum(row) for row in matrix]
    cols = [sum(matrix[i][j] for i in range(3)) for j in range(3)]
    exact = sum(matrix[k][k] for k in range(3))
    kappa = {"value": None, "reason": "zero_denominator", "chance_agreement": None}
    ac1 = dict(kappa)
    if n:
        observed = Fraction(exact, n)
        pe_k = sum(Fraction(a * b, n * n) for a, b in zip(rows, cols))
        pooled = [Fraction(a + b, 2 * n) for a, b in zip(rows, cols)]
        pe_ac = sum(p * (1 - p) for p in pooled) / 2
        for metric, pe, reason in ((kappa, pe_k, "undefined_due_to_degenerate_marginals"),
                                   (ac1, pe_ac, "undefined_chance_denominator")):
            metric.update(value=float((observed - pe) / (1 - pe)) if pe != 1 else None,
                          reason=None if pe != 1 else reason, chance_agreement=float(pe))
    return {
        "categories": list(EVENTS), "rows": "A01", "columns": "A02", "denominator": n,
        "confusion_matrix": matrix, "exact_agreement_count": exact, "exact_agreement_rate": _ratio(exact, n),
        "annotator_marginals": {
            owner: {category: {"count": counts[i], "rate": _ratio(counts[i], n), "denominator": n}
                    for i, category in enumerate(EVENTS)}
            for owner, counts in (("A01", rows), ("A02", cols))},
        "cohen_kappa": kappa, "gwet_ac1": {**ac1, "K": 3},
    }


def availability_class(a01: str, a02: str) -> str:
    """Partition status pairs without collapsing the underlying six statuses."""
    _require(a01 in STATUSES and a02 in STATUSES, "Unknown boundary status")
    if a01 == a02 == "observed":
        return "both_observed"
    if a01 == a02:
        return "same_nonobserved_status"
    return "observed_vs_nonobserved" if "observed" in (a01, a02) else "different_nonobserved_status"


def status_metrics(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """One boundary at a time, with no chance-corrected or pooled coefficient."""
    matrix = confusion_matrix(pairs, STATUSES)
    counts = {key: 0 for key in AVAILABILITY}
    for first, second in pairs:
        counts[availability_class(first, second)] += 1
    exact = sum(matrix[k][k] for k in range(6))
    return {"categories": list(STATUSES), "rows": "A01", "columns": "A02",
            "confusion_matrix": matrix, "denominator": len(pairs),
            "status_exact_agreement_count": exact, "status_exact_agreement_rate": _ratio(exact, len(pairs)),
            **counts}


def _boundary_triplet(boundary: Mapping[str, Any], frame_count: int | None) -> tuple[int, int, int] | None:
    _require(boundary.get("status") in STATUSES, "Unknown boundary status")
    if frame_count is not None:
        _require(type(frame_count) is int and frame_count > 0, "Invalid frozen frame count")
    if boundary["status"] != "observed":
        _require(not set((*COORDINATES, "preferred_timestamp_ms")) & boundary.keys(),
                 "Non-observed coordinates must be absent under the frozen schema")
        return None
    values = tuple(boundary.get(name) for name in COORDINATES)
    _require(all(type(x) is int and x >= 0 for x in values), "Observed coordinates require nonnegative integers")
    e, p, l = values
    _require(e <= p <= l, "Observed coordinates violate E <= P <= L")
    _require(frame_count is None or l < frame_count, "Observed coordinate exceeds frozen frame count")
    if "preferred_timestamp_ms" in boundary:
        _require(type(boundary["preferred_timestamp_ms"]) is int
                 and boundary["preferred_timestamp_ms"] == p * MS_PER_FRAME, "Invalid derived timestamp")
    return e, p, l


def temporal_metrics(a01: Mapping[str, Any], a02: Mapping[str, Any], *,
                     frame_count: int | None = None) -> dict[str, Any]:
    """Only observed/observed is comparable. No event-presence filtering."""
    first, second = (_boundary_triplet(b, frame_count) for b in (a01, a02))
    result = {name: None for name in TEMPORAL_FIELDS}
    comparable = first is not None and second is not None
    reason = None
    if not comparable:
        if first is None and second is not None:
            reason = "A01_NONOBSERVED_ONLY"
        elif second is None and first is not None:
            reason = "A02_NONOBSERVED_ONLY"
        else:
            reason = ("BOTH_NONOBSERVED_SAME_STATUS" if a01["status"] == a02["status"]
                      else "BOTH_NONOBSERVED_DIFFERENT_STATUS")
    result.update(temporal_comparable=comparable, non_comparable_reason=reason)
    if not comparable:
        return result
    e1, p1, l1 = first
    e2, p2, l2 = second
    w1, w2 = l1 - e1 + 1, l2 - e2 + 1
    intersection = max(0, min(l1, l2) - max(e1, e2) + 1)
    union = w1 + w2 - intersection
    difference = p2 - p1
    in_second, in_first = e2 <= p1 <= l2, e1 <= p2 <= l1
    result.update({
        "preferred_signed_difference_frames": difference, "preferred_absolute_difference_frames": abs(difference),
        "preferred_signed_difference_ms": difference * MS_PER_FRAME,
        "preferred_absolute_difference_ms": abs(difference) * MS_PER_FRAME,
        "preferred_exact_match": p1 == p2, "A01_interval_width_frames": w1, "A02_interval_width_frames": w2,
        "A01_interval_width_ms": w1 * MS_PER_FRAME, "A02_interval_width_ms": w2 * MS_PER_FRAME,
        "intersection_width_frames": intersection, "union_width_frames": union,
        "interval_overlap_any": intersection > 0, "interval_exact_match": (e1, l1) == (e2, l2),
        "interval_iou": intersection / union, "overlap_coefficient": intersection / min(w1, w2),
        "A01_preferred_in_A02_interval": in_second, "A02_preferred_in_A01_interval": in_first,
        "mutual_preferred_containment": in_second and in_first,
        GAP: max(e1, e2) - min(l1, l2) - 1 if intersection == 0 else None,
    })
    return result


def quantile_type7(values: Sequence[int | float], probability: float) -> float | None:
    """Hyndman-Fan Type 7, with NA for an empty sample (no NumPy dependency)."""
    _require(0 <= probability <= 1, "Quantile probability outside [0,1]")
    if not values:
        return None
    ordered = sorted(values)
    h = (len(ordered) - 1) * probability
    index = math.floor(h)
    fraction = h - index
    return ordered[index] + fraction * (ordered[min(index + 1, len(ordered) - 1)] - ordered[index])


def boundary_summaries(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Summarize only comparable rows of one boundary, with explicit denominators."""
    _require(len({row.get("boundary_type") for row in rows}) <= 1, "Boundary pooling prohibited")
    eligible = [row for row in rows if row["temporal_comparable"]]
    n = len(eligible)
    def values(key: str) -> list[Any]:
        data = [row[key] for row in eligible]
        _require(all(value is not None for value in data), "Missing comparable metric")
        return data
    def median(key: str) -> float | None:
        return quantile_type7(values(key), 0.5)
    def extreme(key: str, fn: Any) -> int | float | None:
        return fn(values(key)) if n else None
    exact = sum(row["preferred_exact_match"] for row in eligible)
    q1 = quantile_type7(values("preferred_absolute_difference_frames"), 0.25)
    q3 = quantile_type7(values("preferred_absolute_difference_frames"), 0.75)
    temporal = {
        "n_temporally_comparable": n, "preferred_exact_match_denominator": n,
        "preferred_exact_match_count": exact, "preferred_exact_match_rate": _ratio(exact, n),
        "median_signed_difference_frames": median("preferred_signed_difference_frames"),
        "median_signed_difference_ms": median("preferred_signed_difference_ms"),
        "median_absolute_difference_frames": median("preferred_absolute_difference_frames"),
        "median_absolute_difference_ms": median("preferred_absolute_difference_ms"),
        "Q1_absolute_difference_frames": q1, "Q3_absolute_difference_frames": q3,
        "IQR_absolute_difference_frames": q3 - q1 if n else None,
        "min_signed_difference_frames": extreme("preferred_signed_difference_frames", min),
        "max_signed_difference_frames": extreme("preferred_signed_difference_frames", max),
        "max_absolute_difference_frames": extreme("preferred_absolute_difference_frames", max),
    }
    intervals = {"n_temporally_comparable": n, "interval_denominator": n, "containment_denominator": n}
    for name, key in (("any_overlap", "interval_overlap_any"), ("interval_exact_match", "interval_exact_match"),
                      ("A01_preferred_in_A02_interval", "A01_preferred_in_A02_interval"),
                      ("A02_preferred_in_A01_interval", "A02_preferred_in_A01_interval"),
                      ("mutual_preferred_containment", "mutual_preferred_containment")):
        count = sum(row[key] for row in eligible)
        intervals.update({f"{name}_count": count, f"{name}_rate": _ratio(count, n)})
    for key in ("interval_iou", "overlap_coefficient"):
        intervals.update({f"median_{key}": median(key), f"min_{key}": extreme(key, min), f"max_{key}": extreme(key, max)})
    for owner in ANNOTATORS:
        intervals[f"{owner}_median_interval_width_frames"] = median(f"{owner}_interval_width_frames")
    gaps = [row[GAP] for row in eligible if not row["interval_overlap_any"]]
    _require(all(type(gap) is int and gap >= 0 for gap in gaps), "Missing disjoint blank gap")
    intervals.update(disjoint_interval_count=len(gaps), blank_gap_denominator=len(gaps))
    intervals[f"median_{GAP}_among_disjoint"] = quantile_type7(gaps, 0.5)
    intervals[f"max_{GAP}_among_disjoint"] = max(gaps) if gaps else None
    return temporal, intervals


@dataclass(frozen=True, repr=False)
class AnnotationPair:
    annotation_unit_id: str
    a01: Mapping[str, Any]
    a02: Mapping[str, Any]
    a01_path: str | None = None
    a02_path: str | None = None


def construct_pairs(spec: PilotSpec, files: Mapping[str, Sequence[FinalizedBlob]]) -> tuple[AnnotationPair, ...]:
    """Structural pairing only; never compare the two annotation decisions."""
    completion = check_completion(spec, files)
    _require(completion.complete, "Invalid finalized input: " + ",".join(completion.errors))
    owners = {}
    for owner in ANNOTATORS:
        owners[owner] = {}
        for blob in files[owner]:
            record = json.loads(blob.content)["record"]
            owners[owner][record["source"]["neutral_clip_id"]] = (record, blob.relative_path)
    return tuple(AnnotationPair(key, owners["A01"][key][0], owners["A02"][key][0],
                                owners["A01"][key][1], owners["A02"][key][1]) for key in sorted(spec.presentation_order))


def _validate_pairs(pairs: Sequence[AnnotationPair]) -> None:
    seen: set[str] = set()
    record_ids: set[str] = set()
    for pair in pairs:
        _require(pair.annotation_unit_id not in seen, "Duplicate canonical pair")
        seen.add(pair.annotation_unit_id)
        for owner, record in (("A01", pair.a01), ("A02", pair.a02)):
            validate_record(record)
            _require(record["source"]["neutral_clip_id"] == pair.annotation_unit_id, "Pair identity mismatch")
            _require(record["annotator"]["pseudonymous_id"] == owner, "Pair owner mismatch")
            _require(record["annotation_record_id"] not in record_ids, "Duplicate original record ID")
            record_ids.add(record["annotation_record_id"])
        _require(pair.a01["source"] == pair.a02["source"], "Pair source metadata mismatch")


def _boundary_row(pair: AnnotationPair, name: str) -> dict[str, Any]:
    first, second = pair.a01["boundaries"][name], pair.a02["boundaries"][name]
    row = {"annotation_unit_id": pair.annotation_unit_id, "boundary_type": name,
           "A01_record_id": pair.a01["annotation_record_id"], "A02_record_id": pair.a02["annotation_record_id"],
           "A01_source_file": pair.a01_path, "A02_source_file": pair.a02_path,
           "A01_status": first["status"], "A02_status": second["status"],
           "status_exact_match": first["status"] == second["status"],
           "availability_class": availability_class(first["status"], second["status"]),
           "event_presence_A01": pair.a01["event_presence"], "event_presence_A02": pair.a02["event_presence"],
           "event_presence_match": pair.a01["event_presence"] == pair.a02["event_presence"]}
    for owner, boundary in (("A01", first), ("A02", second)):
        for short, key in zip(("earliest", "preferred", "latest"), COORDINATES):
            row[f"{owner}_{short}"] = boundary.get(key)
    row.update(temporal_metrics(first, second, frame_count=pair.a01["source"]["frame_count"]))
    return row


def analyze_pairs(pairs: Sequence[AnnotationPair]) -> dict[str, Any]:
    """Pure computation. Call only with synthetic inputs until code is frozen."""
    _validate_pairs(pairs)
    ordered = sorted(pairs, key=lambda pair: pair.annotation_unit_id)
    rows = [_boundary_row(pair, name) for pair in ordered for name in BOUNDARIES]
    status, temporal, interval = {}, {}, {}
    for name in BOUNDARIES:
        selected = [row for row in rows if row["boundary_type"] == name]
        status[name] = status_metrics([(row["A01_status"], row["A02_status"]) for row in selected])
        temporal[name], interval[name] = boundary_summaries(selected)
    discrepancies = []
    for row in rows:
        flags = []
        for flag, condition in (
            ("EVENT_PRESENCE_MISMATCH", not row["event_presence_match"]),
            ("BOUNDARY_STATUS_MISMATCH", not row["status_exact_match"]),
            ("PREFERRED_FRAME_MISMATCH", row["temporal_comparable"] and not row["preferred_exact_match"]),
            ("INTERVAL_DISJOINT", row["temporal_comparable"] and not row["interval_overlap_any"]),
        ):
            if condition:
                flags.append(flag)
        if flags:
            discrepancies.append({**row, "discrepancy_flags": "|".join(flags)})
    paired = []
    for pair in ordered:
        entry = {"annotation_unit_id": pair.annotation_unit_id}
        for owner, record, path in (("A01", pair.a01, pair.a01_path), ("A02", pair.a02, pair.a02_path)):
            entry.update({f"{owner}_record_id": record["annotation_record_id"], f"{owner}_source_file": path,
                          f"{owner}_event_presence": record["event_presence"]})
            for boundary in BOUNDARIES:
                for key in ("status", *COORDINATES):
                    entry[f"{owner}_{boundary}_{key}"] = record["boundaries"][boundary].get(key)
        paired.append(entry)
    result = {"paired_annotations": paired, "boundary_rows": rows,
              "event": event_metrics([(p.a01["event_presence"], p.a02["event_presence"]) for p in ordered]),
              "status": status, "temporal": temporal, "interval": interval, "discrepancies": discrepancies}
    audit_denominators(result, len(pairs))
    return result


def audit_denominators(result: Mapping[str, Any], expected_n: int) -> None:
    """Fail closed on row, matrix, availability or temporal denominator drift."""
    _require(len(result["paired_annotations"]) == expected_n, "Pair row count mismatch")
    event = result["event"]
    _require(event["denominator"] == sum(map(sum, event["confusion_matrix"])) == expected_n, "Event denominator mismatch")
    _require(len(result["boundary_rows"]) == expected_n * 3, "Boundary row count mismatch")
    for name in BOUNDARIES:
        status, temporal, interval = (result[key][name] for key in ("status", "temporal", "interval"))
        rows = [row for row in result["boundary_rows"] if row["boundary_type"] == name]
        _require(len(rows) == status["denominator"] == sum(map(sum, status["confusion_matrix"]))
                 == sum(status[key] for key in AVAILABILITY) == expected_n, "Status denominator mismatch")
        n = sum(row["temporal_comparable"] for row in rows)
        _require(n == status["both_observed"] == temporal["n_temporally_comparable"]
                 == temporal["preferred_exact_match_denominator"] == interval["n_temporally_comparable"]
                 == interval["interval_denominator"] == interval["containment_denominator"], "Temporal denominator mismatch")
        disjoint = sum(row["temporal_comparable"] and row["interval_overlap_any"] is False for row in rows)
        _require(disjoint == interval["blank_gap_denominator"] == interval["disjoint_interval_count"], "Gap denominator mismatch")


@dataclass(frozen=True, repr=False)
class LoadedPilot:
    spec: PilotSpec
    files: Mapping[str, Sequence[FinalizedBlob]] = field(repr=False)
    validation: Mapping[str, Any]
    identities: Mapping[str, str]


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _no_symlinks(path: Path) -> None:
    _require(not any(part.is_symlink() for part in (path, *path.parents)), "Symlinked input/output path prohibited")


def _git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=REPO)


def _verify_frozen_sources() -> None:
    _require((REPO / SPEC_PATH).read_bytes() == _git("show", f"{SPEC_COMMIT}:{SPEC_PATH}"), "Frozen analysis specification changed")
    _require(_git("rev-parse", f"{SOURCE_RUNTIME_COMMIT}^").decode().strip() == EXECUTION_COMMIT,
             "Source runtime lineage mismatch")
    for relative in READINESS_FILES:
        _no_symlinks(REPO / relative)
        _require((REPO / relative).read_bytes() == _git("show", f"{SOURCE_RUNTIME_COMMIT}:{relative}"),
                 "Frozen readiness file changed")


def load_genuine() -> LoadedPilot:
    """Read-only fixed-root loader. No path discovery, video access or metrics.

    Historical launch checks require HEAD == runtime_commit. Analysis instead
    verifies the immutable historical files/binding and keeps its own code commit.
    """
    _no_symlinks(REAL_ROOT)
    control = REAL_ROOT / "_control"
    freeze_path = control / "raw_freezes/raw_annotation_freeze.json"
    active_path = control / "run_initialization/active_run.json"
    binding_path = control / "run_initialization/runtime_binding.json"
    for path in (freeze_path, active_path, binding_path):
        _no_symlinks(path)
    freeze_bytes = freeze_path.read_bytes()
    _require(_sha(freeze_bytes) == RAW_FREEZE_SHA256, "Raw-freeze manifest SHA-256 mismatch")
    _verify_frozen_sources()
    active = read_active()  # Includes unchanged Stage 3.2e/f/g metadata/code identities.
    active_bytes = active_path.read_bytes()
    _require(json.loads(active_bytes) == active, "Active run changed during preflight")
    binding_bytes = binding_path.read_bytes()
    binding = json.loads(binding_bytes)
    _require(active["run_id"] == GENUINE_RUN and active["record_annotation_run_id"] == PILOT_VERSION,
             "Genuine run identity mismatch")
    _require(binding.get("version") == "stage32h_runtime_binding_v1"
             and binding.get("run_id") == GENUINE_RUN and binding.get("parent_commit") == EXECUTION_COMMIT
             and binding.get("runtime_commit") == SOURCE_RUNTIME_COMMIT
             and binding.get("active_run_sha256") == _sha(active_bytes)
             and binding.get("runtime_requirements") == RUNTIME_REQUIREMENTS, "Historical runtime binding mismatch")
    spec = PilotSpec.from_frozen()
    for source in spec.private_by_id.values():
        _require(source["split"] == "train" and source["subject_id"] in {1, 2, 3, 4, 8, 9}
                 and source["source_fps"] == FPS, "Pilot must contain frozen Train-only source metadata")
    files = collect_finalized(REAL_ROOT)
    freeze = json.loads(freeze_bytes)
    require_agreement_ready(spec, files, freeze, code_commit=SOURCE_RUNTIME_COMMIT)
    completion = check_completion(spec, files)
    # Verify run membership, independently for each record. Do not pair decisions.
    for owner, blobs in files.items():
        for blob in blobs:
            record = json.loads(blob.content)["record"]
            _require(re.fullmatch(re.escape(f"{GENUINE_RUN}-{owner}-") + r"[0-9a-f]{32}",
                                  record["annotator"]["session_id"]) is not None, "Session/run identity mismatch")
    identities = {str(REAL_ROOT / path): digest for path, digest in completion.file_sha256.items()}
    identities.update({str(freeze_path): _sha(freeze_bytes), str(active_path): _sha(active_bytes), str(binding_path): _sha(binding_bytes)})
    validation = {"input_integrity": "PASS", "genuine_run_id": GENUINE_RUN,
                  "finalized_counts": dict(completion.counts), "original_record_count": 24,
                  "canonical_pair_count": 12, "pairing_source_field": "record.source.neutral_clip_id",
                  "unmatched_records": 0, "duplicate_annotation_units": 0,
                  "schema_validation": "PASS", "raw_freeze_sha256": RAW_FREEZE_SHA256,
                  "heldout_access": "NONE", "agreement_calculated_by_preflight": False}
    return LoadedPilot(spec, files, validation, identities)


def verify_analysis_commit(commit: str) -> None:
    """Require a committed, clean analysis checkout, distinct from source runtime."""
    _require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None
             and commit not in (SPEC_COMMIT, SOURCE_RUNTIME_COMMIT, EXECUTION_COMMIT), "A distinct analysis-code commit is required")
    _require(_git("rev-parse", "HEAD").decode().strip() == commit, "Analysis commit must equal HEAD")
    _require(not _git("status", "--porcelain", "--untracked-files=all"), "Analysis checkout must be clean")
    _require(subprocess.run(["git", "merge-base", "--is-ancestor", SPEC_COMMIT, commit], cwd=REPO).returncode == 0,
             "Analysis code must descend from specification freeze")
    for relative in ("ml/preprocessing/stage32i_agreement.py", "tests/test_stage32i_agreement.py"):
        _require(_git("show", f"{commit}:{relative}") == (REPO / relative).read_bytes(), "Analysis implementation/test bytes differ from commit")


def _utc_timestamp(value: str) -> None:
    _require(isinstance(value, str) and "T" in value and value.endswith("Z"),
             "Analysis timestamp must be ISO 8601 UTC with Z")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AgreementError("Invalid analysis timestamp") from exc
    _require(parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0,
             "Analysis timestamp must identify UTC")


def analysis_manifest(loaded: LoadedPilot, *, code_commit: str, run_id: str,
                      timestamp: str, output: Path) -> dict[str, Any]:
    """Record source runtime separately from the committed analysis implementation."""
    _utc_timestamp(timestamp)
    _require(re.fullmatch(r"[0-9a-f]{40}", code_commit) is not None
             and code_commit not in (SOURCE_RUNTIME_COMMIT, EXECUTION_COMMIT, SPEC_COMMIT),
             "Manifest requires a distinct analysis-code commit")
    _require(re.fullmatch(r"stage32i-[a-zA-Z0-9_-]+", run_id) is not None, "Invalid analysis run ID")
    return {
        "stage": "3.2i", "analysis_spec_version": SPEC_COMMIT, "analysis_spec_path": SPEC_PATH,
        "analysis_spec_sha256": _sha((REPO / SPEC_PATH).read_bytes()), "artifact_kind": "derived_analysis",
        "analysis_run_id": run_id, "analysis_timestamp": timestamp,
        "execution_protocol_commit": EXECUTION_COMMIT, "source_annotation_runtime_commit": SOURCE_RUNTIME_COMMIT,
        "raw_freeze_manifest_sha256": RAW_FREEZE_SHA256, "analysis_code_commit": code_commit,
        "source_genuine_run_id": GENUINE_RUN, "annotator_ids": list(ANNOTATORS),
        "expected_original_record_count": 24, "observed_original_record_count": loaded.validation["original_record_count"],
        "expected_pair_count": 12, "observed_pair_count": loaded.validation["canonical_pair_count"],
        "fps": FPS, "canonical_temporal_coordinate": "0-based source AVI frame_index",
        "analysis_input_paths": sorted(loaded.identities), "input_file_sha256": dict(loaded.identities),
        "analysis_output_paths": [], "output_directory": str(output),
        "python_version": sys.version.split()[0], "category_order": list(EVENTS),
        "boundary_order": list(BOUNDARIES), "status_order": list(STATUSES),
        "NA_encoding": {"json": "null", "csv": "NA", "booleans_csv": "true/false"},
        "row_order": "canonical annotation-unit identity, then frozen boundary order",
        "gate_type": "process_methodology", "completion_gate": "HOLD",
    }


def _json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _csv_new(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    def encode(value: Any) -> Any:
        if value is None:
            return "NA"
        if type(value) is bool:
            return "true" if value else "false"
        return value
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: encode(row[key]) for key in columns} for row in rows)


def _matrix_rows(metric: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"A01\\A02": category, **dict(zip(metric["categories"], row))}
            for category, row in zip(metric["categories"], metric["confusion_matrix"])]


def _paired_columns() -> list[str]:
    columns = ["annotation_unit_id"]
    for owner in ANNOTATORS:
        columns.extend(f"{owner}_{key}" for key in ("record_id", "source_file", "event_presence"))
        columns.extend(f"{owner}_{boundary}_{key}" for boundary in BOUNDARIES for key in ("status", *COORDINATES))
    return columns


def _md_table(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    def cell(value: Any) -> str:
        return ("NA" if value is None else str(value)).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |",
                      *("| " + " | ".join(cell(row[key]) for key in columns) + " |" for row in rows)])


def render_report(result: Mapping[str, Any], manifest: Mapping[str, Any], validation: Mapping[str, Any]) -> str:
    """Descriptive tables and provenance, with no quality/severity thresholds."""
    synthetic = manifest.get("artifact_kind") == "synthetic_test_only"
    heading = "SYNTHETIC TEST ONLY - NOT OFFICIAL RESULTS" if synthetic else "Stage 3.2i agreement analysis"
    event = result["event"]
    parts = [f"# {heading}", "## Provenance and integrity",
             "```json\n" + json.dumps({**manifest, "input_validation": validation}, sort_keys=True, indent=2) + "\n```",
             "## Event presence", f"Paired clips: {event['denominator']}. Rows A01, columns A02.",
             _md_table(("A01\\A02", *EVENTS), _matrix_rows(event)),
             _md_table(("metric", "value", "denominator"), [
                 {"metric": "exact_agreement_count", "value": event["exact_agreement_count"], "denominator": event["denominator"]},
                 {"metric": "exact_agreement_rate", "value": event["exact_agreement_rate"], "denominator": event["denominator"]},
                 *({"metric": key, "value": event[key]["value"], "denominator": event["denominator"]}
                   for key in ("cohen_kappa", "gwet_ac1"))]),
             "Secondary descriptive coefficients and NA reasons:\n```json\n" + json.dumps(
                 {key: event[key] for key in ("cohen_kappa", "gwet_ac1")}, sort_keys=True, indent=2) + "\n```",
             "Annotator marginal distributions:\n```json\n" + json.dumps(event["annotator_marginals"], sort_keys=True, indent=2) + "\n```"]
    for name in BOUNDARIES:
        parts.extend([f"## {name}", "Rows A01, columns A02.",
                      _md_table(("A01\\A02", *STATUSES), _matrix_rows(result["status"][name]))])
        for key in ("status", "temporal", "interval"):
            summary = {k: v for k, v in result[key][name].items() if k not in ("categories", "rows", "columns", "confusion_matrix")}
            parts.extend([f"### {key}", _md_table(("metric", "value"), [{"metric": k, "value": v} for k, v in summary.items()])])
    parts.extend(["## Objective discrepancy inventory",
                  "One row per flagged annotation unit and boundary. Event flags may repeat across boundaries; these are not independent samples.",
                  _md_table(("annotation_unit_id", "boundary_type", "discrepancy_flags"), result["discrepancies"]),
                  "Raw-derived values and objective flags are retained in discrepancy_inventory.csv. No correctness or severity is assigned.",
                  "## Limitations", "SYNTHETIC FIXTURE; the following required statement applies to the future genuine pilot:" if synthetic else "",
                  LIMITATIONS, "## Conclusion",
                  f"Stage 3.2i Analysis Execution: {manifest.get('completion_gate', 'HOLD')}",
                  f"Input integrity: {validation.get('input_integrity', 'HOLD')}",
                  f"Raw immutability after analysis: {validation.get('raw_integrity_after_analysis', 'NOT_YET_VERIFIED')}",
                  f"Paired clips: {event['denominator']} / {manifest['expected_pair_count']}",
                  "Event presence and each boundary: descriptive results above; no overall agreement scalar.",
                  "Adjudication: NOT PERFORMED\n\nOriginal-record modification: NONE\n\nValidation-subject access: NONE\n\n"
                  "Test Subject 6/7 access: NONE\n\nDownstream supervision design: NOT PERFORMED"])
    return "\n\n".join(parts) + "\n"


def write_artifacts(output: Path, result: Mapping[str, Any], manifest: dict[str, Any],
                    validation: dict[str, Any]) -> None:
    """Exclusive-create required outputs in a NEW directory; never overwrite.

    The runner supplies preflight metadata and post-verifies later. Until that
    succeeds both manifest and report explicitly remain HOLD.
    """
    _no_symlinks(output)
    _require(not output.exists(), "Output directory already exists")
    _require(manifest.get("artifact_kind") in ("derived_analysis", "synthetic_test_only"), "Output provenance missing")
    audit_denominators(result, manifest["expected_pair_count"])
    _require(result["event"]["denominator"] == manifest["observed_pair_count"], "Manifest pair count mismatch")
    # Never let the serializer write into the dataset or annotation namespaces.
    _require(not output.resolve().is_relative_to(REPO / "data"), "Data namespace is read-only")
    output.mkdir(parents=True, exist_ok=False)
    _json_new(output / "stage32i_analysis_manifest.json", manifest)
    _json_new(output / "stage32i_input_validation.json", validation)
    _csv_new(output / "paired_annotations.csv", _paired_columns(), result["paired_annotations"])
    _json_new(output / "event_presence_metrics.json", result["event"])
    _csv_new(output / "event_presence_confusion_matrix.csv", ("A01\\A02", *EVENTS), _matrix_rows(result["event"]))
    _csv_new(output / "boundary_status_pairs.csv", ROW_FIELDS, result["boundary_rows"])
    status_rows, temporal_rows, interval_rows = [], [], []
    for name in BOUNDARIES:
        metric = result["status"][name]
        _csv_new(output / f"boundary_status_confusion_{name}.csv", ("A01\\A02", *STATUSES), _matrix_rows(metric))
        status_rows.append({"boundary_type": name, **{k: v for k, v in metric.items()
                           if k not in ("categories", "rows", "columns", "confusion_matrix")}})
        temporal_rows.append({"boundary_type": name, **result["temporal"][name]})
        interval_rows.append({"boundary_type": name, **result["interval"][name]})
    _csv_new(output / "boundary_status_metrics.csv", tuple(status_rows[0]), status_rows)
    _csv_new(output / "boundary_temporal_comparable.csv", ROW_FIELDS,
             [row for row in result["boundary_rows"] if row["temporal_comparable"]])
    _csv_new(output / "boundary_temporal_summary.csv", tuple(temporal_rows[0]), temporal_rows)
    _csv_new(output / "boundary_interval_summary.csv", tuple(interval_rows[0]), interval_rows)
    _csv_new(output / "discrepancy_inventory.csv", (*ROW_FIELDS, "discrepancy_flags"), result["discrepancies"])
    with (output / "stage32i_report.md").open("x", encoding="utf-8") as stream:
        stream.write(render_report(result, manifest, validation))


def write_plots(output: Path, result: Mapping[str, Any], *, synthetic: bool = False) -> str:
    """Optional Section 32 PNG diagnostics using Matplotlib, no quality bands."""
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    prefix = "SYNTHETIC TEST ONLY: " if synthetic else ""
    def save(figure: Any, filename: str) -> None:
        _no_symlinks(output)
        path = output / filename
        _require(not path.exists(), "Plot already exists")
        FigureCanvasAgg(figure)
        figure.savefig(path, dpi=140, metadata={"Software": "Stage 3.2i"})
    for name, metric, filename in (("Event presence", result["event"], "event_presence_confusion_matrix.png"),
                                  *((name, result["status"][name], f"boundary_status_confusion_{name}.png") for name in BOUNDARIES)):
        figure = Figure(figsize=(9, 7), layout="constrained")
        ax = figure.subplots()
        ax.imshow(metric["confusion_matrix"], cmap="Blues", vmin=0)
        categories = metric["categories"]
        ax.set_xticks(range(len(categories)), categories, rotation=40, ha="right")
        ax.set_yticks(range(len(categories)), categories)
        ax.set(xlabel="A02", ylabel="A01", title=prefix + name)
        for i, row in enumerate(metric["confusion_matrix"]):
            for j, value in enumerate(row):
                # Text contrast only; this is not an agreement-quality threshold.
                color = "white" if value > max(map(max, metric["confusion_matrix"])) / 2 else "black"
                ax.text(j, i, str(value), ha="center", va="center", color=color)
        save(figure, filename)
    for name in BOUNDARIES:
        rows = [row for row in result["boundary_rows"] if row["boundary_type"] == name and row["temporal_comparable"]]
        if not rows:
            continue
        labels = [row["annotation_unit_id"] for row in rows]
        figure = Figure(figsize=(10, max(3, len(rows) * .45)), layout="constrained")
        ax = figure.subplots()
        ax.scatter([row["preferred_signed_difference_frames"] for row in rows], range(len(rows)))
        ax.axvline(0, color="black", linewidth=.6)
        ax.set_yticks(range(len(rows)), labels)
        ax.set(xlabel="A02 - A01 preferred frame (source AVI frames)", title=prefix + name)
        save(figure, f"preferred_frame_difference_{name}.png")
        figure = Figure(figsize=(10, max(3, len(rows) * .5)), layout="constrained")
        ax = figure.subplots()
        for i, row in enumerate(rows):
            for owner, offset, color in (("A01", -.12, "tab:blue"), ("A02", .12, "tab:orange")):
                ax.hlines(i + offset, row[f"{owner}_earliest"], row[f"{owner}_latest"], color=color)
                ax.scatter(row[f"{owner}_preferred"], i + offset, color=color, label=owner if i == 0 else None)
        ax.set_yticks(range(len(rows)), labels)
        ax.set(xlabel="0-based source AVI frame_index (inclusive E/L; dot = P)", title=prefix + name)
        ax.legend()
        save(figure, f"interval_agreement_{name}.png")
    return matplotlib.__version__


def _replace_derived_json(path: Path, value: Any) -> None:
    # Only the runner's newly created derived files; never used on source inputs.
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def run_genuine(*, code_commit: str, run_id: str, timestamp: str, plots: bool = False) -> Path:
    """FUTURE EXECUTION ONLY. No call to this function in implementation freeze."""
    verify_analysis_commit(code_commit)
    _utc_timestamp(timestamp)
    _require(re.fullmatch(r"stage32i-[a-zA-Z0-9_-]+", run_id) is not None, "Invalid analysis run ID")
    output = OUTPUT_ROOT / run_id
    _no_symlinks(output)
    _require(not output.exists(), "Analysis output already exists")
    if plots:
        import importlib.util
        _require(importlib.util.find_spec("matplotlib") is not None, "Requested plots require Matplotlib")
    loaded = load_genuine()
    result = analyze_pairs(construct_pairs(loaded.spec, loaded.files))
    audit_denominators(result, 12)
    manifest = analysis_manifest(loaded, code_commit=code_commit, run_id=run_id, timestamp=timestamp, output=output)
    validation = dict(loaded.validation, denominator_audit="PASS", raw_integrity_after_analysis="NOT_YET_VERIFIED")
    write_artifacts(output, result, manifest, validation)
    if plots:
        manifest["matplotlib_version"] = write_plots(output, result)
    manifest["plots_requested"] = plots
    manifest["analysis_output_paths"] = [str(path) for path in sorted(output.iterdir())]
    # Fresh reads AFTER generating derived tables/report/figures, not cached hashes.
    try:
        after = load_genuine()
        _require(after.identities == loaded.identities, "Source identity changed during analysis")
    except Exception:
        validation["raw_integrity_after_analysis"] = "FAIL"
        _replace_derived_json(output / "stage32i_input_validation.json", validation)
        (output / "stage32i_report.md").write_text(render_report(result, manifest, validation), encoding="utf-8")
        raise
    validation["raw_integrity_after_analysis"] = "PASS"
    manifest["completion_gate"] = "PASS"
    _replace_derived_json(output / "stage32i_input_validation.json", validation)
    _replace_derived_json(output / "stage32i_analysis_manifest.json", manifest)
    (output / "stage32i_report.md").write_text(render_report(result, manifest, validation), encoding="utf-8")
    # Certification rewrites are derived outputs too. Recheck sources after them.
    try:
        _require(load_genuine().identities == loaded.identities, "Source identity changed during final certification")
    except Exception:
        manifest["completion_gate"] = "HOLD"
        validation["raw_integrity_after_analysis"] = "FAIL"
        _replace_derived_json(output / "stage32i_analysis_manifest.json", manifest)
        _replace_derived_json(output / "stage32i_input_validation.json", validation)
        (output / "stage32i_report.md").write_text(render_report(result, manifest, validation), encoding="utf-8")
        raise
    return output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", help="Structural/hash checks only; no agreement computation")
    run = commands.add_parser("run", help="Genuine analysis; requires separate execution authorization")
    run.add_argument("--analysis-code-commit", required=True)
    run.add_argument("--analysis-run-id", required=True)
    run.add_argument("--analysis-timestamp", required=True)
    run.add_argument("--plots", action="store_true", help="Generate Section 32 figures using installed Matplotlib")
    args = parser.parse_args(argv)
    if args.command == "preflight":
        print(json.dumps(load_genuine().validation, sort_keys=True, indent=2))
    else:
        path = run_genuine(code_commit=args.analysis_code_commit, run_id=args.analysis_run_id,
                           timestamp=args.analysis_timestamp, plots=args.plots)
        print(json.dumps({"output_directory": str(path), "completion_gate": "PASS"}))


if __name__ == "__main__":
    main()
