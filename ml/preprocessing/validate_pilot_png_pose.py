"""Validate saved Stage 3.2c pilot evidence without rerunning MediaPipe."""

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ml.preprocessing import pilot_png_pose as pilot
from ml.preprocessing import contract


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate() -> dict:
    root = pilot.REPO
    output = pilot.OUTPUT
    freeze = json.loads((output / "sample_freeze.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    pilot.check_source_artifacts()
    assert freeze["status"] == "sample_frozen_before_inference"
    assert freeze["parent_commit"] == summary["parent_commit"] == pilot.PARENT
    assert sha(Path(pilot.__file__)) == freeze["extractor_code_sha256"]
    assert sha(pilot.MODEL) == freeze["model_sha256"] == pilot.MODEL_SHA256
    assert all(sha(pilot.SOURCE / name) == h for name, h in freeze["source_hashes"].items())
    assert all(sha(output / name) == h for name, h in freeze["output_hashes"].items())
    selected = pilot.read_rows(output / "pilot_sample_manifest.csv")
    identity = pilot.read_rows(output / "annotation_identity_manifest.csv")
    repeatability_subset = pilot.read_rows(output / "repeatability_subset.csv")
    sequence_results = pilot.read_rows(output / "per_sequence_pose_summary.csv")
    states = pilot.read_rows(output / "pose_availability_by_annotation_state.csv")
    duplicate_rows = pilot.read_rows(output / "duplicate_pose_diagnostics.csv")
    repeat_rows = pilot.read_rows(output / "repeatability_diagnostics.csv")
    provenance = json.loads((output / "runtime_provenance.json").read_text())
    assert len(selected) == len(sequence_results) == 10
    assert len(identity) == 1788 == freeze["selected_frame_count"] == summary["png_frames"]
    assert [r["source_video_identity"] for r in selected] == freeze["selected_sequences"]
    assert {int(r["subject_id"]) for r in selected} <= set(contract.FROZEN_SUBJECTS["train"])
    assert all(r["split"] == "train" for r in selected + identity)
    assert all(r["annotation_timebase_status"] == "unresolved" for r in selected)
    assert summary["annotation_timebase_status"] == "unresolved" and summary["physical_png_fps"] is None
    assert summary["timestamps_fabricated"] is False and summary["labels_in_raw_pose_npz"] is False
    assert provenance["runtime_mode"] == "IMAGE" and provenance["physical_timestamp_required"] is False
    assert provenance["model_sha256"] == pilot.MODEL_SHA256
    by_seq = defaultdict(list)
    for r in identity:
        pilot.require_train(int(r["subject_id"]), r["split"])
        pilot.check_pairing(r)
        by_seq[r["source_video_identity"]].append(r)
    expected_aliases = {(r["source_video_identity"], r["png_filename"], r["annotation_filename"])
                        for r in identity if r["annotation_pairing_status"] == "verified_explicit_sequence_alias"}
    assert expected_aliases == pilot.EXPLICIT_ALIASES
    found_npz = {p.relative_to(output).as_posix() for p in (output / "pose_npz").rglob("*.npz")}
    assert found_npz == {r["pose_npz_path"] for r in selected}
    results = {r["source_video_identity"]: r for r in sequence_results}
    total_detected = 0
    by_state = Counter()
    by_state_found = Counter()
    poses = {}
    for sequence in selected:
        name = sequence["source_video_identity"]
        rows = by_seq[name]
        count = int(sequence["png_count"])
        pilot.check_sequence_identity(rows, count)
        assert count == int(sequence["txt_count"])
        result = results[name]
        path = output / sequence["pose_npz_path"]
        assert sha(path) == result["pose_npz_sha256"]
        arrays = pilot.load_pilot(path, count)
        detected = int(arrays["pose_detected"].sum())
        assert detected == int(result["pose_detected_frames"])
        assert count - detected == int(result["pose_missing_frames"])
        assert detected / count == float(result["pose_availability"])
        total_detected += detected
        for r in rows:
            ordinal = int(r["annotation_ordinal"])
            key = (name, ordinal)
            assert key not in poses
            poses[key] = (bool(arrays["pose_detected"][ordinal]), arrays["landmarks"][ordinal])
            state = r["annotation_semantic_label"]
            by_state[state] += 1
            by_state_found[state] += bool(arrays["pose_detected"][ordinal])
    assert len(poses) == len(identity) == 1788
    assert total_detected == summary["pose_detected_frames"] == 1151
    assert len(identity) - total_detected == summary["pose_missing_frames"] == 637
    assert total_detected / len(identity) == summary["pose_availability"]
    assert {r["annotation_semantic_label"] for r in states} == {"nofall", "grounded_fall_state"}
    for r in states:
        state = r["annotation_semantic_label"]
        assert int(r["png_frames"]) == by_state[state]
        assert int(r["pose_detected_frames"]) == by_state_found[state]
        assert int(r["pose_missing_frames"]) == by_state[state] - by_state_found[state]
    frozen_groups = pilot.read_rows(pilot.SOURCE / "png_duplicate_audit.csv")
    selected_groups = [r for r in frozen_groups if r["identity_kind"] == "decoded_pixel_sha256"
                       and r["source_video_identity"] in by_seq]
    expected_pairs = {(r["source_video_identity"], json.loads(r["ordinals"])[0], other)
                      for r in selected_groups for other in json.loads(r["ordinals"])[1:]}
    actual_pairs = {(r["source_video_identity"], int(r["first_ordinal"]), int(r["second_ordinal"]))
                    for r in duplicate_rows}
    assert actual_pairs == expected_pairs and len(actual_pairs) == 49
    for r in duplicate_rows:
        first = poses[(r["source_video_identity"], int(r["first_ordinal"]))]
        second = poses[(r["source_video_identity"], int(r["second_ordinal"]))]
        comparison = pilot.compare_observations(first, second)
        assert (r["exact_equal"] == "True") == comparison["exact_equal"]
        assert (r["finite_missing_structure_equal"] == "True") == comparison["finite_missing_structure_equal"]
    assert len(repeatability_subset) == 4 and len(repeat_rows) == 8
    assert {(r["source_video_identity"], int(r["annotation_ordinal"])) for r in repeat_rows} == {
        (r["source_video_identity"], int(r["annotation_ordinal"])) for r in repeatability_subset}
    assert Counter(int(r["repeat_pass"]) for r in repeat_rows) == {1:4,2:4}
    assert sum(r["exact_equal"] == "True" for r in duplicate_rows) == summary["duplicate_exact_equal"]
    assert sum(r["exact_equal"] == "True" for r in repeat_rows) == summary["repeatability_exact_equal"]
    s8 = results["Subject.8/Fall forward/FallForwardS8.avi"]
    assert (int(s8["png_frames"]), int(s8["pose_detected_frames"])) == (123, 0)
    assert all(x == "UNDECIDED" for x in contract.read_contract()["decisions"].values())
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() == pilot.PARENT
    assert not subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=root, text=True)
    return dict(status="mechanical_validation_complete", parent_commit=pilot.PARENT,
        selected_sequences=len(selected), identity_rows=len(identity), pilot_npz_files=len(found_npz),
        pose_detected_frames=total_detected, pose_missing_frames=len(identity)-total_detected,
        exact_duplicate_comparisons=len(duplicate_rows), repeatability_comparisons=len(repeat_rows),
        frozen_stage32b_artifact_hashes_verified=len(json.loads((pilot.SOURCE/"validation_record.json").read_text())["artifact_sha256"]),
        every_pose_has_one_frozen_png_txt_ordinal=True, no_timestamp_fps_or_label_in_npz=True,
        all_downstream_decisions_undecided=True)


def main() -> None:
    record = validate()
    path = pilot.OUTPUT / "validation_record.json"
    with path.open("x") as handle:
        json.dump(record, handle, indent=2); handle.write("\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
