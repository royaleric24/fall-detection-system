"""Stage 3.2c: frozen ten-sequence Train-only PNG pose feasibility pilot."""

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ml.datasets.extract_pose_video import MODEL
from ml.datasets.pose_raw import encode_pose
from ml.datasets.pose_runtime import MODEL_SHA256, MODEL_URL
from ml.preprocessing import contract
from ml.preprocessing import audit_annotation_timeline as timeline
from ml.preprocessing import audit_annotation_alignment as previous

REPO = contract.REPO
RAW = timeline.RAW
PARENT = "8a43017170b1a9dff66476f83f95878a6c0945f8"
VERSION = "stage32c_png_pose_feasibility_v1"
OUTPUT = REPO / "artifacts/preprocessing" / VERSION
SOURCE = timeline.OUTPUT
LOG = logging.getLogger(__name__)
REQUIRED = (
    "Subject.2/Fall backwards/FallBackwardsS2.avi",
    "Subject.8/Hop/HopS8.avi",
    "Subject.8/Pick up object/PickupobjectS8.avi",
    "Subject.9/Walk/WalkS9.avi",
    "Subject.4/Fall forward/FallForwardS4.avi",
    "Subject.4/Pick up object/PickupobjectS4.avi",
    "Subject.9/Fall sitting/FallSittingS9.avi",
    "Subject.8/Fall forward/FallForwardS8.avi",
)
EXPLICIT_ALIASES = {
    ("Subject.2/Fall backwards/FallBackwardsS2.avi", "cas200091 - copia.png", "cas200091.txt"),
    ("Subject.8/Hop/HopS8.avi", "sals800096.png", "sals800096a.txt"),
    ("Subject.8/Pick up object/PickupobjectS8.avi", "res800090.png", "res800090a.txt"),
    ("Subject.9/Walk/WalkS9.avi", "cams900140.png", "cams900140w.txt"),
}
POSE_KEYS = {"annotation_ordinal", "pose_detected", "landmarks"}


class ImageDecodeError(ValueError):
    """A corrupt image is an input-integrity blocker, not a missing pose."""


def digest(path: Path) -> str:
    return previous.file_digest(path)


def require_train(subject_id: int, split: str) -> None:
    if split != "train":
        raise contract.ContractError("Stage 3.2c is Train only")
    contract.require_subject_access(subject_id, split)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def rows_bytes(rows: list[dict[str, Any]]) -> bytes:
    return previous.csv_bytes(rows)


def check_source_artifacts() -> None:
    record = json.loads((SOURCE / "validation_record.json").read_text())
    for name, expected in record["artifact_sha256"].items():
        if digest(SOURCE / name) != expected:
            raise ValueError(f"Frozen Stage 3.2b artifact changed: {name}")
    summary = json.loads((SOURCE / "summary.json").read_text())
    if (summary["totals"]["sequences"] != 60
            or summary["totals"]["order_status_counts"] != {"verified": 60}
            or summary["totals"]["timebase_status_counts"] != {"unresolved": 60}):
        raise ValueError("Frozen annotation-native evidence changed")


def select_pilot(inventory: list[dict[str, str]], sources: tuple[contract.PoseSource, ...]) -> list[dict[str, Any]]:
    """Use reviewed metadata only; never inspect pixels or inference outputs."""
    identity = {s.source_video: s for s in sources}
    if len(identity) != 60 or {s.subject_id for s in sources} != set(contract.FROZEN_SUBJECTS["train"]):
        raise ValueError("Unexpected Train source scope")
    by_name = {r["source_video_identity"]: r for r in inventory}
    if len(by_name) != 60 or set(by_name) != set(identity):
        raise ValueError("Frozen inventory/source identities differ")
    for name, row in by_name.items():
        source = identity[name]
        require_train(source.subject_id, source.split)
        if (row["split"] != "train" or int(row["subject_id"]) != source.subject_id
                or row["activity"] != source.activity
                or row["annotation_order_status"] != "verified"
                or row["annotation_timebase_status"] != "unresolved"):
            raise ValueError(f"Unusable reviewed Train identity: {name}")
    if len(set(REQUIRED)) != 8 or not set(REQUIRED).issubset(by_name):
        raise ValueError("Required stress sequence is absent")
    def clean(row: dict[str, str]) -> bool:
        return (row["source_video_identity"] not in REQUIRED
                and int(row["explicit_sequence_alias_pairs"]) == 0
                and int(row["decoded_pixel_duplicate_groups"]) == 0
                and row["annotation_order_status"] == "verified")
    controls = []
    for label in ("fall", "non_fall"):
        eligible = sorted((r["source_video_identity"] for r in inventory
                           if clean(r) and identity[r["source_video_identity"]].video_label == label))
        if not eligible:
            raise ValueError(f"No clean {label} control")
        controls.append(eligible[0])
    names = list(REQUIRED) + controls
    if len(names) != len(set(names)) or len(names) != 10:
        raise ValueError("Pilot must contain exactly ten distinct sequences")
    selected = []
    for name in names:
        row = by_name[name]; source = identity[name]
        role = ("required_stress" if name in REQUIRED else
                "clean_fall_control" if source.video_label == "fall" else "clean_non_fall_control")
        selected.append(dict(subject_id=source.subject_id, activity=source.activity,
            source_video_identity=name, split="train", video_label=source.video_label,
            pilot_role=role, png_count=int(row["png_count"]), txt_count=int(row["txt_count"]),
            annotation_order_status="verified", annotation_timebase_status="unresolved",
            explicit_alias_pairs=int(row["explicit_sequence_alias_pairs"]),
            exact_duplicate_groups=int(row["decoded_pixel_duplicate_groups"]),
            pose_npz_path=(Path("pose_npz") / Path(name).with_suffix(".npz")).as_posix()))
    return selected


def select_repeatability(identity: list[dict[str, Any]], duplicates: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Freeze nofall, grounded and one exact duplicate pair before inference."""
    nofall = next((r for r in identity if r["annotation_class_id"] == 0), None)
    grounded = next((r for r in identity if r["annotation_class_id"] == 1), None)
    if nofall is None or grounded is None:
        raise ValueError("Repeatability subset needs both annotation states")
    selected = {(r["source_video_identity"], r["annotation_ordinal"]): r for r in (nofall, grounded)}
    groups = [r for r in duplicates if r["identity_kind"] == "decoded_pixel_sha256"
              and r["source_video_identity"] in {x["source_video_identity"] for x in identity}]
    if not groups:
        raise ValueError("Repeatability subset needs an exact duplicate group")
    group = sorted(groups, key=lambda r:(r["source_video_identity"], json.loads(r["ordinals"])[0]))[0]
    for ordinal in json.loads(group["ordinals"])[:2]:
        row = next(r for r in identity if r["source_video_identity"] == group["source_video_identity"]
                   and r["annotation_ordinal"] == ordinal)
        selected[(row["source_video_identity"], row["annotation_ordinal"])] = row
    return [dict(source_video_identity=r["source_video_identity"], annotation_ordinal=r["annotation_ordinal"],
                 png_filename=r["png_filename"], png_sha256=r["png_sha256"],
                 annotation_class_id=r["annotation_class_id"],
                 annotation_semantic_label=r["annotation_semantic_label"])
            for _, r in sorted(selected.items())]


def check_pairing(row: dict[str, Any]) -> None:
    alias = (row["source_video_identity"], row["png_filename"], row["annotation_filename"])
    if row["annotation_pairing_status"] == "verified_explicit_sequence_alias":
        if alias not in EXPLICIT_ALIASES:
            raise ValueError(f"Unknown alias: {alias}")
    elif row["annotation_pairing_status"] != "exact_stem" or Path(row["png_filename"]).stem != Path(row["annotation_filename"]).stem:
        raise ValueError(f"Unknown or mismatched annotation identity: {alias}")


def check_sequence_identity(rows: list[dict[str, Any]], count: int) -> None:
    if (len(rows) != count
            or [int(r["annotation_ordinal"]) for r in rows] != list(range(count))
            or len({r["png_filename"] for r in rows}) != count
            or len({r["annotation_filename"] for r in rows}) != count):
        raise ValueError("Pilot PNG/TXT/ordinal identity must be one-to-one and complete")


def freeze() -> None:
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() != PARENT:
        raise ValueError("Frozen Stage 3.2b parent changed")
    check_source_artifacts()
    sources = contract.select_sources(("train",), purpose="exploration")
    inventory = read_rows(SOURCE / "annotation_native_sequence_inventory.csv")
    selected = select_pilot(inventory, sources)
    names = {r["source_video_identity"] for r in selected}
    reviewed = [r for r in read_rows(SOURCE / "annotation_native_frames.csv")
                if r["source_video_identity"] in names]
    identity = []
    for row in reviewed:
        require_train(int(row["subject_id"]), row["split"])
        if (row["annotation_order_status"] != "verified"
                or row["annotation_timebase_status"] != "unresolved"
                or row["png_timestamp_ms"] or row["avi_frame_index"]):
            raise ValueError("Pilot requires ordinal-only reviewed identities")
        check_pairing(row)
        label = int(row["annotation_class_id"])
        if label not in (0, 1) or row["annotation_semantic_label"] != previous.SEMANTICS[label]:
            raise ValueError("Annotation semantics changed")
        identity.append(dict(subject_id=int(row["subject_id"]), activity=row["activity"],
            source_video_identity=row["source_video_identity"], split="train",
            annotation_ordinal=int(row["png_ordinal_index"]), png_filename=row["png_filename"],
            png_sha256=row["file_sha256"], pixel_sha256=row["pixel_sha256"],
            annotation_filename=row["annotation_filename"], annotation_sha256=row["annotation_sha256"],
            annotation_class_id=label, annotation_semantic_label=row["annotation_semantic_label"],
            annotation_pairing_status=row["annotation_pairing_status"]))
    by_seq = defaultdict(list)
    for row in identity:
        by_seq[row["source_video_identity"]].append(row)
    for item in selected:
        rows = by_seq[item["source_video_identity"]]
        if item["png_count"] != item["txt_count"]:
            raise ValueError("Pilot PNG/TXT count changed")
        check_sequence_identity(rows, item["png_count"])
    observed_aliases = {(r["source_video_identity"], r["png_filename"], r["annotation_filename"])
                        for r in identity if r["annotation_pairing_status"] == "verified_explicit_sequence_alias"}
    if observed_aliases != EXPLICIT_ALIASES:
        raise ValueError("Four reviewed aliases are not exactly preserved")
    duplicates = read_rows(SOURCE / "png_duplicate_audit.csv")
    repeatability = select_repeatability(identity, duplicates)
    output = contract.validate_output_path(OUTPUT)
    files = {
        "pilot_sample_manifest.csv": rows_bytes(selected),
        "annotation_identity_manifest.csv": rows_bytes(identity),
        "repeatability_subset.csv": rows_bytes(repeatability),
    }
    record = dict(stage="3.2c", status="sample_frozen_before_inference", parent_commit=PARENT,
        train_subjects=list(contract.FROZEN_SUBJECTS["train"]), selection_rule="Eight specified stress sequences in task order, then lexicographically first eligible fall and non_fall Train controls without alias or exact duplicate groups and with verified annotation order",
        selected_sequences=[r["source_video_identity"] for r in selected],
        selected_frame_count=len(identity), repeatability_subset_count=len(repeatability),
        source_hashes={name:digest(SOURCE/name) for name in ("annotation_native_sequence_inventory.csv",
            "annotation_native_frames.csv", "png_duplicate_audit.csv", "summary.json", "validation_record.json")},
        output_hashes={name:hashlib.sha256(content).hexdigest() for name,content in files.items()},
        extractor_code_sha256=digest(Path(__file__)), model_sha256=digest(MODEL))
    if record["model_sha256"] != MODEL_SHA256:
        raise ValueError("Stage 2 model asset changed")
    output = contract.validate_output_path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        with (output/name).open("xb") as handle:
            handle.write(content)
    with (output/"sample_freeze.json").open("x") as handle:
        json.dump(record, handle, indent=2); handle.write("\n")
    LOG.info("Frozen %d complete Train sequences and %d PNG identities before MediaPipe", len(selected), len(identity))


def validate_pilot_arrays(arrays: dict[str, np.ndarray], count: int) -> None:
    if set(arrays) != POSE_KEYS or count < 1:
        raise ValueError("Expected exact pilot raw-pose keys and positive frame count")
    ordinal, detected, landmarks = (arrays[k] for k in ("annotation_ordinal", "pose_detected", "landmarks"))
    if (ordinal.dtype != np.int32 or ordinal.shape != (count,)
            or not np.array_equal(ordinal, np.arange(count, dtype=np.int32))
            or detected.dtype != np.bool_ or detected.shape != (count,)
            or landmarks.dtype != np.float32 or landmarks.shape != (count, 33, 4)):
        raise ValueError("Pilot raw-pose shape, dtype or ordinal mismatch")
    if not np.isfinite(landmarks[detected]).all() or not np.isnan(landmarks[~detected]).all():
        raise ValueError("Detected poses must be finite; missing poses must be all NaN")


def load_pilot(path: Path, count: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != 3 or set(archive.files) != POSE_KEYS:
            raise ValueError("Pilot NPZ has unexpected fields")
        arrays = {name: archive[name] for name in archive.files}
    validate_pilot_arrays(arrays, count)
    return arrays


def write_pilot(path: Path, arrays: dict[str, np.ndarray]) -> None:
    count = len(arrays["annotation_ordinal"])
    validate_pilot_arrays(arrays, count)
    if path.exists() or path.is_symlink() or path.resolve() != path.absolute():
        raise ValueError(f"Refusing noncanonical/existing pilot NPZ: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pilot-", suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush(); os.fsync(handle.fileno())
        load_pilot(temporary, count)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def image_path(row: dict[str, Any]) -> Path:
    require_train(int(row["subject_id"]), row["split"])
    sequence = Path(row["source_video_identity"])
    if (len(sequence.parts) != 3 or sequence.parts[0] != f"Subject.{row['subject_id']}"
            or sequence.parts[1] != row["activity"] or sequence.suffix != ".avi"):
        raise ValueError("Noncanonical source sequence")
    name = row["png_filename"]
    if Path(name).name != name or not name.endswith(".png"):
        raise ValueError("Noncanonical PNG filename")
    path = RAW / sequence.parent / name
    return previous.checked_path(path, path.parent)


def annotation_path(row: dict[str, Any]) -> Path:
    require_train(int(row["subject_id"]), row["split"])
    name = row["annotation_filename"]
    if Path(name).name != name or not name.endswith(".txt"):
        raise ValueError("Noncanonical annotation filename")
    path = RAW / Path(row["source_video_identity"]).parent / name
    return previous.checked_path(path, path.parent)


def frozen_run_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() != PARENT:
        raise ValueError("Frozen parent changed")
    check_source_artifacts()
    freeze_record = json.loads((OUTPUT/"sample_freeze.json").read_text())
    if freeze_record["status"] != "sample_frozen_before_inference" or freeze_record["parent_commit"] != PARENT:
        raise ValueError("Missing valid pre-inference sample freeze")
    if digest(Path(__file__)) != freeze_record["extractor_code_sha256"] or digest(MODEL) != MODEL_SHA256:
        raise ValueError("Runtime or model changed after sample freeze")
    for name, expected in freeze_record["source_hashes"].items():
        if digest(SOURCE/name) != expected:
            raise ValueError("Frozen source evidence changed")
    for name, expected in freeze_record["output_hashes"].items():
        if digest(OUTPUT/name) != expected:
            raise ValueError("Pilot manifest changed after freeze")
    selected = read_rows(OUTPUT/"pilot_sample_manifest.csv")
    identity = read_rows(OUTPUT/"annotation_identity_manifest.csv")
    repeatability = read_rows(OUTPUT/"repeatability_subset.csv")
    sources = contract.select_sources(("train",), purpose="exploration")
    expected = select_pilot(read_rows(SOURCE/"annotation_native_sequence_inventory.csv"), sources)
    if [r["source_video_identity"] for r in selected] != [r["source_video_identity"] for r in expected]:
        raise ValueError("Pilot selection changed")
    if len(selected) != 10 or len(identity) != int(freeze_record["selected_frame_count"]):
        raise ValueError("Pilot count changed")
    for row in identity:
        require_train(int(row["subject_id"]), row["split"])
        png = image_path(row); txt = annotation_path(row)
        if digest(png) != row["png_sha256"] or digest(txt) != row["annotation_sha256"]:
            raise ValueError(f"Frozen image or annotation bytes changed: {png}")
        ann = previous.parse_annotation(txt.read_text(encoding="utf-8-sig"))
        if not ann["valid"] or ann["class_id"] != int(row["annotation_class_id"]):
            raise ValueError("Frozen annotation label changed")
        check_pairing(row)
    return selected, identity, repeatability, freeze_record


def image_options(mp: Any) -> Any:
    """Stage 2 Full float16/CPU/one-pose settings, with explicit stateless IMAGE mode."""
    return mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL),
                                         delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_poses=1,
        min_pose_detection_confidence=.5, min_pose_presence_confidence=.5,
        min_tracking_confidence=.5, output_segmentation_masks=False)


def infer_image(content: bytes, landmarker: Any, cv2_module: Any, mp: Any) -> tuple[bool, np.ndarray, tuple[int, int]]:
    """Independent PNG inference; deliberately has no annotation/class argument."""
    bgr = cv2_module.imdecode(np.frombuffer(content, dtype=np.uint8), cv2_module.IMREAD_COLOR)
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ImageDecodeError("PNG failed original-resolution BGR decoding")
    height, width = bgr.shape[:2]
    rgb = cv2_module.cvtColor(bgr, cv2_module.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    present, values = encode_pose(landmarker.detect(image))
    return present, values, (height, width)


def compare_observations(first: tuple[bool, np.ndarray], second: tuple[bool, np.ndarray]) -> dict[str, Any]:
    a, x = first; b, y = second
    structure_equal = a == b and np.array_equal(np.isfinite(x), np.isfinite(y))
    exact = structure_equal and np.array_equal(x, y, equal_nan=True)
    difference = float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64)))) if a and b else None
    return dict(first_pose_detected=bool(a), second_pose_detected=bool(b),
                finite_missing_structure_equal=bool(structure_equal), exact_equal=bool(exact),
                max_abs_landmark_difference=difference)


def publish_table(name: str, rows: list[dict[str, Any]]) -> None:
    with (OUTPUT/name).open("xb") as handle:
        handle.write(rows_bytes(rows))


def publish_json(name: str, content: dict[str, Any]) -> None:
    with (OUTPUT/name).open("x") as handle:
        json.dump(content, handle, indent=2, allow_nan=False); handle.write("\n")


def run() -> None:
    selected, identity, repeatability, freeze_record = frozen_run_inputs()
    # All authorized PNG/TXT bytes and identities have been checked before MediaPipe import.
    import mediapipe as mp
    from importlib.metadata import version as package_version

    by_sequence = defaultdict(list)
    for row in identity:
        by_sequence[row["source_video_identity"]].append(row)
    failures = []; dimensions = defaultdict(set); observations = {}; npz_hashes = {}
    with mp.tasks.vision.PoseLandmarker.create_from_options(image_options(mp)) as landmarker:
        for index, item in enumerate(selected, 1):
            name = item["source_video_identity"]; rows = by_sequence[name]
            count = int(item["png_count"])
            check_sequence_identity(rows, count)
            detected = np.zeros(count, dtype=np.bool_)
            landmarks = np.full((count, 33, 4), np.nan, dtype=np.float32)
            LOG.info("Pilot PNG sequence %d/10: %s (%d images)", index, name, count)
            for row in rows:
                ordinal = int(row["annotation_ordinal"])
                content = image_path(row).read_bytes()
                if hashlib.sha256(content).hexdigest() != row["png_sha256"]:
                    raise ValueError("PNG changed during pilot run")
                try:
                    present, values, shape = infer_image(content, landmarker, cv2, mp)
                except ImageDecodeError:
                    raise  # Decode/shape/source failure is a pilot blocker.
                except Exception as exc:
                    present, values = False, np.full((33, 4), np.nan, np.float32)
                    shape = None
                    failures.append(dict(source_video_identity=name, annotation_ordinal=ordinal,
                        png_filename=row["png_filename"], error_type=type(exc).__name__, error_message=str(exc)))
                detected[ordinal] = present
                landmarks[ordinal] = values
                if shape is not None:
                    dimensions[name].add(shape)
                observations[(name, ordinal)] = (bool(present), values.copy())
            arrays = dict(annotation_ordinal=np.arange(count, dtype=np.int32),
                          pose_detected=detected, landmarks=landmarks)
            path = OUTPUT/item["pose_npz_path"]
            write_pilot(path, arrays)
            npz_hashes[item["pose_npz_path"]] = digest(path)
    duplicate_rows = []
    frozen_groups = read_rows(SOURCE/"png_duplicate_audit.csv")
    for group in frozen_groups:
        if group["identity_kind"] != "decoded_pixel_sha256" or group["source_video_identity"] not in by_sequence:
            continue
        name = group["source_video_identity"]
        ordinals = json.loads(group["ordinals"])
        baseline = ordinals[0]
        for other in ordinals[1:]:
            left = by_sequence[name][baseline]; right = by_sequence[name][other]
            duplicate_rows.append(dict(source_video_identity=name, first_ordinal=baseline,
                second_ordinal=other, first_png_filename=left["png_filename"],
                second_png_filename=right["png_filename"],
                byte_identical=left["png_sha256"] == right["png_sha256"],
                decoded_pixel_identical=left["pixel_sha256"] == right["pixel_sha256"],
                **compare_observations(observations[(name,baseline)], observations[(name,other)])))
    repeat_rows = []
    for pass_number in (1, 2):
        with mp.tasks.vision.PoseLandmarker.create_from_options(image_options(mp)) as landmarker:
            for row in repeatability:
                name = row["source_video_identity"]; ordinal = int(row["annotation_ordinal"])
                source = by_sequence[name][ordinal]
                content = image_path(source).read_bytes()
                if hashlib.sha256(content).hexdigest() != row["png_sha256"]:
                    raise ValueError("Repeatability PNG identity changed")
                try:
                    present, values, _ = infer_image(content, landmarker, cv2, mp)
                except ImageDecodeError:
                    raise
                except Exception as exc:
                    present, values = False, np.full((33,4), np.nan, np.float32)
                    failures.append(dict(source_video_identity=name, annotation_ordinal=ordinal,
                        png_filename=row["png_filename"], error_type=type(exc).__name__,
                        error_message=str(exc), repeat_pass=pass_number))
                repeat_rows.append(dict(repeat_pass=pass_number, source_video_identity=name,
                    annotation_ordinal=ordinal, png_filename=row["png_filename"],
                    annotation_class_id=row["annotation_class_id"],
                    **compare_observations(observations[(name,ordinal)], (bool(present),values))))
    per_sequence = []; by_state = Counter(); by_state_detected = Counter()
    for item in selected:
        name = item["source_video_identity"]; rows = by_sequence[name]
        npz = load_pilot(OUTPUT/item["pose_npz_path"], len(rows))
        found = int(npz["pose_detected"].sum())
        per_sequence.append(dict(subject_id=int(item["subject_id"]), activity=item["activity"],
            source_video_identity=name, pilot_role=item["pilot_role"], png_frames=len(rows),
            pose_detected_frames=found, pose_missing_frames=len(rows)-found,
            pose_availability=found/len(rows), image_dimensions_hw=sorted(dimensions[name]),
            pose_npz_path=item["pose_npz_path"], pose_npz_sha256=npz_hashes[item["pose_npz_path"]]))
        for row in rows:
            label = row["annotation_semantic_label"]; ordinal=int(row["annotation_ordinal"])
            by_state[label]+=1; by_state_detected[label]+=bool(npz["pose_detected"][ordinal])
    states=[dict(annotation_class_id=class_id,annotation_semantic_label=label,
        png_frames=by_state[label], pose_detected_frames=by_state_detected[label],
        pose_missing_frames=by_state[label]-by_state_detected[label],
        pose_availability=by_state_detected[label]/by_state[label])
        for class_id,label in ((0,"nofall"),(1,"grounded_fall_state"))]
    if sum(r["png_frames"] for r in per_sequence)!=len(identity) or sum(r["png_frames"] for r in states)!=len(identity):
        raise ValueError("Pilot completeness failure")
    publish_table("per_sequence_pose_summary.csv", per_sequence)
    publish_table("pose_availability_by_annotation_state.csv", states)
    publish_table("duplicate_pose_diagnostics.csv", duplicate_rows)
    publish_table("repeatability_diagnostics.csv", repeat_rows)
    publish_table("inference_failures.csv", failures or [dict(status="none")])
    provenance=dict(stage="3.2c", parent_commit=PARENT, sample_freeze_sha256=digest(OUTPUT/"sample_freeze.json"),
        source_stage32b_summary_sha256=digest(SOURCE/"summary.json"),
        python=platform.python_version(), platform=platform.platform(),
        operating_system=platform.system(), architecture=platform.machine(),
        mediapipe_version=package_version("mediapipe"), numpy_version=np.__version__,
        opencv_package_version=package_version("opencv-contrib-python"), opencv_runtime_version=cv2.__version__,
        model_identity="PoseLandmarker Full float16 v1", model_url=MODEL_URL, model_sha256=digest(MODEL),
        runtime_mode="IMAGE", stateless_independent_images=True, physical_timestamp_required=False,
        delegate="CPU", num_poses=1, detection_confidence=.5, presence_confidence=.5,
        tracking_confidence_configured=.5, tracking_confidence_applicability="not used for IMAGE temporal tracking",
        output_segmentation_masks=False, image_decoder="OpenCV imdecode IMREAD_COLOR",
        decoded_channel_order="BGR", before_mediapipe="cv2.COLOR_BGR2RGB to SRGB mp.Image",
        image_dimensions_hw={r["source_video_identity"]:r["image_dimensions_hw"] for r in per_sequence},
        input_policy="Original resolution; no normalization, clipping, filtering, interpolation, smoothing, resampling or derived features",
        pilot_pose_schema={"annotation_ordinal":"int32[T]", "pose_detected":"bool[T]",
            "landmarks":"float32[T,33,4] x,y,z,visibility"},
        forbidden_npz_fields=["timestamp_ms","fps","annotation_class_id","label"],
        repeatability_passes=2, same_landmarker_configuration_all_passes=True)
    publish_json("runtime_provenance.json",provenance)
    total=len(identity); detected=sum(r["pose_detected_frames"] for r in per_sequence)
    s8=next(r for r in per_sequence if r["source_video_identity"]=="Subject.8/Fall forward/FallForwardS8.avi")
    summary=dict(status="pending_external_gate_review",stage="3.2c",parent_commit=PARENT,
        sample_frozen_before_inference=True, selected_sequences=len(selected), png_frames=total,
        pose_detected_frames=detected, pose_missing_frames=total-detected,
        pose_availability=detected/total, annotation_order_status="verified",
        annotation_timebase_status="unresolved", physical_png_fps=None,
        timestamps_fabricated=False, labels_in_raw_pose_npz=False,
        subject8_fall_forward=dict(png_frames=s8["png_frames"],
            png_pose_detected_frames=s8["pose_detected_frames"],
            stage2_avi_frames=122, stage2_avi_pose_detected_frames=0,
            comparison="source-level descriptive only; no AVI/PNG frame correspondence or cause inferred"),
        duplicate_comparisons=len(duplicate_rows), duplicate_exact_equal=sum(r["exact_equal"] for r in duplicate_rows),
        repeatability_comparisons=len(repeat_rows), repeatability_exact_equal=sum(r["exact_equal"] for r in repeat_rows),
        inference_failure_records=len(failures), downstream_decisions=contract.read_contract()["decisions"])
    publish_json("summary.json",summary)
    report=["# Stage 3.2c PNG-native pose feasibility pilot", "",
        "Pending external Gate Review; no PASS claim.", "",
        f"Frozen full Train sequences: {len(selected)}; PNGs: {total}; pose detected: {detected}; missing: {total-detected}; availability: {detected/total:.6f}.", "",
        "## Per-sequence availability", "", "| Sequence | PNGs | Detected | Missing | Availability |", "| --- | ---: | ---: | ---: | ---: |"]
    report += [f"| {r['source_video_identity']} | {r['png_frames']} | {r['pose_detected_frames']} | {r['pose_missing_frames']} | {r['pose_availability']:.6f} |" for r in per_sequence]
    report += ["", "## By original annotation state", "",
        "| State | PNGs | Detected | Missing | Availability |", "| --- | ---: | ---: | ---: | ---: |"]
    report += [f"| {r['annotation_semantic_label']} | {r['png_frames']} | {r['pose_detected_frames']} | {r['pose_missing_frames']} | {r['pose_availability']:.6f} |" for r in states]
    report += ["", f"Subject.8 / Fall forward: {s8['pose_detected_frames']}/{s8['png_frames']} PNG poses; frozen Stage 2 AVI had 0/122. This is a source-level diagnostic only, with no frame mapping or causal conclusion.", "",
        f"Exact duplicate diagnostic: {summary['duplicate_exact_equal']}/{len(duplicate_rows)} comparison pairs exactly equal. Repeatability diagnostic: {summary['repeatability_exact_equal']}/{len(repeat_rows)} exactly equal. See CSVs for per-pair state, structure and exact float32 differences; no tolerance is used.", "",
        f"Inference/encoding exception records: {len(failures)}. Such frames remain explicit all-NaN missing observations; no frame is skipped.", "",
        "The sample and repeatability subset were frozen before inference. This is a selected feasibility stress pilot, not an estimate for all Train videos or a classifier score.", "",
        "MediaPipe Full float16 v1, CPU, IMAGE mode processed original-resolution OpenCV BGR PNG decodes converted to RGB. IMAGE mode needs no timestamp. Annotation identity and labels are stored only in the separate manifest. Each NPZ has exactly ordinal, detection and raw landmark arrays. No physical PNG FPS or timestamp was created.", "",
        "No AVI/PNG frame alignment, preprocessing, windows or training occurred. Validation/Test contents were not accessed. All downstream decisions remain UNDECIDED.", ""]
    with (OUTPUT/"REPORT.md").open("x") as handle:
        handle.write("\n".join(report))
    LOG.info("Published Stage 3.2c pilot: %d/%d PNG poses, %d duplicate pairs, %d repeat comparisons",
             detected,total,len(duplicate_rows),len(repeat_rows))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("freeze","run"))
    args=parser.parse_args()
    if args.phase=="freeze": freeze()
    else: run()


if __name__=="__main__":
    logging.basicConfig(level=logging.INFO,format="%(levelname)s %(message)s")
    main()
