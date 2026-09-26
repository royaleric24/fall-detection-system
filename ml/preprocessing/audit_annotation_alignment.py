"""Stage 3.2a: Train annotation/AVI correspondence evidence, never pose labels."""

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import platform
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ml.preprocessing import contract

REPO = contract.REPO
RAW = REPO / "data/raw/caucafall_v5/CAUCAFall"
BASELINE = "46f4e299e943db24f386787e30ab05f484949262"
VERSION = "stage32a_train_annotation_alignment_v1"
OUTPUT = REPO / "artifacts/preprocessing" / VERSION
LOCAL_RADIUS = 3  # Diagnostic search extent, never a similarity acceptance threshold.
SEMANTICS = {0: "nofall", 1: "grounded_fall_state"}
SOURCE_DOCUMENTATION = "https://pmc.ncbi.nlm.nih.gov/articles/PMC9508401/"
LOG = logging.getLogger(__name__)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require_train(split: str) -> None:
    if split != "train":
        raise contract.ContractError("Stage 3.2a is Train only")


def checked_path(path: Path, directory: Path) -> Path:
    """Reject redirection before reading bytes, including redirected parent paths."""
    if path.absolute() != path.resolve() or path.parent != directory or not path.is_file():
        raise ValueError(f"Noncanonical or absent source: {path}")
    return path


def filename_identity(name: str) -> dict[str, Any]:
    """Preserve the complete numeric token; do not strip an assumed subject prefix."""
    match = re.fullmatch(r"(.*?)(\d+)([^\d]*)", Path(name).stem)
    return {"filename": name, "filename_index": int(match[2]) if match else None,
            "prefix": match[1] if match else None, "suffix": match[3] if match else None,
            "numeric_token": match[2] if match else None}


def ordered_names(names: list[str]) -> list[str]:
    return sorted(names, key=lambda n: (filename_identity(n)["filename_index"] is None,
                                       filename_identity(n)["filename_index"] or 0, n))


def numbering(names: list[str]) -> dict[str, Any]:
    names = ordered_names(names)
    identities = [filename_identity(n) for n in names]
    indices = [x["filename_index"] for x in identities if x["filename_index"] is not None]
    counts = Counter(indices)
    unique = sorted(counts)
    gaps = [[a + 1, b - 1] for a, b in zip(unique, unique[1:]) if b > a + 1]
    offsets = sorted({index - ordinal for ordinal, index in enumerate(indices)})
    return dict(first_filename=names[0] if names else None,
                last_filename=names[-1] if names else None,
                first_index=identities[0]["filename_index"] if names else None,
                last_index=identities[-1]["filename_index"] if names else None,
                contiguous=bool(names) and len(indices) == len(names) and not gaps
                and len(unique) == len(names),
                duplicate_indices=[i for i, n in sorted(counts.items()) if n > 1],
                gaps=gaps, unparsed=[x["filename"] for x in identities if x["filename_index"] is None],
                prefixes=sorted({x["prefix"] for x in identities if x["prefix"] is not None}),
                suffix_variants=[x["filename"] for x in identities if x["suffix"]],
                index_minus_ordinal=offsets if len(indices) == len(names) else None)


def parse_annotation(text: str) -> dict[str, Any]:
    """Validate one YOLO class/cx/cy/w/h row; coordinates are never used downstream."""
    rows = [line.split() for line in text.splitlines() if line.strip()]
    issues = []
    result = dict(row_count=len(rows), numeric_field_count=None, class_id=None,
                  bbox_edges_outside_unit_square=None)
    if len(rows) != 1:
        issues.append("empty" if not rows else "multiple_rows")
    if len(rows) == 1:
        fields = rows[0]
        result["numeric_field_count"] = len(fields)
        if len(fields) != 5:
            issues.append("expected_five_fields")
        try:
            numbers = [float(x) for x in fields]
            if not all(math.isfinite(x) for x in numbers):
                issues.append("nonfinite")
            if not fields or fields[0] not in ("0", "1"):
                issues.append("invalid_class_id")
            else:
                result["class_id"] = int(fields[0])
            if len(numbers) == 5 and all(math.isfinite(x) for x in numbers):
                _, x, y, w, h = numbers
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    issues.append("invalid_normalized_box")
                result["bbox_edges_outside_unit_square"] = (
                    x-w/2 < 0 or x+w/2 > 1 or y-h/2 < 0 or y+h/2 > 1)
        except ValueError:
            issues.append("nonnumeric")
    result.update(valid=not issues, issues=issues)
    return result


def sequence_summary(annotations: list[dict[str, Any]], video_label: str) -> dict[str, Any]:
    """Transitions are between consecutive observed TXT rows, never AVI/pose frames."""
    nums = numbering([x["filename"] for x in annotations])
    order_ok = not nums["duplicate_indices"] and not nums["unparsed"]
    labels = [x["class_id"] if x["valid"] else None for x in annotations]
    positives = [i for i, value in enumerate(labels) if value == 1]
    complete = order_ok and all(x is not None for x in labels)
    return dict(class0_annotations=labels.count(0), class1_annotations=labels.count(1),
                invalid_annotations=labels.count(None), sequence_order_unambiguous=order_ok,
                first_class1_filename=annotations[positives[0]]["filename"] if positives else None,
                last_class1_filename=annotations[positives[-1]]["filename"] if positives else None,
                first_class1_annotation_index=annotations[positives[0]]["filename_index"] if positives else None,
                last_class1_annotation_index=annotations[positives[-1]]["filename_index"] if positives else None,
                transitions_0_to_1=sum(x == 0 and y == 1 for x, y in zip(labels, labels[1:])) if complete else None,
                transitions_1_to_0=sum(x == 1 and y == 0 for x, y in zip(labels, labels[1:])) if complete else None,
                class1_one_contiguous_interval=(positives == list(range(positives[0], positives[-1]+1)))
                if positives and complete else None,
                adl_contains_class1=bool(positives) if video_label == "non_fall" else None,
                fall_activity_contains_class1=bool(positives) if video_label == "fall" else None)


def pixel_digest(image: np.ndarray | None) -> str | None:
    if image is None:
        return None
    return digest(str((image.shape, image.dtype.str)).encode() + b"\0" + image.tobytes(order="C"))


def image_metrics(a: np.ndarray, b: np.ndarray | None) -> dict[str, Any]:
    if b is None or a.shape != b.shape or a.dtype != b.dtype:
        return dict(same_dimensions=False, exact=False, mae=None, max_abs_error=None,
                    mse=None, psnr_db=None)
    difference = cv2.absdiff(a, b)
    mse = float(np.mean(np.square(difference.astype(np.float64))))
    exact = not np.any(difference)
    return dict(same_dimensions=True, exact=exact, mae=float(np.mean(difference)),
                max_abs_error=int(difference.max()), mse=mse,
                psnr_db=None if exact else float(10 * math.log10(255**2 / mse)))


def exact_correspondence(avi: list[str | None], png: list[str | None]) -> dict[int, tuple[int, str]]:
    """Full ordinal equality or unique monotonic pixel identities; no nearest-match acceptance."""
    if len(avi) == len(png) and all(h is not None and h == png[i] for i, h in enumerate(avi)):
        return {i: (i, "full_sequence_ordinal_pixel_equality") for i in range(len(avi))}
    av = defaultdict(list); pn = defaultdict(list)
    for i, h in enumerate(avi):
        if h is not None:
            av[h].append(i)
    for j, h in enumerate(png):
        if h is not None:
            pn[h].append(j)
    pairs = sorted((indices[0], pn[h][0]) for h, indices in av.items()
                   if len(indices) == 1 and len(pn[h]) == 1)
    # Crossed unique identities do not establish a monotonic timeline.
    if any(left[1] >= right[1] for left, right in zip(pairs, pairs[1:])):
        return {}
    return {i: (j, "unique_pixel_identity_monotonic") for i, j in pairs}


def distribution(values: list[float]) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    return dict(count=len(values), min=float(data.min()) if values else None,
                p50=float(np.percentile(data, 50)) if values else None,
                p95=float(np.percentile(data, 95)) if values else None,
                max=float(data.max()) if values else None,
                mean=float(data.mean()) if values else None)


def duplicate_groups(hashes: list[str | None]) -> list[list[int]]:
    groups = defaultdict(list)
    for i, h in enumerate(hashes):
        if h is not None:
            groups[h].append(i)
    return [indices for indices in groups.values() if len(indices) > 1]


def decode_avi(path: Path, metadata: dict[str, str]) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Same open/read-to-termination semantics as Stage 2; no inference or seeking."""
    capture = cv2.VideoCapture()
    frames = []
    try:
        if not capture.open(str(path)):
            raise ValueError(f"Cannot open {path}")
        properties = {"fps": cv2.CAP_PROP_FPS, "width": cv2.CAP_PROP_FRAME_WIDTH,
                      "height": cv2.CAP_PROP_FRAME_HEIGHT, "frame_count": cv2.CAP_PROP_FRAME_COUNT}
        observed = {key: capture.get(prop) for key, prop in properties.items()}
        if any(observed[key] != float(metadata[key]) for key in properties):
            raise ValueError(f"Stage 2 source metadata changed: {path}")
        backend = capture.getBackendName()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.shape != (int(observed["height"]), int(observed["width"]), 3):
                raise ValueError(f"Invalid decoded frame: {path}, {len(frames)}")
            frames.append(frame)
        return frames, dict(**observed, backend=backend,
                            matches_stage2_count=len(frames) == int(float(metadata["frame_count"])))
    finally:
        capture.release()


def csv_bytes(rows: list[dict[str, Any]], empty_fields: tuple[str, ...] = ("source_video", "detail")) -> bytes:
    fields = list(dict.fromkeys(k for row in rows for k in row)) or list(empty_fields)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, separators=(",", ":")) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()})
    return stream.getvalue().encode()


def audit_video(source: contract.PoseSource, metadata: dict[str, str], expected_avi_hash: str) -> dict[str, Any]:
    require_train(source.split)
    contract.require_subject_access(source.subject_id, "train")
    relative = Path(source.source_video)
    if relative.parts[:2] != (f"Subject.{source.subject_id}", source.activity) or len(relative.parts) != 3:
        raise ValueError("Source path/subject mismatch")
    path = RAW / relative
    checked_path(path, path.parent)
    common = dict(subject_id=source.subject_id, activity=source.activity,
                  source_video=source.source_video, split="train")
    files = [checked_path(p, path.parent) for p in path.parent.iterdir()
             if p.suffix.lower() in (".png", ".txt")]
    png_names = ordered_names([p.name for p in files if p.suffix.lower() == ".png"])
    txt_names = ordered_names([p.name for p in files if p.suffix.lower() == ".txt" and p.name.lower() != "classes.txt"])
    classes_files = [p for p in files if p.name.lower() == "classes.txt"]
    identities = []; schemas = []; annotations = []; classes = []
    for p in sorted(files):
        content = p.read_bytes()
        role = "png" if p.suffix.lower() == ".png" else "classes" if p.name.lower() == "classes.txt" else "annotation"
        identities.append(dict(**common, role=role, source_path=p.relative_to(RAW).as_posix(), sha256=digest(content)))
        if role == "classes":
            classes = content.decode("utf-8-sig").splitlines()
    for name in txt_names:
        try:
            parsed = parse_annotation((path.parent / name).read_text(encoding="utf-8-sig"))
        except UnicodeError:
            parsed = dict(row_count=None, numeric_field_count=None, class_id=None, valid=False,
                          bbox_edges_outside_unit_square=None, issues=["invalid_text_encoding"])
        if parsed["class_id"] is not None and (parsed["class_id"] >= len(classes)
                or classes[parsed["class_id"]] != ("nofall", "fall")[parsed["class_id"]]):
            parsed["issues"].append("class_metadata_mismatch")
            parsed["valid"] = False
        annotation = dict(**filename_identity(name), **parsed)
        annotations.append(annotation)
        schemas.append(dict(**common, **annotation))
    avi_hash = file_digest(path)
    if avi_hash != expected_avi_hash:
        raise ValueError(f"AVI hash differs from Stage 2 source checksum: {source.source_video}")
    identities.append(dict(**common, role="avi", source_path=source.source_video, sha256=avi_hash))
    frames, decoding = decode_avi(path, metadata)
    pngs = [cv2.imread(str(path.parent / name), cv2.IMREAD_COLOR) for name in png_names]
    ah = [pixel_digest(frame) for frame in frames]; ph = [pixel_digest(frame) for frame in pngs]
    mapping = exact_correspondence(ah, ph) if decoding["matches_stage2_count"] else {}
    # Hash agreement is checked by full equality for every accepted mapping.
    assert all(np.array_equal(frames[i], pngs[j]) for i, (j, _) in mapping.items())
    txt_by_stem = defaultdict(list)
    for annotation in annotations:
        txt_by_stem[Path(annotation["filename"]).stem].append(annotation)
    png_by_stem = Counter(Path(n).stem for n in png_names)
    orphan_png = [n for n in png_names if Path(n).stem not in txt_by_stem]
    orphan_txt = [n for n in txt_names if Path(n).stem not in png_by_stem]
    direct = []; local = []; correspondences = []; issues = []
    for i, frame in enumerate(frames):
        if i < len(pngs):
            metrics = image_metrics(frame, pngs[i])
            direct.append(dict(**common, stage2_frame_index=i, png_ordinal_index=i,
                               png_filename=png_names[i], **metrics))
        else:
            direct.append(dict(**common, stage2_frame_index=i, png_ordinal_index=None,
                               png_filename=None, same_dimensions=False, exact=False,
                               mae=None, max_abs_error=None, mse=None, psnr_db=None))
        if i not in mapping:
            # The bounded candidates are diagnostic only, including boundaries of unequal sequences.
            offsets = {0} | {j-k for k, (j, _) in mapping.items() if abs(k-i) <= LOCAL_RADIUS}
            candidates = sorted({j for offset in offsets for j in range(i+offset-LOCAL_RADIUS, i+offset+LOCAL_RADIUS+1)
                                 if 0 <= j < len(pngs)})
            for j in candidates:
                local.append(dict(**common, stage2_frame_index=i, png_ordinal_index=j,
                                  png_filename=png_names[j], offset=j-i, **image_metrics(frame, pngs[j])))
        row = dict(**common, row_kind="avi_frame", stage2_frame_index=i,
                   stage2_timestamp_ms=round(i*1000/decoding["fps"]), avi_pixel_sha256=ah[i],
                   png_filename=None, png_ordinal_index=None, png_filename_index=None,
                   annotation_filename=None, annotation_class_id=None, annotation_semantic_label=None,
                   image_correspondence_status="unresolved", correspondence_status="unresolved",
                   correspondence_method="no_unique_monotonic_exact_evidence")
        if i in mapping:
            j, method = mapping[i]
            name = png_names[j]
            row.update(png_filename=name, png_ordinal_index=j,
                       png_filename_index=filename_identity(name)["filename_index"],
                       image_correspondence_status="exact", correspondence_method=method)
            candidates = txt_by_stem[Path(name).stem]
            if len(candidates) == 1 and png_by_stem[Path(name).stem] == 1:
                ann = candidates[0]
                row.update(annotation_filename=ann["filename"])
                if ann["valid"]:
                    row.update(annotation_class_id=ann["class_id"],
                               annotation_semantic_label=SEMANTICS[ann["class_id"]], correspondence_status="exact")
                else:
                    row.update(correspondence_status="invalid_annotation")
            else:
                row.update(correspondence_status="missing_annotation" if not candidates else "unresolved")
        correspondences.append(row)
    used = {j for j, _ in mapping.values()}
    compared = {(r["stage2_frame_index"], r["png_ordinal_index"]) for r in local}
    for j in range(len(pngs)):
        if j in used or not frames:
            continue
        # Explicitly investigate excess PNGs too, rather than ending at the AVI count.
        centers = {min(j, len(frames)-1)}
        centers.update(i+j-k for i, (k, _) in mapping.items() if abs(k-j) <= LOCAL_RADIUS)
        for i in sorted({i for center in centers for i in range(center-LOCAL_RADIUS, center+LOCAL_RADIUS+1)
                         if 0 <= i < len(frames)}):
            if (i, j) not in compared:
                local.append(dict(**common, stage2_frame_index=i, png_ordinal_index=j,
                                  png_filename=png_names[j], offset=j-i, **image_metrics(frames[i], pngs[j])))
                compared.add((i, j))
    for j, name in enumerate(png_names):
        if j not in used:
            correspondences.append(dict(**common, row_kind="unmapped_png", stage2_frame_index=None,
                                        stage2_timestamp_ms=None, png_filename=name, png_ordinal_index=j,
                                        png_filename_index=filename_identity(name)["filename_index"],
                                        annotation_filename=None, annotation_class_id=None,
                                        annotation_semantic_label=None, image_correspondence_status="unresolved",
                                        correspondence_status="unresolved", correspondence_method="no_accepted_avi_mapping"))
    for name in orphan_txt:
        correspondences.append(dict(**common, row_kind="orphan_annotation", stage2_frame_index=None,
                                    stage2_timestamp_ms=None, png_filename=None, png_ordinal_index=None,
                                    annotation_filename=name, annotation_class_id=None, annotation_semantic_label=None,
                                    image_correspondence_status="missing_png", correspondence_status="missing_png",
                                    correspondence_method="no_exact_stem_png_identity"))
    for role, names, other in (("png_without_txt", orphan_png, txt_names), ("txt_without_png", orphan_txt, png_names)):
        for name in names:
            index = filename_identity(name)["filename_index"]
            candidates = [n for n in other if index is not None and filename_identity(n)["filename_index"] == index]
            issues.append(dict(**common, issue=role, filename=name, candidate_same_numeric_index=candidates,
                               detail="Filename variants are candidates only; no annotation alias is accepted."))
    pn = numbering(png_names); tn = numbering(txt_names)
    inventory = dict(**common, avi_path=path.relative_to(REPO).as_posix(),
                     stage2_frame_count=int(float(metadata["frame_count"])), decoded_avi_frame_count=len(frames),
                     png_count=len(png_names), annotation_txt_count=len(txt_names),
                     classes_txt_presence=bool(classes_files), classes_txt_contents=classes,
                     png_without_txt_count=len(orphan_png), txt_without_png_count=len(orphan_txt),
                     png_minus_avi=len(png_names)-len(frames), txt_minus_avi=len(txt_names)-len(frames),
                     **{f"png_{key}": value for key, value in pn.items()},
                     **{f"txt_{key}": value for key, value in tn.items()},
                     invalid_annotations=sum(not x["valid"] for x in annotations),
                     direct_ordinal_exact_frames=sum(x["exact"] for x in direct),
                     exact_image_mapped_frames=len(mapping),
                     exact_annotation_mapped_frames=sum(x["correspondence_status"] == "exact" for x in correspondences),
                     unresolved_avi_frames=len(frames)-len(mapping),
                     unmapped_png_count=len(png_names)-len(used),
                     png_duplicate_pixel_groups=duplicate_groups(ph), avi_duplicate_pixel_groups=duplicate_groups(ah),
                     mapping_offsets=sorted({j-i for i, (j, _) in mapping.items()}),
                     png_unreadable=[png_names[j] for j, image in enumerate(pngs) if image is None],
                     **decoding)
    if len(frames) != len(png_names) or len(frames) != len(txt_names):
        issues.append(dict(**common, issue="count_mismatch", filename=None,
                           detail=dict(avi=len(frames), png=len(png_names), txt=len(txt_names))))
    if len(mapping) != len(frames) or len(used) != len(png_names):
        issues.append(dict(**common, issue="unresolved_image_correspondence", filename=None,
                           detail=dict(avi_indices=[i for i in range(len(frames)) if i not in mapping],
                                       png_ordinals=[j for j in range(len(png_names)) if j not in used])))
    for role, nums in (("png", pn), ("txt", tn)):
        if nums["duplicate_indices"] or nums["gaps"] or nums["unparsed"]:
            issues.append(dict(**common, issue=f"{role}_numbering", filename=None, detail=nums))
    if inventory["png_duplicate_pixel_groups"] or inventory["avi_duplicate_pixel_groups"]:
        issues.append(dict(**common, issue="duplicate_pixels", filename=None,
                           detail=dict(png=inventory["png_duplicate_pixel_groups"], avi=inventory["avi_duplicate_pixel_groups"])))
    return dict(inventory=inventory, sequence=dict(**common, video_label=source.video_label,
                **sequence_summary(annotations, source.video_label)), schema=schemas, identities=identities,
                correspondence=correspondences, direct=direct, local=local, issues=issues)


def analyze(split: str = "train") -> dict[str, bytes]:
    require_train(split)  # Must precede selector and all source I/O.
    sources = contract.select_sources(("train",), purpose="exploration")
    if len(sources) != 60 or {s.subject_id for s in sources} != set(contract.FROZEN_SUBJECTS["train"]):
        raise ValueError("Unexpected Train scope")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if head != BASELINE:
        raise ValueError("Stage 3.1 parent changed")
    cfg = contract.read_contract()
    evidence = REPO / cfg["source"]["evidence_root"]
    with (REPO / "artifacts/dataset_inspection/inventory.csv").open(newline="") as f:
        inventory = {r["relative_video_path"]: r for r in csv.DictReader(f)}
    with (evidence / "source_checksums.csv").open(newline="") as f:
        checksums = {r["relative_video_path"]: r["sha256"] for r in csv.DictReader(f)}
    results = []
    for index, source in enumerate(sources):
        LOG.info("Train video %d/%d: %s", index+1, len(sources), source.source_video)
        results.append(audit_video(source, inventory[source.source_video], checksums[source.source_video]))
    tables = {key: [row for result in results for row in result[key]]
              for key in ("schema", "identities", "correspondence", "direct", "local", "issues")}
    videos = [r["inventory"] for r in results]; sequences = [r["sequence"] for r in results]
    definitions = dict(pixel_representation="OpenCV uint8 BGR, unresized; original dimensions; PNG IMREAD_COLOR",
        pixel_hash="SHA256(str((shape,dtype.str)).encode + NUL + C-order image bytes); accepted matches also array_equal",
        mae="mean(abs(AVI-PNG)) across all original-resolution pixels and BGR channels; intensity units 0..255",
        mse="mean((AVI-PNG)^2), float64; all pixels/channels",
        max_abs_error="maximum absolute pixel/channel difference",
        psnr_db="10*log10(255^2/MSE); null for exact pairs (infinity) or incompatible dimensions",
        metric_distributions="NumPy linear quantiles; only comparable pairs; same_dimensions/exact disambiguate null PSNR",
        local_search=f"Unresolved AVI frames: ordinal +/- {LOCAL_RADIUS}, plus offsets of exact anchors within {LOCAL_RADIUS} AVI frames. Unmapped PNGs: bounded ordinal/boundary and nearby exact-anchor comparisons too. All candidates recorded; no threshold and no nearest-match acceptance",
        accepted_image_mapping="full equal-length ordinal pixel equality, else unique exact pixel identities with strictly monotonic PNG ordinals; crossed/duplicate identities remain unresolved",
        accepted_annotation_identity="unique exact basename only; same numeric token with different suffix is an unresolved candidate",
        filename_index="entire final digit group of stem, preserving any subject digits; not assumed zero/one based",
        png_ordinal_index="zero-based position in numeric-token/name ordering, separately recorded",
        stage2_frame_index="sequential AVI read ordinal, zero-based; round(i*1000/source_fps) milliseconds",
        class_semantics=SEMANTICS, annotation_sequence="ordered original TXT only; observed-row transitions, no imputation across filename gaps; ambiguous order/invalid rows suppress transitions",
        bbox="one class/cx/cy/w/h row; finite; class literal 0/1; centers [0,1], sizes (0,1]; edge overhang reported separately, never repaired",
        statuses=["exact", "unresolved", "missing_annotation", "missing_png", "invalid_annotation"],
        row_accounting="every AVI frame once; every PNG not accepted onto AVI once; every orphan TXT once; schema table contains every TXT",
        limitations="OpenCV read termination cannot distinguish EOF from decoder failure; count agreement checked against Stage 2. No pose arrays read. Local V5 download identity not independently authenticated.")
    counts = {key: sum(v[key] for v in videos) for key in (
        "decoded_avi_frame_count", "png_count", "annotation_txt_count", "png_without_txt_count",
        "txt_without_png_count", "invalid_annotations", "direct_ordinal_exact_frames", "exact_image_mapped_frames",
        "exact_annotation_mapped_frames", "unresolved_avi_frames", "unmapped_png_count")}
    counts.update(videos=len(videos), classes_txt=sum(v["classes_txt_presence"] for v in videos),
                  class0_annotations=sum(x["class0_annotations"] for x in sequences),
                  class1_annotations=sum(x["class1_annotations"] for x in sequences))
    summary = dict(analysis_version=VERSION, status="pending_external_gate_review", counts=counts,
        provenance=dict(parent_stage31_commit=BASELINE, git_head=head, code_sha256=file_digest(Path(__file__)),
                        stage2_run_id=contract.RUN_ID, split="train", subject_ids=list(contract.FROZEN_SUBJECTS["train"]),
                        source_root=str(RAW.relative_to(REPO)), source_identities="source_files.csv",
                        stage3_config_sha256=file_digest(contract.CONFIG), stage3_guard_sha256=file_digest(Path(contract.__file__)),
                        stage2_manifest_sha256=file_digest(evidence/"manifest.csv"),
                        python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__,
                        official_annotation_documentation=SOURCE_DOCUMENTATION), definitions=definitions,
        count_mismatch_videos=[v["source_video"] for v in videos if v["png_minus_avi"] or v["txt_minus_avi"]],
        unresolved_videos=[v["source_video"] for v in videos if v["exact_annotation_mapped_frames"] != v["decoded_avi_frame_count"]
                           or v["unmapped_png_count"] or v["txt_without_png_count"]],
        direct_similarity={key: distribution([r[key] for r in tables["direct"] if r.get(key) is not None])
                           for key in ("mae", "max_abs_error", "mse", "psnr_db")},
        local_similarity={key: distribution([r[key] for r in tables["local"] if r.get(key) is not None])
                          for key in ("mae", "max_abs_error", "mse", "psnr_db")},
        sequence_findings=sequences)
    files = {"per_video_annotation_inventory.csv": csv_bytes(videos),
             "annotation_sequence_summary.csv": csv_bytes(sequences),
             "annotation_schema.csv": csv_bytes(tables["schema"]),
             "source_files.csv": csv_bytes(tables["identities"]),
             "frame_correspondence.csv": csv_bytes(tables["correspondence"]),
             "direct_ordinal_diagnostics.csv": csv_bytes(tables["direct"]),
             "local_alignment_diagnostics.csv": csv_bytes(tables["local"]),
             "mismatch_cases.csv": csv_bytes(tables["issues"]),
             "summary.json": (json.dumps(summary, indent=2, allow_nan=False)+"\n").encode()}
    lines = ["# Stage 3.2a Train annotation and alignment audit", "", "Pending external Gate Review. No PASS claim.", "",
             "## Counts", "", *[f"- {k}: {v}" for k, v in counts.items()], "", "## Count mismatches", ""]
    lines += [f"- {v['source_video']}: AVI {v['decoded_avi_frame_count']}, PNG {v['png_count']}, TXT {v['annotation_txt_count']}"
              for v in videos if v["png_minus_avi"] or v["txt_minus_avi"]] or ["None."]
    lines += ["", "## Unresolved videos", "", *[f"- {p}" for p in summary["unresolved_videos"]], "",
              "## Evidence and interpretation", "",
              "The CSV tables retain all AVI frames, unmatched PNGs and orphan TXT files. Exact mappings require pixel equality; filename suffix candidates and non-exact image similarities remain unresolved. No similarity threshold is used.", "",
              "The full filename numeric token is separate from PNG ordinal and Stage 2 frame_index. See per-video offsets and direct/local diagnostics; no leading, trailing or internal element is silently discarded.", "",
              f"The [dataset authors]({SOURCE_DOCUMENTATION}) describe class 1 as a body on the ground following a fall. Its semantic name here is grounded_fall_state, not motion onset.", "",
              "Annotation sequence summaries concern original TXT order only. Boxes are schema diagnostics. No labels were joined to pose arrays, no arrays were loaded, no windows or preprocessing outputs were built, and Validation/Test contents were not accessed.", "",
              "Metric definitions, provenance, source hashes and all unresolved cases are machine-readable in summary.json and the companion CSVs.", ""]
    files["REPORT.md"] = "\n".join(lines).encode()
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    require_train(args.split)
    output = contract.validate_output_path(args.output)
    if not output.is_relative_to(REPO / "artifacts/preprocessing"):
        raise ValueError("Stage 3.2a writes analysis artifacts only")
    files = analyze(args.split)
    output = contract.validate_output_path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        with (output / name).open("xb") as handle:
            handle.write(content)
    LOG.info("Published %d audit artifacts to %s", len(files), output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
