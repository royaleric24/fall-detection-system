"""Stage 3.2b Train PNG/TXT ordinal feasibility; no AVI decoding or pose extraction."""

import argparse
import csv
import json
import logging
import platform
import re
import struct
import subprocess
import xml.etree.ElementTree as ET
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ml.preprocessing import audit_annotation_alignment as prior
from ml.preprocessing import contract

REPO = contract.REPO
RAW = prior.RAW
BASELINE = "98c34e553b3a75dddef6f28816db2227a6188aa7"
VERSION = "stage32b_train_annotation_native_timeline_v1"
OUTPUT = REPO / "artifacts/preprocessing" / VERSION
WORKBOOK = RAW.parent / "Dataset details.xlsx"
EXTERNAL_URL = "https://data.mendeley.com/datasets/7w7fccy7ky/5"
LOG = logging.getLogger(__name__)
ALIASES = {
    "Subject.2/Fall backwards/FallBackwardsS2.avi": {"cas200091 - copia.png": "cas200091.txt"},
    "Subject.8/Hop/HopS8.avi": {"sals800096.png": "sals800096a.txt"},
    "Subject.8/Pick up object/PickupobjectS8.avi": {"res800090.png": "res800090a.txt"},
    "Subject.9/Walk/WalkS9.avi": {"cams900140.png": "cams900140w.txt"},
}


def require_train_subject(subject: int, split: str) -> None:
    require_train(split)
    contract.require_subject_access(subject, split)


def require_train(split: str) -> None:
    if split != "train":
        raise contract.ContractError("Stage 3.2b is Train only")


def sequence_key(name: str) -> tuple[str | None, str | None]:
    identity = prior.filename_identity(name)
    # Preserve digit spelling too: no implicit removal of zeros or subject digits.
    return identity["prefix"], identity["numeric_token"]


def pair_sequence(png_names: list[str], txt_names: list[str], aliases: dict[str, str]) -> tuple[list[dict[str, Any]], str]:
    """Resolve exact stems or four explicit aliases only after unique identity checks."""
    png_names = prior.ordered_names(png_names); txt_names = prior.ordered_names(txt_names)
    png_counts = Counter(sequence_key(n) for n in png_names)
    txt_counts = Counter(sequence_key(n) for n in txt_names)
    stems = defaultdict(list)
    for name in txt_names:
        stems[Path(name).stem].append(name)
    records = []; used = []
    for ordinal, name in enumerate(png_names):
        identity = prior.filename_identity(name)
        key = sequence_key(name)
        options = stems[Path(name).stem]
        annotation = None; status = "unresolved"
        unique = identity["filename_index"] is not None and png_counts[key] == txt_counts[key] == 1
        if unique and len(options) == 1:
            annotation = options[0]; status = "exact_stem"
        elif unique and not options and name in aliases:
            target = aliases[name]
            if target in txt_names and sequence_key(target) == key:
                annotation = target; status = "verified_explicit_sequence_alias"
        if annotation is not None:
            used.append(annotation)
        records.append(dict(png_filename=name, png_ordinal_index=ordinal,
            png_filename_index=identity["filename_index"], png_numeric_token=identity["numeric_token"],
            png_prefix=identity["prefix"], png_suffix=identity["suffix"], annotation_filename=annotation,
            annotation_pairing_status=status, png_timestamp_ms=None, avi_frame_index=None,
            annotation_timebase_status="unresolved"))
    pn = prior.numbering(png_names); tn = prior.numbering(txt_names)
    verified = (bool(records) and len(used) == len(png_names) == len(txt_names)
                and len(set(used)) == len(used) and not pn["duplicate_indices"] and not tn["duplicate_indices"]
                and not pn["unparsed"] and not tn["unparsed"]
                and len(pn["prefixes"]) == 1 and pn["prefixes"] == tn["prefixes"])
    order = "verified" if verified else "unresolved"
    for row in records:
        row["annotation_order_status"] = order
    return records, order


def duplicate_audit(hashes: list[str], names: list[str], kind: str) -> list[dict[str, Any]]:
    """Exact hash groups only; pixel groups are confirmed by array equality by the caller."""
    rows = []
    for group in prior.duplicate_groups(hashes):
        adjacent = [[a, b] for a, b in zip(group, group[1:]) if b == a+1]
        leading = [0, 1] in adjacent
        trailing = [len(names)-2, len(names)-1] in adjacent
        rows.append(dict(identity_kind=kind, ordinals=group, filenames=[names[i] for i in group],
                         count=len(group), excess_identical_copies=len(group)-1,
                         leading_boundary_duplicate=leading, trailing_boundary_duplicate=trailing,
                         internal_adjacent_pairs=[pair for pair in adjacent if pair not in ([0,1],[len(names)-2,len(names)-1])],
                         touches_first=0 in group, touches_last=len(names)-1 in group,
                         interpretation="exact repeated content; cause and physical timing not inferred"))
    return rows


def png_metadata(content: bytes) -> dict[str, Any]:
    """Inspect structural/ancillary chunks; never interpret file modification time as capture time."""
    if content[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Invalid PNG signature")
    offset = 8; chunks = []; metadata = []
    while offset < len(content):
        if offset+12 > len(content):
            raise ValueError("Truncated PNG chunk")
        length = struct.unpack(">I", content[offset:offset+4])[0]
        kind = content[offset+4:offset+8].decode("ascii")
        data = content[offset+8:offset+8+length]
        if offset+12+length > len(content):
            raise ValueError("Truncated PNG payload")
        chunks.append(kind)
        if kind == "tEXt":
            metadata.append(dict(kind=kind, value=data.decode("latin1")))
        elif kind == "zTXt":
            key, packed = data.split(b"\0", 1)
            metadata.append(dict(kind=kind, key=key.decode("latin1"),
                                 value=zlib.decompress(packed[1:]).decode("latin1")))
        elif kind in ("iTXt", "eXIf", "tIME"):
            metadata.append(dict(kind=kind, payload_hex=data.hex(), interpretation="not a verified frame capture timestamp"))
        offset += length+12
        if kind == "IEND":
            break
    return dict(chunk_types=sorted(set(chunks)), ancillary_records=metadata)


def ordinal_labels(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [r.get("annotation_class_id") for r in records]
    complete = all(x in (0, 1) for x in labels)
    positives = [i for i, x in enumerate(labels) if x == 1]
    return dict(class0_count=labels.count(0), class1_count=labels.count(1),
        first_grounded_fall_ordinal=positives[0] if positives else None,
        last_grounded_fall_ordinal=positives[-1] if positives else None,
        transitions_0_to_1=[i for i in range(1,len(labels)) if labels[i-1:i+1] == [0,1]] if complete else None,
        transitions_1_to_0=[i for i in range(1,len(labels)) if labels[i-1:i+1] == [1,0]] if complete else None,
        class1_one_interval=positives == list(range(positives[0],positives[-1]+1)) if positives and complete else None)


def timing_diagnostics(png_count: int, avi_count: int, avi_fps: float) -> dict[str, Any]:
    """Ratios use AVI reference durations, with no assignment of PNG timestamps/rate."""
    if avi_count < 2 or avi_fps <= 0:
        raise ValueError("Expected at least two AVI samples and positive source FPS")
    count_duration = avi_count/avi_fps
    first_last_span = (avi_count-1)/avi_fps
    return dict(png_count=png_count, avi_count=avi_count, local_avi_fps=avi_fps,
        avi_count_duration_seconds=count_duration, avi_first_last_sample_span_seconds=first_last_span,
        png_count_over_avi_count_duration=png_count/count_duration,
        png_count_over_avi_first_last_span=png_count/first_last_span,
        png_intervals_over_avi_count_duration=(png_count-1)/count_duration,
        png_intervals_over_avi_first_last_span=(png_count-1)/first_last_span,
        annotation_timebase_status="unresolved", verified_png_fps=None,
        interpretation="ratios only; equal acquisition coverage and constant PNG spacing are unverified")


def workbook_train_counts(path: Path, subjects: tuple[int, ...]) -> list[dict[str, Any]]:
    """Read the known Hoja1 block layout; select subjects before resolving data cells."""
    for subject in subjects:
        require_train_subject(subject, "train")
    ns = {"s":"http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        strings = ET.fromstring(z.read("xl/sharedStrings.xml")).findall("s:si", ns)
        tree = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        cells = {c.get("r"):c for c in tree.findall(".//s:sheetData/s:row/s:c",ns)}
        def value(ref):
            c = cells.get(ref)
            if c is None:
                return ""
            v = c.find("s:v",ns)
            text = v.text if v is not None else ""
            if c.get("t") == "s" and text:
                return "".join(t.text or "" for t in strings[int(text)].findall(".//s:t",ns))
            return text
        records = []
        for subject in subjects:
            row = 8+15*((subject-1)//2)
            activity_col, count_col = ("C","D") if subject%2 else ("I","J")
            if value(f"{activity_col}{row}") != f"Subject {subject}:":
                raise ValueError("Workbook subject block layout changed")
            if value(f"{activity_col}{row+2}") != "Activity" or value(f"{count_col}{row+2}") != "Frames":
                raise ValueError("Workbook headers changed")
            for r in range(row+3,row+13):
                a_ref = f"{activity_col}{r}"; n_ref = f"{count_col}{r}"
                n = value(n_ref)
                if not n.isdigit():
                    raise ValueError(f"Nonintegral documented frame count at {n_ref}")
                if cells[n_ref].find("s:f",ns) is not None:
                    raise ValueError("Unexpected formula; frame-count derivation requires review")
                records.append(dict(subject_id=subject, activity=value(a_ref), workbook=path.name,
                    sheet="Hoja1", activity_cell=a_ref, frames_cell=n_ref, documented_frames=int(n),
                    count_generation_method="undocumented", fps_or_timestamp_field="absent_from_inspected_Train_blocks"))
    return records


def discover_local_provenance(sources: tuple[contract.PoseSource, ...], frame_paths: set[str]) -> list[dict[str, Any]]:
    directories = {RAW.parent, RAW}
    for source in sources:
        require_train_subject(source.subject_id, source.split)
        directory = (RAW/source.source_video).parent
        directories.update((directory, directory.parent))
    rows = []
    for directory in sorted(directories):
        if directory.absolute() != directory.resolve():
            raise ValueError("Redirected provenance directory")
        for p in sorted(directory.iterdir()):
            if not p.is_file() or p.name == ".DS_Store":
                continue
            rel = p.relative_to(REPO).as_posix()
            raw_rel = p.relative_to(RAW).as_posix() if p.is_relative_to(RAW) else None
            if raw_rel in frame_paths or p.suffix.lower() == ".avi":
                continue
            prior.checked_path(p,directory)
            inspected = p == WORKBOOK
            notes = "Workbook Train blocks inspected" if inspected else "General illustration; not used as timing evidence"
            excerpts = []
            if p.suffix.lower() in (".txt", ".md", ".rst", ".json", ".py", ".sh", ".yaml", ".yml", ".xml", ".csv"):
                text = p.read_text(encoding="utf-8")
                excerpts = [line for line in text.splitlines() if re.search(r"fps|frame|timestamp|extract|transcod",line,re.I)]
                inspected = True; notes = "Text provenance searched; relevant lines retained for review"
            rows.append(dict(source_path=rel, content_inspected=inspected, sha256=prior.file_digest(p),
                             notes=notes, relevant_text=excerpts))
    return rows


def audit_sequence(source: contract.PoseSource, ledger: dict[str,str], old: dict[str,str], workbook: dict[str,Any]) -> dict[str, Any]:
    require_train_subject(source.subject_id, source.split)
    relative = Path(source.source_video)
    if relative.parts[:2] != (f"Subject.{source.subject_id}", source.activity) or len(relative.parts) != 3:
        raise ValueError("Source identity mismatch")
    directory = (RAW/relative).parent
    if directory.resolve() != directory.absolute():
        raise ValueError("Redirected sequence directory")
    paths = {p.name:prior.checked_path(p,directory) for p in directory.iterdir() if p.suffix.lower() in (".png",".txt")}
    png_names = prior.ordered_names([n for n in paths if n.lower().endswith(".png")])
    txt_names = prior.ordered_names([n for n in paths if n.lower().endswith(".txt") and n.lower() != "classes.txt"])
    expected = {Path(p).name for p in ledger if Path(p).parent == relative.parent}
    if set(paths) != expected:
        raise ValueError("PNG/TXT inventory changed since Stage 3.2a")
    common = dict(subject_id=source.subject_id, activity=source.activity, source_video_identity=source.source_video, split="train")
    annotations = {}; classes = []
    for name in sorted(n for n in paths if n.lower().endswith(".txt")):
        content = paths[name].read_bytes()
        if prior.digest(content) != ledger[(relative.parent/name).as_posix()]:
            raise ValueError(f"Annotation bytes changed: {name}")
        if name.lower() == "classes.txt":
            classes = content.decode("utf-8-sig").splitlines()
        else:
            annotations[name] = prior.parse_annotation(content.decode("utf-8-sig"))
    if not all(r["valid"] for r in annotations.values()):
        raise ValueError("Annotation schema differs from reviewed evidence")
    records, order_status = pair_sequence(png_names,txt_names,ALIASES.get(source.source_video,{}))
    file_hashes = []; pixel_hashes = []; unique_images = {}; metadata_counter = Counter()
    for row in records:
        name = row["png_filename"]; data = paths[name].read_bytes(); h = prior.digest(data)
        if h != ledger[(relative.parent/name).as_posix()]:
            raise ValueError(f"PNG bytes changed: {name}")
        image = cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"PNG decode failure: {name}")
        pixel = prior.pixel_digest(image)
        if pixel in unique_images:
            if not np.array_equal(image,unique_images[pixel]):
                raise ValueError("Pixel hash collision")
        else:
            unique_images[pixel] = image
        file_hashes.append(h); pixel_hashes.append(pixel)
        meta = png_metadata(data); metadata_counter.update(meta["chunk_types"])
        row.update(common, file_sha256=h, pixel_sha256=pixel, **meta)
        ann = annotations.get(row["annotation_filename"])
        label = ann["class_id"] if ann else None
        if label is not None and (label >= len(classes) or classes[label] != ("nofall","fall")[label]):
            raise ValueError("Class metadata mismatch")
        row.update(annotation_class_id=label, annotation_semantic_label=prior.SEMANTICS.get(label),
                   annotation_sha256=ledger.get((relative.parent/row["annotation_filename"]).as_posix()) if ann else None)
    duplicates = [dict(common,**r) for r in duplicate_audit(file_hashes,png_names,"file_sha256")
                  + duplicate_audit(pixel_hashes,png_names,"decoded_pixel_sha256")]
    boundaries = []
    for role,ordinal in (("first",0),("second",1),("penultimate",len(records)-2),("last",len(records)-1)):
        if 0 <= ordinal < len(records):
            row = records[ordinal]
            boundaries.append(dict(common,boundary_role=role,**{k:row[k] for k in (
                "png_ordinal_index","png_filename","file_sha256","pixel_sha256")}))
    anomalies = []
    for row in records:
        if row["annotation_pairing_status"] != "exact_stem":
            ordinal = row["png_ordinal_index"]
            anomalies.append(dict(common, anomaly="stem_identity", status=row["annotation_pairing_status"],
                png_filename=row["png_filename"], annotation_filename=row["annotation_filename"],
                png_ordinal_index=ordinal, sequence_key=sequence_key(row["png_filename"]),
                previous_png=png_names[ordinal-1] if ordinal else None,
                next_png=png_names[ordinal+1] if ordinal+1<len(png_names) else None,
                evidence="explicit known pair; unique same prefix and unchanged numeric token in each file set; no competing identity; sources retained"))
    pn=prior.numbering(png_names); tn=prior.numbering(txt_names)
    for kind,numbering in (("png",pn),("txt",tn)):
        if numbering["gaps"] or numbering["duplicate_indices"] or numbering["unparsed"]:
            anomalies.append(dict(common,anomaly=f"{kind}_numbering",status="observed_no_repair",evidence=numbering))
    pixel_groups = [r for r in duplicates if r["identity_kind"] == "decoded_pixel_sha256"]
    excess = sum(r["excess_identical_copies"] for r in pixel_groups)
    avi_count = int(old["decoded_avi_frame_count"])
    delta = len(records)-avi_count
    anomalies.append(dict(common,anomaly="avi_png_count_difference",status="unresolved",
        evidence=dict(png_minus_avi=delta,pixel_duplicate_excess=excess,
                      unique_png_count_minus_avi=len(records)-excess-avi_count,
                      interpretation="duplicate removal is hypothetical arithmetic, not an authorized timeline repair or causal explanation")))
    workbook_record = dict(common,**{k:v for k,v in workbook.items() if k not in ("subject_id","activity")},
                           actual_png_count=len(records),png_minus_documented_frames=len(records)-workbook["documented_frames"])
    inventory = dict(common,png_count=len(records),txt_count=len(txt_names),
        ordinal_first=0 if records else None,ordinal_last=len(records)-1 if records else None,
        **{f"png_{k}":v for k,v in pn.items()},**{f"txt_{k}":v for k,v in tn.items()},
        annotation_order_status=order_status,annotation_timebase_status="unresolved",
        exact_stem_pairs=sum(r["annotation_pairing_status"]=="exact_stem" for r in records),
        explicit_sequence_alias_pairs=sum(r["annotation_pairing_status"]=="verified_explicit_sequence_alias" for r in records),
        decoded_pixel_duplicate_groups=len(pixel_groups),decoded_pixel_duplicate_excess=excess,
        file_duplicate_groups=sum(r["identity_kind"]=="file_sha256" for r in duplicates),
        leading_boundary_duplicate=any(r["leading_boundary_duplicate"] for r in pixel_groups),
        trailing_boundary_duplicate=any(r["trailing_boundary_duplicate"] for r in pixel_groups),
        avi_frame_count_reference=avi_count,png_minus_avi=delta,count_difference_status="unresolved",
        workbook_frames=workbook["documented_frames"],workbook_reference=f"Hoja1!{workbook['frames_cell']}",
        provenance_references=["P1","P3","P4","P7","P14","P15"],
        png_chunk_type_counts=dict(metadata_counter),anomaly_flags=[r["anomaly"] for r in anomalies],
        **ordinal_labels(records))
    return dict(records=records,inventory=inventory,duplicates=duplicates,boundaries=boundaries,
                anomalies=anomalies,workbook=workbook_record,
                timing=dict(common,**timing_diagnostics(len(records),avi_count,float(old["fps"]))))


def analyze(split: str = "train") -> dict[str,bytes]:
    require_train(split)
    sources=contract.select_sources(("train",),purpose="exploration")
    if len(sources)!=60 or {s.subject_id for s in sources}!=set(contract.FROZEN_SUBJECTS["train"]):
        raise ValueError("Unexpected Train scope")
    if subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()!=BASELINE:
        raise ValueError("Frozen Stage 3.2a baseline changed")
    cfg=contract.read_contract()
    if any(v!="UNDECIDED" for v in cfg["decisions"].values()):
        raise ValueError("Frozen preprocessing decisions changed")
    def read_prior(name):
        with (prior.OUTPUT/name).open(newline="") as f:
            return list(csv.DictReader(f))
    validation=json.loads((prior.OUTPUT/"validation_record.json").read_text())
    for name,h in validation["artifact_sha256"].items():
        if prior.file_digest(prior.OUTPUT/name)!=h:
            raise ValueError(f"Frozen Stage 3.2a artifact changed: {name}")
    selected={s.source_video for s in sources}
    ledger_rows=read_prior("source_files.csv")
    if any(r["source_video"] not in selected or r["split"]!="train" for r in ledger_rows):
        raise ValueError("Unexpected non-Train ledger identity")
    ledger={r["source_path"]:r["sha256"] for r in ledger_rows if r["role"] in ("png","annotation","classes")}
    old={r["source_video"]:r for r in read_prior("per_video_annotation_inventory.csv")}
    workbook_rows=workbook_train_counts(WORKBOOK,contract.FROZEN_SUBJECTS["train"])
    workbook={(r["subject_id"],r["activity"]):r for r in workbook_rows}
    if set(workbook)!={(s.subject_id,s.activity) for s in sources}:
        raise ValueError("Workbook Train activity coverage mismatch")
    search=discover_local_provenance(sources,set(ledger))
    results=[]
    for i,source in enumerate(sources):
        LOG.info("Train PNG sequence %d/60: %s",i+1,source.source_video)
        results.append(audit_sequence(source,ledger,old[source.source_video],workbook[(source.subject_id,source.activity)]))
    inventories=[r["inventory"] for r in results]
    tables={k:[x for r in results for x in r[k]] for k in ("records","duplicates","boundaries","anomalies")}
    workbook_evidence=[r["workbook"] for r in results]
    timing=[r["timing"] for r in results]
    totals=dict(sequences=len(inventories),png_files=sum(r["png_count"] for r in inventories),
        txt_files=sum(r["txt_count"] for r in inventories),exact_stem_pairs=sum(r["exact_stem_pairs"] for r in inventories),
        explicit_sequence_alias_pairs=sum(r["explicit_sequence_alias_pairs"] for r in inventories),
        order_status_counts=dict(Counter(r["annotation_order_status"] for r in inventories)),
        timebase_status_counts=dict(Counter(r["annotation_timebase_status"] for r in inventories)),
        sequences_with_pixel_duplicates=sum(r["decoded_pixel_duplicate_groups"]>0 for r in inventories),
        pixel_duplicate_groups=sum(r["decoded_pixel_duplicate_groups"] for r in inventories),
        pixel_duplicate_excess=sum(r["decoded_pixel_duplicate_excess"] for r in inventories),
        file_duplicate_groups=sum(r["file_duplicate_groups"] for r in inventories),
        class0_count=sum(r["class0_count"] for r in inventories),
        class1_count=sum(r["class1_count"] for r in inventories),
        avi_frames_reference=sum(r["avi_frame_count_reference"] for r in inventories),
        png_minus_avi_counts=dict(Counter(r["png_minus_avi"] for r in inventories)),
        leading_boundary_duplicate_sequences=sum(r["leading_boundary_duplicate"] for r in inventories),
        trailing_boundary_duplicate_sequences=sum(r["trailing_boundary_duplicate"] for r in inventories),
        workbook_train_frames=sum(r["documented_frames"] for r in workbook_evidence),
        workbook_count_mismatches=sum(r["png_minus_documented_frames"]!=0 for r in workbook_evidence),
        png_ancillary_record_count=sum(len(r["ancillary_records"]) for r in tables["records"]))
    provenance_rows=[
        dict(evidence_id="P1",category="FACT_SUPPORTED_BY_LOCAL_EVIDENCE",topic="local AVI FPS",statement="Selected Train AVI metadata and Stage 2 timeline use 20 FPS; reused frozen Stage 1/3.2a records, no AVI reread.",source="artifacts/dataset_inspection/inventory.csv; Stage 3.2a per_video_annotation_inventory.csv",limitation="Does not establish PNG sample spacing"),
        dict(evidence_id="P2",category="FACT_SUPPORTED_ONLY_BY_EXTERNAL_DATASET_DOCUMENTATION",topic="camera acquisition specification",statement="Official V5 page describes camera capability of 23 FPS at 1080x960, with motion-triggered DVR recording.",source=EXTERNAL_URL,limitation="Capability wording is not a verified per-recording clock or distributed PNG rate"),
        dict(evidence_id="P3",category="FACT_SUPPORTED_BY_LOCAL_EVIDENCE",topic="workbook counts",statement=f"Train blocks have Activity/Frames columns; documented Train total {totals['workbook_train_frames']}; no FPS or timestamp field in these blocks.",source="data/raw/caucafall_v5/Dataset details.xlsx; workbook_train_evidence.csv",limitation="How frame counts were produced is undocumented; count agreement does not establish sampling intervals"),
        dict(evidence_id="P4",category="DERIVED_OBSERVATION",topic="workbook reconciliation",statement=f"{totals['workbook_count_mismatches']} Train sequences differ from the workbook frame count.",source="workbook_train_evidence.csv",limitation="Missing filename indices can explain count arithmetic, not physical capture history"),
        dict(evidence_id="P5",category="DERIVED_OBSERVATION",topic="PNG time ratios",statement="PNG sample/interval counts divided by two explicitly defined AVI duration conventions are diagnostics only.",source="timebase_diagnostics.csv",limitation="Equal temporal coverage and constant PNG spacing are not established"),
        dict(evidence_id="P6",category="FACT_SUPPORTED_BY_LOCAL_EVIDENCE",topic="local provenance search",statement=f"Inspected available workbook Train blocks and searched global/Train directories for text records; {sum(r['content_inspected'] for r in search)} supplemental files inspected.",source="local_provenance_search.csv",limitation="Limited to supplied dataset root and Train directories; general JPEG illustrations supply no asserted timing facts"),
        dict(evidence_id="P7",category="FACT_SUPPORTED_BY_LOCAL_EVIDENCE",topic="PNG ancillary metadata",statement=f"{totals['png_ancillary_record_count']} textual/time/EXIF ancillary records found among selected PNGs.",source="annotation_native_frames.csv",limitation="Modification-time metadata, if present, is not assumed to be capture time"),
    ]
    for i,topic in enumerate(("AVI transcoding FPS/process","PNG extraction/sample rate","every captured frame exported to PNG","PNGs extracted from distributed AVI versus original recording","PNG physical timestamps","Dataset_details frame-count generation","systematic PNG-minus-AVI +1 cause","Subject.2 Fall backwards +2 cause"),8):
        provenance_rows.append(dict(evidence_id=f"P{i}",category="UNRESOLVED",topic=topic,
            statement="Available evidence does not establish this fact.",source="local_provenance_search.csv; annotation_native_sequence_inventory.csv; png_duplicate_audit.csv",limitation="No assumption, sequence repair, or downstream policy selected"))
    summary=dict(analysis_version=VERSION,status="pending_external_gate_review",totals=totals,
        provenance=dict(parent_stage32a_commit=BASELINE,stage2_run_id=contract.RUN_ID,split="train",
            subject_ids=list(contract.FROZEN_SUBJECTS["train"]),code_sha256=prior.file_digest(Path(__file__)),
            workbook_path=str(WORKBOOK.relative_to(REPO)),workbook_sha256=prior.file_digest(WORKBOOK),
            stage32a_summary_sha256=prior.file_digest(prior.OUTPUT/"summary.json"),
            stage32a_sources_sha256=prior.file_digest(prior.OUTPUT/"source_files.csv"),
            stage3_config_sha256=prior.file_digest(contract.CONFIG),
            external_documentation_url=EXTERNAL_URL,external_documentation_version=5,
            python=platform.python_version(),opencv=cv2.__version__,numpy=np.__version__,png_source_hashes_verified_against_stage32a=True),
        definitions=dict(annotation_order_status={"verified":"Unique filename order and bijective PNG/TXT sequence identity; not proof of uniform physical time or complete acquisition coverage","unresolved":"Unparsed/duplicate ordering keys or non-bijective identity"},
            annotation_timebase_status={"unresolved":"No verified PNG sampling interval or per-image physical timestamp"},
            explicit_alias_rule="Only the four declared anomalies; unique identical prefix and numeric token in each sequence, no competing stem, explicit names preserved. Establishes deterministic identity, not independent author-intent certification.",
            duplicates="Exact byte SHA-256 or IMREAD_UNCHANGED decoded shape/dtype/pixel SHA-256; repeated pixel hashes confirmed by array_equal. No similarity threshold and no deletion.",
            workbook="Known Hoja1 block layout; Train subjects selected before resolving activity/count cells; no subject-specific Validation/Test values analyzed.",
            timing_formulas={"avi_count_duration_seconds":"T_avi/f_avi", "avi_first_last_sample_span_seconds":"(T_avi-1)/f_avi", "png_count_over_avi_count_duration":"N_png/(T_avi/f_avi)", "png_count_over_avi_first_last_span":"N_png/((T_avi-1)/f_avi)", "png_intervals_over_avi_count_duration":"(N_png-1)/(T_avi/f_avi)", "png_intervals_over_avi_first_last_span":"(N_png-1)/((T_avi-1)/f_avi)"},
            timing_limitations="These compare counts to AVI reference durations only; same endpoints/coverage and constant PNG cadence are unverified. None is accepted as PNG FPS.",
            filename_vs_ordinal="Full displayed numeric token retained; ordinal is zero-based sorted sequence position; AVI frame index and PNG timestamp remain null.",
            labels="0 nofall, 1 grounded_fall_state; transitions reported only as PNG ordinals"),
        conclusions=dict(physical_png_fps=None,systematic_plus_one="unresolved",subject2_fall_backwards_plus_two="unresolved; exact duplicate group documented separately",
            acquisition_avi_difference="External 23 FPS camera capability and local 20 FPS AVI are distinct provenance layers; their relationship is undocumented",
            downstream_decisions=cfg["decisions"],avi_decoded=False,pose_arrays_loaded=False,mediapipe_run=False,png_timestamps_fabricated=False))
    files={"annotation_native_sequence_inventory.csv":prior.csv_bytes(inventories),
        "annotation_native_frames.csv":prior.csv_bytes(tables["records"]),
        "png_duplicate_audit.csv":prior.csv_bytes(tables["duplicates"]),
        "png_boundary_audit.csv":prior.csv_bytes(tables["boundaries"]),
        "timeline_provenance.csv":prior.csv_bytes(provenance_rows),
        "timebase_diagnostics.csv":prior.csv_bytes(timing),
        "anomaly_cases.csv":prior.csv_bytes(tables["anomalies"]),
        "workbook_train_evidence.csv":prior.csv_bytes(workbook_evidence),
        "local_provenance_search.csv":prior.csv_bytes(search),
        "summary.json":(json.dumps(summary,indent=2,allow_nan=False)+"\n").encode()}
    lines=["# Stage 3.2b annotation-native timeline feasibility", "", "Pending external Gate Review; no PASS claim.", "",
        "## Findings", "",*[f"- {k}: {v}" for k,v in totals.items()], "",
        "Verified order means deterministic annotation-native ordinal identity only. It does not establish physical spacing, a PNG FPS, or correspondence to the Stage 2 AVI timeline. Every PNG timestamp and AVI frame-index field remains empty.", "",
        "The four explicit filename aliases require unique identical prefix/numeric-token identity and no competing candidate. No source name is normalized or changed. See anomaly_cases.csv for names and neighbors.", "",
        "## Exact duplicate groups", "", "| Sequence | Kind | PNG ordinals | Leading pair | Trailing pair |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {r['source_video_identity']} | {r['identity_kind']} | {r['ordinals']} | {r['leading_boundary_duplicate']} | {r['trailing_boundary_duplicate']} |" for r in tables["duplicates"]]
    lines += ["", "Exact repetition is retained. Uneven duplication across sequences does not objectively establish a common +1 cause. Subject.2 / Fall backwards has its own +2 anomaly; duplicate counts are arithmetic diagnostics, not a repaired mapping.", "",
        "## Local workbook discrepancies", ""]
    lines += [f"- {r['source_video_identity']}: workbook {r['documented_frames']} at {r['sheet']}!{r['frames_cell']}; PNGs {r['actual_png_count']}." for r in workbook_evidence if r["png_minus_documented_frames"]]
    lines += ["", "## Temporal provenance", "",
        f"The [official V5 documentation]({EXTERNAL_URL}) describes a camera capable of 23 FPS. Frozen local AVI metadata says 20 FPS. Neither value establishes PNG sampling. The inspected local workbook contains frame counts without a documented count-generation method or timebase. See timeline_provenance.csv for local facts, external claims, derived observations and unresolved facts.", "",
        "Timebase diagnostics separately use AVI T/f and (T-1)/f denominators, and PNG N or N-1 numerators. These ratios are not PNG FPS estimates accepted for preprocessing. All PNG timebases remain unresolved.", "",
        "Grounded-fall transitions are ordinal positions only. No AVI-pose-label join, PNG pose extraction, MediaPipe call, resampling, window, or model training occurred. Validation/Test subject contents were not accessed. All downstream preprocessing decisions remain UNDECIDED.", ""]
    files["REPORT.md"]="\n".join(lines).encode()
    return files


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split",default="train")
    args=parser.parse_args()
    require_train(args.split)
    output=contract.validate_output_path(OUTPUT)
    files=analyze(args.split)
    output=contract.validate_output_path(output)
    output.mkdir(parents=True,exist_ok=False)
    for name,content in files.items():
        with (output/name).open("xb") as f:
            f.write(content)
    LOG.info("Published %d Stage 3.2b audit artifacts to %s",len(files),output)


if __name__=="__main__":
    logging.basicConfig(level=logging.INFO,format="%(levelname)s %(message)s")
    main()
