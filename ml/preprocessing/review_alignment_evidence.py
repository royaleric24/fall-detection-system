"""Summarize saved Stage 3.2a comparisons without rereading media or accepting mappings."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ml.preprocessing import audit_annotation_alignment as audit


def candidate_rankings(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Rank only the recorded candidates. Ties and absent comparisons remain explicit."""
    groups = defaultdict(list)
    for row in rows:
        if row["mae"] and row["mse"]:
            groups[(row["source_video"], int(row["stage2_frame_index"]))].append(row)
    result = []
    for (video, index), candidates in sorted(groups.items()):
        record = dict(source_video=video, stage2_frame_index=index,
                      candidate_png_ordinals=sorted(int(r["png_ordinal_index"]) for r in candidates),
                      correspondence_status="unresolved", method="descriptive_rank_only")
        for metric in ("mae", "mse"):
            scores = sorted({float(r[metric]) for r in candidates})
            best = sorted(int(r["png_ordinal_index"]) for r in candidates if float(r[metric]) == scores[0])
            record.update({f"best_{metric}_png_ordinals": best,
                           f"best_{metric}": scores[0],
                           f"next_distinct_{metric}_margin": scores[1]-scores[0] if len(scores)>1 else None,
                           f"ordinal_is_{metric}_minimum": index in best})
        result.append(record)
    return result


def offset_summaries(rows: list[dict[str, str]], inventories: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        if row["mae"]:
            groups[(row["source_video"], int(row["offset"]))].append(row)
    result = []
    for inventory in inventories:
        video = inventory["source_video"]
        for offset in range(-audit.LOCAL_RADIUS, audit.LOCAL_RADIUS+1):
            selected = groups[(video, offset)]
            result.append(dict(source_video=video, png_ordinal_minus_avi_index=offset,
                comparable_pairs=len(selected),
                avi_frames_not_compared=int(inventory["decoded_avi_frame_count"])-len(selected),
                png_frames_not_compared=int(inventory["png_count"])-len(selected),
                mae=audit.distribution([float(r["mae"]) for r in selected]),
                mse=audit.distribution([float(r["mse"]) for r in selected]),
                status="diagnostic_only_no_threshold_or_acceptance"))
    return result


def main() -> None:
    selected = audit.contract.select_sources(("train",), purpose="exploration")
    sources = {s.source_video for s in selected}
    base = audit.OUTPUT
    summary = json.loads((base/"summary.json").read_text())
    if summary["provenance"]["split"] != "train" or summary["provenance"]["code_sha256"] != audit.file_digest(Path(audit.__file__)):
        raise ValueError("Audit provenance mismatch")
    def read(name):
        with (base/name).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if any(r["source_video"] not in sources or r.get("split", "train") != "train" for r in rows):
            raise ValueError("Non-Train artifact row")
        return rows
    inventories = read("per_video_annotation_inventory.csv")
    if {r["source_video"] for r in inventories} != sources or len(inventories) != 60:
        raise ValueError("Incomplete Train evidence")
    local = read("local_alignment_diagnostics.csv")
    ranks = candidate_rankings(local)
    offsets = offset_summaries(local, inventories)
    sequences = read("annotation_sequence_summary.csv")
    schema = read("annotation_schema.csv")
    issues = read("mismatch_cases.csv")
    expected_counts = dict(videos=60, decoded_avi_frame_count=11115, png_count=11176,
                           annotation_txt_count=11176, invalid_annotations=0,
                           class0_annotations=7927, class1_annotations=3249,
                           exact_image_mapped_frames=0, png_without_txt_count=4, txt_without_png_count=4)
    if any(summary["counts"][key] != value for key, value in expected_counts.items()):
        raise ValueError("These review notes require the recorded v1 counts")
    stats = {}
    for metric in ("mae", "mse"):
        unique = [r for r in ranks if len(r[f"best_{metric}_png_ordinals"]) == 1]
        stats[metric] = dict(ranked_avi_frames=len(ranks),
            unique_minimum=len(unique), tied_minimum=len(ranks)-len(unique),
            ordinal_unique_minimum=sum(r[f"best_{metric}_png_ordinals"] == [r["stage2_frame_index"]] for r in ranks),
            ordinal_among_minima=sum(r[f"ordinal_is_{metric}_minimum"] for r in ranks),
            unique_best_offset_counts=dict(sorted(Counter(r[f"best_{metric}_png_ordinals"][0]-r["stage2_frame_index"] for r in unique).items())))
    supplemental = dict(status="pending_external_gate_review_all_candidate_ranks_are_diagnostic",
        code_sha256=audit.file_digest(Path(__file__)), audit_summary_sha256=audit.file_digest(base/"summary.json"),
        input_sha256={name:audit.file_digest(base/name) for name in (
            "per_video_annotation_inventory.csv", "local_alignment_diagnostics.csv",
            "annotation_sequence_summary.csv", "annotation_schema.csv", "mismatch_cases.csv")},
        split="train", pose_or_media_read=False, candidate_ranking=stats,
        limitation="Local minimum is not a temporal match. Compression, low motion, repeated frames and the bounded candidate set can make minima/ties misleading. No candidate rank changes frame_correspondence.csv.")
    lines = ["# Stage 3.2a review notes from saved evidence", "", "Pending external Gate Review; no PASS claim.", "",
        "The source audit decoded each Train AVI once. This supplement reads only its CSV/JSON artifacts.", "",
        "## Main findings", "",
        "All 60 Train videos have more PNG/TXT files than decoded AVI frames. 59 have one extra pair; Subject.2 / Fall backwards has two. The totals are 11,115 AVI frames and 11,176 PNGs / 11,176 frame TXT files. Subject.8 / Fall forward is 122 / 123 / 123; its count does not explain the earlier full-dataset difference.", "",
        "No pixel-exact AVI/PNG identity was found. All 11,115 AVI frames and all 60 videos therefore remain unresolved under the declared evidence rules. This does not establish different scene content: direct comparisons have small nonzero errors, but those alone do not prove timing.", "",
        f"Among saved local candidates, PNG ordinal i is the unique MAE minimum for {stats['mae']['ordinal_unique_minimum']} AVI frames, and the unique MSE minimum for {stats['mse']['ordinal_unique_minimum']}. MAE ties occur for {stats['mae']['tied_minimum']} frames. These are descriptive ranks, not accepted correspondences. No threshold is selected.", "",
        "Ordinary ordinal correspondence is a candidate for much of the sequence, with possible trailing extra images; it is not proven across every frame. Duplicate PNG runs, non-ordinal minima, and the Subject.2 / Fall backwards local shift prevent a globally certified rule. Low-motion tails cannot locate every extra image reliably. Nothing is truncated.", "",
        "## Every count mismatch / unresolved video", "", "| Source video | AVI | PNG | TXT |", "| --- | ---: | ---: | ---: |"]
    lines += [f"| {r['source_video']} | {r['decoded_avi_frame_count']} | {r['png_count']} | {r['annotation_txt_count']} |" for r in inventories]
    lines += ["", "## Orphan identities", ""]
    lines += [f"- {r['source_video']}: {r['issue']} `{r['filename']}`; numeric-token candidate {r['candidate_same_numeric_index']}." for r in issues if r["issue"] in ("png_without_txt", "txt_without_png")]
    lines += ["", "Only exact stems are accepted as PNG/TXT identities; all four suffix/name variants remain unresolved candidates.", "",
        "## Annotation and sequence findings", "",
        "All 11,176 frame TXT files have one valid five-field row, finite normalized-box values, valid class IDs and matching classes metadata. There are 7,927 class-0 and 3,249 class-1 annotations. No duplicate numeric TXT identities were found. Three boxes extend beyond the image boundary when center/size is converted to corners; these were reported without repair:", ""]
    lines += [f"- {r['source_video']}: `{r['filename']}`." for r in schema if r["bbox_edges_outside_unit_square"] == "True"]
    lines += ["", "All 30 ADL videos contain only class 0. All 30 fall-activity videos contain one class-1 interval in observed TXT order, with one 0→1 transition. 29 remain class 1 to the final annotation; Subject.9 / Fall forward returns to class 0 after annotation `cfs900103.txt` (class 1 starts at `cfs900057.txt`). These describe grounded_fall_state in the original annotations, not fall onset or window labels.", "",
        "Most filename indices start at subject_id*100000+1; Subject.1 / Fall forward starts at 1. Subject.1 / Sit down lacks numeric index 100148 in both PNG and TXT, so its index-minus-ordinal offset changes. Numeric order is not a Stage 2 frame index.", "",
        "## Exact duplicate image evidence", "", "PNG ordinal groups below are exactly identical to each other. That does not make them identical to decoded AVI frames:", ""]
    lines += [f"- {r['source_video']}: {r['png_duplicate_pixel_groups']}." for r in inventories if r["png_duplicate_pixel_groups"] != "[]"]
    lines += ["", "In Subject.2 / Fall backwards, PNG ordinals 90 and 91 are identical (`cas200091 - copia.png` and `cas200092.png`). This localizes a duplicate-image pair; it does not prove which AVI frame should receive either annotation or fully explain both excess images. Bounded local ranks and per-offset comparison coverage are provided separately.", "",
        "## Gate questions", "",
        "Annotation schema is internally well formed under the documented checks. Exact-stem PNG/TXT association covers 11,172 pairs, with four unresolved aliases. AVI-to-annotation correspondence is not yet established without ambiguity. There is no accepted universal mapping or supported_nonexact label assignment. All unresolved identities are retained, and no pose-label join, preprocessing output, window or training was created. Validation/Test contents were not accessed.", ""]
    files = {"candidate_rankings.csv": audit.csv_bytes(ranks),
             "offset_similarity_summary.csv": audit.csv_bytes(offsets),
             "review_summary.json": (json.dumps(supplemental, indent=2, allow_nan=False)+"\n").encode(),
             "REVIEW_NOTES.md": "\n".join(lines).encode()}
    for name in files:
        if (base/name).exists():
            raise ValueError(f"Refusing overwrite: {name}")
    for name, content in files.items():
        with (base/name).open("xb") as handle:
            handle.write(content)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
