# Stage 3.1 Train-only pose characterization

Analysis version: `stage31_train_pose_characterization_v1`. Source: official Stage 2 run `4df7dd5f-fb38-4908-8233-1a81deb8dc05`. Stage 3.0 baseline: `9d1912cd4c1566438a60496d5d055a43189d3a66`. Status: pending external Gate Review.

All results are descriptive and Train-only. Pose availability is MediaPipe pose return rate, not fall-detection accuracy or robustness. Frame/joint samples within videos are correlated.

## Inventory and missingness

6 subjects, 60 videos, 11115 frames; 9535 detected and 1580 missing. Pooled pose availability: 85.7850%.

171 missing runs; median 100.0 ms, p95 2025.0 ms, maximum 7600.0 ms. Zero-detected-pose Train videos: 1 (Subject.8/Fall forward/FallForwardS8.avi). Missing-frame counts by progress quintile: [609, 288, 285, 269, 129].

## Missingness shortcut-risk diagnostic

Fall videos: 30, pooled availability 82.4069%, mean per-video missing fraction 0.1705, mean longest gap 1173.3 ms, mean run count 2.97.

Non-fall videos: 30, pooled availability 88.6774%, mean per-video missing fraction 0.1105, mean longest gap 793.3 ms, mean run count 2.73.

These differences cannot establish causality or a predictive shortcut. A later mask-only ablation would need a separately approved evaluation.

## Visibility and geometry

Visibility was summarized for 9535 detected frames; see `landmark_visibility.csv` for all 33 joints and `summary.json` for per-frame summaries. No visibility threshold was selected.

Raw x/y shoulder width median: 0.0686563; hip width median: 0.0392608; shoulder-to-hip midpoint torso median: 0.146927. These are diagnostic image-relative 2-D distances, not chosen normalization scales. Exact-zero counts and full distributions are in `summary.json`.

## Integrity and limits

All selected NPZs passed the frozen raw schema validation. Reported zero violation counts follow from that validation; no raw values were repaired, clipped or rejected by range. The source download's V5 identity was locally supplied and not independently authenticated. All Stage 3.0 preprocessing decisions remain `UNDECIDED`. No Validation/Test arrays were loaded.
