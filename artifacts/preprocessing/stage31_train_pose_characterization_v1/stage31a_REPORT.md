# Stage 3.1a — Train-only missingness addendum

Status: HOLD addendum for external Gate Review. This extends the existing Stage 3.1 analysis without changing its original files or the dataset.

## Missing-run topology

| Type | Runs | Missing frames | Share of missing frames | Median support (ms) | Maximum support (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| internal | 135 | 672 | 42.5% | 50.0 | 6450.0 |
| leading | 33 | 633 | 40.1% | 700.0 | 4600.0 |
| trailing | 2 | 153 | 9.7% | 3825.0 | 7600.0 |
| whole_video | 1 | 122 | 7.7% | 6100.0 | 6100.0 |

`missing_support_ms` is the timestamp support of missing samples: last missing timestamp minus first missing timestamp plus one 20 FPS frame period. For N missing frames it is N × 50 ms. For internal gaps, `anchor_gap_ms` is the right observed timestamp minus the left observed timestamp; it is a different quantity. Non-internal anchors are null.

## Subject-stratified fall minus non-fall differences

Each subject contributes five fall and five non-fall videos. Values below are descriptive differences of equal-video summaries; no significance test was run.

| Subject | Mean missing fraction | Median missing fraction | Mean longest support (ms) | Median longest support (ms) | Mean run count |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0358 | 0.0110 | 570.0000 | 50.0000 | 0.6000 |
| 2 | 0.0039 | 0.0000 | -60.0000 | 0.0000 | -1.2000 |
| 3 | -0.1501 | -0.2755 | -920.0000 | -950.0000 | -6.6000 |
| 4 | 0.0047 | -0.0100 | -50.0000 | -50.0000 | 0.4000 |
| 8 | 0.4049 | 0.5597 | 2950.0000 | 5050.0000 | 0.4000 |
| 9 | 0.0607 | -0.0126 | -210.0000 | 50.0000 | 7.8000 |

## Whole-video-missing sensitivity

Omitted **only in the sensitivity view**: Subject.8/Fall forward/FallForwardS8.avi. The original 60-video dataset and Stage 3.1 tables remain unchanged.

| View | Class | Videos | Mean video missing fraction | Mean longest support (ms) | Mean run count |
| --- | --- | ---: | ---: | ---: | ---: |
| all_train | fall | 30 | 0.1705 | 1173.3 | 2.97 |
| all_train | non_fall | 30 | 0.1105 | 793.3 | 2.73 |
| excluding_whole_video_missing | fall | 29 | 0.1419 | 1003.4 | 3.03 |
| excluding_whole_video_missing | non_fall | 30 | 0.1105 | 793.3 | 2.73 |

Frames are nested within videos, and videos within subjects. Class differences can reflect subject, activity or tracking characteristics. This diagnostic establishes neither causality nor predictive usefulness. No preprocessing threshold or exclusion policy is selected. Validation/Test arrays were not loaded.
