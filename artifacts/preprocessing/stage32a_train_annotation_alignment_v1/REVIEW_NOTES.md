# Stage 3.2a review notes from saved evidence

Pending external Gate Review; no PASS claim.

The source audit decoded each Train AVI once. This supplement reads only its CSV/JSON artifacts.

## Main findings

All 60 Train videos have more PNG/TXT files than decoded AVI frames. 59 have one extra pair; Subject.2 / Fall backwards has two. The totals are 11,115 AVI frames and 11,176 PNGs / 11,176 frame TXT files. Subject.8 / Fall forward is 122 / 123 / 123; its count does not explain the earlier full-dataset difference.

No pixel-exact AVI/PNG identity was found. All 11,115 AVI frames and all 60 videos therefore remain unresolved under the declared evidence rules. This does not establish different scene content: direct comparisons have small nonzero errors, but those alone do not prove timing.

Among saved local candidates, PNG ordinal i is the unique MAE minimum for 10086 AVI frames, and the unique MSE minimum for 10542. MAE ties occur for 79 frames. These are descriptive ranks, not accepted correspondences. No threshold is selected.

Ordinary ordinal correspondence is a candidate for much of the sequence, with possible trailing extra images; it is not proven across every frame. Duplicate PNG runs, non-ordinal minima, and the Subject.2 / Fall backwards local shift prevent a globally certified rule. Low-motion tails cannot locate every extra image reliably. Nothing is truncated.

## Every count mismatch / unresolved video

| Source video | AVI | PNG | TXT |
| --- | ---: | ---: | ---: |
| Subject.1/Fall backwards/FallBackwardsS1.avi | 125 | 126 | 126 |
| Subject.1/Fall forward/FallForwardS1.avi | 190 | 191 | 191 |
| Subject.1/Fall left/FallLeftS1.avi | 86 | 87 | 87 |
| Subject.1/Fall right/FallRightS1.avi | 114 | 115 | 115 |
| Subject.1/Fall sitting/FallSittingS1.avi | 182 | 183 | 183 |
| Subject.1/Hop/HopS1.avi | 182 | 183 | 183 |
| Subject.1/Kneel/KneelS1.avi | 219 | 220 | 220 |
| Subject.1/Pick up object/PickupobjectS1.avi | 117 | 118 | 118 |
| Subject.1/Sit down/SitDownS1.avi | 187 | 188 | 188 |
| Subject.1/Walk/WalkS1.avi | 242 | 243 | 243 |
| Subject.2/Fall backwards/FallBackwardsS2.avi | 119 | 121 | 121 |
| Subject.2/Fall forward/FallForwardS2.avi | 106 | 107 | 107 |
| Subject.2/Fall left/FallLeftS2.avi | 109 | 110 | 110 |
| Subject.2/Fall right/FallRightS2.avi | 146 | 147 | 147 |
| Subject.2/Fall sitting/FallSittingS2.avi | 157 | 158 | 158 |
| Subject.2/Hop/HopS2.avi | 171 | 172 | 172 |
| Subject.2/Kneel/KneelS2.avi | 130 | 131 | 131 |
| Subject.2/Pick up object/PickupobjectS2.avi | 166 | 167 | 167 |
| Subject.2/Sit down/SitDownS2.avi | 112 | 113 | 113 |
| Subject.2/Walk/WalkS2.avi | 244 | 245 | 245 |
| Subject.3/Fall backwards/FallBackwardsS3.avi | 166 | 167 | 167 |
| Subject.3/Fall forward/FallForwardS3.avi | 153 | 154 | 154 |
| Subject.3/Fall left/FallLeftS3.avi | 171 | 172 | 172 |
| Subject.3/Fall right/FallRightS3.avi | 220 | 221 | 221 |
| Subject.3/Fall sitting/FallSittingS3.avi | 202 | 203 | 203 |
| Subject.3/Hop/HopS3.avi | 162 | 163 | 163 |
| Subject.3/Kneel/KneelS3.avi | 213 | 214 | 214 |
| Subject.3/Pick up object/PickupobjectS3.avi | 186 | 187 | 187 |
| Subject.3/Sit down/SitDownS3.avi | 261 | 262 | 262 |
| Subject.3/Walk/WalkS3.avi | 176 | 177 | 177 |
| Subject.4/Fall backwards/FallBackwardsS4.avi | 192 | 193 | 193 |
| Subject.4/Fall forward/FallForwardS4.avi | 168 | 169 | 169 |
| Subject.4/Fall left/FallLeftS4.avi | 162 | 163 | 163 |
| Subject.4/Fall right/FallRightS4.avi | 150 | 151 | 151 |
| Subject.4/Fall sitting/FallSittingS4.avi | 198 | 199 | 199 |
| Subject.4/Hop/HopS4.avi | 144 | 145 | 145 |
| Subject.4/Kneel/KneelS4.avi | 222 | 223 | 223 |
| Subject.4/Pick up object/PickupobjectS4.avi | 179 | 180 | 180 |
| Subject.4/Sit down/SitDownS4.avi | 199 | 200 | 200 |
| Subject.4/Walk/WalkS4.avi | 240 | 241 | 241 |
| Subject.8/Fall backwards/FallBackwardsS8.avi | 212 | 213 | 213 |
| Subject.8/Fall forward/FallForwardS8.avi | 122 | 123 | 123 |
| Subject.8/Fall left/FallLeftS8.avi | 202 | 203 | 203 |
| Subject.8/Fall right/FallRightS8.avi | 229 | 230 | 230 |
| Subject.8/Fall sitting/FallSittingS8.avi | 234 | 235 | 235 |
| Subject.8/Hop/HopS8.avi | 166 | 167 | 167 |
| Subject.8/Kneel/KneelS8.avi | 256 | 257 | 257 |
| Subject.8/Pick up object/PickupobjectS8.avi | 266 | 267 | 267 |
| Subject.8/Sit down/SitDownS8.avi | 212 | 213 | 213 |
| Subject.8/Walk/WalkS8.avi | 216 | 217 | 217 |
| Subject.9/Fall backwards/FallBackwardsS9.avi | 252 | 253 | 253 |
| Subject.9/Fall forward/FallForwardS9.avi | 148 | 149 | 149 |
| Subject.9/Fall left/FallLeftS9.avi | 207 | 208 | 208 |
| Subject.9/Fall right/FallRightS9.avi | 189 | 190 | 190 |
| Subject.9/Fall sitting/FallSittingS9.avi | 216 | 217 | 217 |
| Subject.9/Hop/HopS9.avi | 212 | 213 | 213 |
| Subject.9/Kneel/KneelS9.avi | 216 | 217 | 217 |
| Subject.9/Pick up object/PickupobjectS9.avi | 220 | 221 | 221 |
| Subject.9/Sit down/SitDownS9.avi | 238 | 239 | 239 |
| Subject.9/Walk/WalkS9.avi | 234 | 235 | 235 |

## Orphan identities

- Subject.2/Fall backwards/FallBackwardsS2.avi: png_without_txt `cas200091 - copia.png`; numeric-token candidate ["cas200091.txt"].
- Subject.2/Fall backwards/FallBackwardsS2.avi: txt_without_png `cas200091.txt`; numeric-token candidate ["cas200091 - copia.png"].
- Subject.8/Hop/HopS8.avi: png_without_txt `sals800096.png`; numeric-token candidate ["sals800096a.txt"].
- Subject.8/Hop/HopS8.avi: txt_without_png `sals800096a.txt`; numeric-token candidate ["sals800096.png"].
- Subject.8/Pick up object/PickupobjectS8.avi: png_without_txt `res800090.png`; numeric-token candidate ["res800090a.txt"].
- Subject.8/Pick up object/PickupobjectS8.avi: txt_without_png `res800090a.txt`; numeric-token candidate ["res800090.png"].
- Subject.9/Walk/WalkS9.avi: png_without_txt `cams900140.png`; numeric-token candidate ["cams900140w.txt"].
- Subject.9/Walk/WalkS9.avi: txt_without_png `cams900140w.txt`; numeric-token candidate ["cams900140.png"].

Only exact stems are accepted as PNG/TXT identities; all four suffix/name variants remain unresolved candidates.

## Annotation and sequence findings

All 11,176 frame TXT files have one valid five-field row, finite normalized-box values, valid class IDs and matching classes metadata. There are 7,927 class-0 and 3,249 class-1 annotations. No duplicate numeric TXT identities were found. Three boxes extend beyond the image boundary when center/size is converted to corners; these were reported without repair:

- Subject.1/Walk/WalkS1.avi: `cams100116.txt`.
- Subject.3/Fall forward/FallForwardS3.avi: `cfs300100.txt`.
- Subject.9/Fall sitting/FallSittingS9.avi: `css900145.txt`.

All 30 ADL videos contain only class 0. All 30 fall-activity videos contain one class-1 interval in observed TXT order, with one 0→1 transition. 29 remain class 1 to the final annotation; Subject.9 / Fall forward returns to class 0 after annotation `cfs900103.txt` (class 1 starts at `cfs900057.txt`). These describe grounded_fall_state in the original annotations, not fall onset or window labels.

Most filename indices start at subject_id*100000+1; Subject.1 / Fall forward starts at 1. Subject.1 / Sit down lacks numeric index 100148 in both PNG and TXT, so its index-minus-ordinal offset changes. Numeric order is not a Stage 2 frame index.

## Exact duplicate image evidence

PNG ordinal groups below are exactly identical to each other. That does not make them identical to decoded AVI frames:

- Subject.1/Fall forward/FallForwardS1.avi: [[12,13]].
- Subject.1/Fall left/FallLeftS1.avi: [[5,6],[71,72]].
- Subject.1/Fall sitting/FallSittingS1.avi: [[89,90],[116,117,118],[119,120,121,122]].
- Subject.1/Kneel/KneelS1.avi: [[8,9]].
- Subject.2/Fall backwards/FallBackwardsS2.avi: [[90,91]].
- Subject.2/Hop/HopS2.avi: [[106,107]].
- Subject.2/Pick up object/PickupobjectS2.avi: [[142,143]].
- Subject.2/Walk/WalkS2.avi: [[191,192]].
- Subject.4/Fall forward/FallForwardS4.avi: [[0,1]].
- Subject.4/Pick up object/PickupobjectS4.avi: [[169,170,171,172,173,174,175,176,177,178,179]].
- Subject.9/Fall backwards/FallBackwardsS9.avi: [[218,219]].
- Subject.9/Fall sitting/FallSittingS9.avi: [[179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216]].
- Subject.9/Hop/HopS9.avi: [[13,14],[59,60]].
- Subject.9/Kneel/KneelS9.avi: [[71,72]].

In Subject.2 / Fall backwards, PNG ordinals 90 and 91 are identical (`cas200091 - copia.png` and `cas200092.png`). This localizes a duplicate-image pair; it does not prove which AVI frame should receive either annotation or fully explain both excess images. Bounded local ranks and per-offset comparison coverage are provided separately.

## Gate questions

Annotation schema is internally well formed under the documented checks. Exact-stem PNG/TXT association covers 11,172 pairs, with four unresolved aliases. AVI-to-annotation correspondence is not yet established without ambiguity. There is no accepted universal mapping or supported_nonexact label assignment. All unresolved identities are retained, and no pose-label join, preprocessing output, window or training was created. Validation/Test contents were not accessed.
