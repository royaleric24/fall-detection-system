# Stage 2.6 full extraction report

Run: `4df7dd5f-fb38-4908-8233-1a81deb8dc05`. Status: **complete**.

Pose availability is diagnostic only, not classifier accuracy or robustness. No coverage threshold is applied.
Only validated complete pairs contribute to stored-frame totals; failed-prefix counters remain in the ledger.
All missing runs are reported without defining a long-run cutoff. Zero-pose videos are retained.
Subjects 6/7 use only the frozen mechanical extraction and validation path; no policy tuning or qualitative analysis.

## Totals and audits

```json
{
  "run_id": "4df7dd5f-fb38-4908-8233-1a81deb8dc05",
  "status": "complete",
  "totals": {
    "videos": 100,
    "statuses": {
      "complete": 100
    },
    "expected_frames": 19877,
    "validated_extracted_frames": 19877,
    "detected": 17116,
    "missing": 2761
  },
  "pose_availability": 0.8610957387935805,
  "audits": {
    "source_checksums": true,
    "implementation_hashes": true,
    "runtime_model_config": true,
    "post_tests": true,
    "held_out_policy": true
  },
  "zero_pose_videos": [
    "Subject.8/Fall forward/FallForwardS8.avi"
  ],
  "per_split": {
    "train": {
      "videos": 60,
      "statuses": {
        "complete": 60
      },
      "expected_frames": 11115,
      "validated_extracted_frames": 11115,
      "detected": 9535,
      "missing": 1580
    },
    "validation": {
      "videos": 20,
      "statuses": {
        "complete": 20
      },
      "expected_frames": 4674,
      "validated_extracted_frames": 4674,
      "detected": 3976,
      "missing": 698
    },
    "test": {
      "videos": 20,
      "statuses": {
        "complete": 20
      },
      "expected_frames": 4088,
      "validated_extracted_frames": 4088,
      "detected": 3605,
      "missing": 483
    }
  },
  "per_activity": {
    "Fall backwards": {
      "videos": 10,
      "statuses": {
        "complete": 10
      },
      "expected_frames": 2040,
      "validated_extracted_frames": 2040,
      "detected": 2021,
      "missing": 19
    },
    "Fall forward": {
      "videos": 10,
      "statuses": {
        "complete": 10
      },
      "expected_frames": 1694,
      "validated_extracted_frames": 1694,
      "detected": 1124,
      "missing": 570
    },
    "Fall left": {
      "videos": 10,
      "statuses": {
        "complete": 10
      },
      "expected_frames": 1837,
      "validated_extracted_frames": 1837,
      "detected": 1348,
      "missing": 489
    },
    "Fall right": {
      "videos": 10,
      "statuses": {
        "complete": 10
      },
      "expected_frames": 1864,
      "validated_extracted_frames": 1864,
      "detected": 1611,
      "missing": 253
    },
    "Fall sitting": {
      "videos": 10,
      "statuses": {
        "complete": 10
      },
      "expected_frames": 2161,
      "validated_extracted_frames": 2161,
      "detected": 1993,
      "missing": 168
    },
    "Hop": {
      "videos": 10,
      "statuses": {
        "complete": 10
      },
      "expected_frames": 1767,
      "validated_extracted_frames": 1767,
      "detected": 1639,
      "missing": 128
    },
    "Kneel": {
      "videos": 10,
      "statuses": {
        "complete": 10
      },
      "expected_frames": 2116,
      "validated_extracted_frames": 2116,
      "detected": 1781,
      "missing": 335
    },
    "Pick up object": {
      "videos": 10,
      "statuses": {
        "complete": 10
      },
      "expected_frames": 2052,
      "validated_extracted_frames": 2052,
      "detected": 1613,
      "missing": 439
    },
    "Sit down": {
      "videos": 10,
      "statuses": {
        "complete": 10
      },
      "expected_frames": 2085,
      "validated_extracted_frames": 2085,
      "detected": 1955,
      "missing": 130
    },
    "Walk": {
      "videos": 10,
      "statuses": {
        "complete": 10
      },
      "expected_frames": 2261,
      "validated_extracted_frames": 2261,
      "detected": 2031,
      "missing": 230
    }
  }
}
```

## Per-video results

| Source | Status | Expected | Extracted | Detected | Missing | Availability | Longest missing run | Error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Subject.1/Fall backwards/FallBackwardsS1.avi | complete | 125 | 125 | 125 | 0 | 1.0 | 0 |  |
| Subject.1/Fall forward/FallForwardS1.avi | complete | 190 | 190 | 94 | 96 | 0.49473684210526314 | 92 |  |
| Subject.1/Fall left/FallLeftS1.avi | complete | 86 | 86 | 86 | 0 | 1.0 | 0 |  |
| Subject.1/Fall right/FallRightS1.avi | complete | 114 | 114 | 110 | 4 | 0.9649122807017544 | 3 |  |
| Subject.1/Fall sitting/FallSittingS1.avi | complete | 182 | 182 | 180 | 2 | 0.989010989010989 | 1 |  |
| Subject.1/Hop/HopS1.avi | complete | 182 | 182 | 182 | 0 | 1.0 | 0 |  |
| Subject.1/Kneel/KneelS1.avi | complete | 219 | 219 | 219 | 0 | 1.0 | 0 |  |
| Subject.1/Pick up object/PickupobjectS1.avi | complete | 117 | 117 | 87 | 30 | 0.7435897435897436 | 22 |  |
| Subject.1/Sit down/SitDownS1.avi | complete | 187 | 187 | 187 | 0 | 1.0 | 0 |  |
| Subject.1/Walk/WalkS1.avi | complete | 242 | 242 | 214 | 28 | 0.8842975206611571 | 17 |  |
| Subject.2/Fall backwards/FallBackwardsS2.avi | complete | 119 | 119 | 119 | 0 | 1.0 | 0 |  |
| Subject.2/Fall forward/FallForwardS2.avi | complete | 106 | 106 | 83 | 23 | 0.7830188679245284 | 22 |  |
| Subject.2/Fall left/FallLeftS2.avi | complete | 109 | 109 | 109 | 0 | 1.0 | 0 |  |
| Subject.2/Fall right/FallRightS2.avi | complete | 146 | 146 | 146 | 0 | 1.0 | 0 |  |
| Subject.2/Fall sitting/FallSittingS2.avi | complete | 157 | 157 | 156 | 1 | 0.9936305732484076 | 1 |  |
| Subject.2/Hop/HopS2.avi | complete | 171 | 171 | 171 | 0 | 1.0 | 0 |  |
| Subject.2/Kneel/KneelS2.avi | complete | 130 | 130 | 130 | 0 | 1.0 | 0 |  |
| Subject.2/Pick up object/PickupobjectS2.avi | complete | 166 | 166 | 156 | 10 | 0.9397590361445783 | 9 |  |
| Subject.2/Sit down/SitDownS2.avi | complete | 112 | 112 | 112 | 0 | 1.0 | 0 |  |
| Subject.2/Walk/WalkS2.avi | complete | 244 | 244 | 209 | 35 | 0.8565573770491803 | 20 |  |
| Subject.3/Fall backwards/FallBackwardsS3.avi | complete | 166 | 166 | 166 | 0 | 1.0 | 0 |  |
| Subject.3/Fall forward/FallForwardS3.avi | complete | 153 | 153 | 49 | 104 | 0.3202614379084967 | 67 |  |
| Subject.3/Fall left/FallLeftS3.avi | complete | 171 | 171 | 147 | 24 | 0.8596491228070176 | 12 |  |
| Subject.3/Fall right/FallRightS3.avi | complete | 220 | 220 | 220 | 0 | 1.0 | 0 |  |
| Subject.3/Fall sitting/FallSittingS3.avi | complete | 202 | 202 | 199 | 3 | 0.9851485148514851 | 2 |  |
| Subject.3/Hop/HopS3.avi | complete | 162 | 162 | 121 | 41 | 0.7469135802469136 | 14 |  |
| Subject.3/Kneel/KneelS3.avi | complete | 213 | 213 | 104 | 109 | 0.48826291079812206 | 58 |  |
| Subject.3/Pick up object/PickupobjectS3.avi | complete | 186 | 186 | 132 | 54 | 0.7096774193548387 | 18 |  |
| Subject.3/Sit down/SitDownS3.avi | complete | 261 | 261 | 216 | 45 | 0.8275862068965517 | 21 |  |
| Subject.3/Walk/WalkS3.avi | complete | 176 | 176 | 113 | 63 | 0.6420454545454546 | 62 |  |
| Subject.4/Fall backwards/FallBackwardsS4.avi | complete | 192 | 192 | 182 | 10 | 0.9479166666666666 | 9 |  |
| Subject.4/Fall forward/FallForwardS4.avi | complete | 168 | 168 | 148 | 20 | 0.8809523809523809 | 14 |  |
| Subject.4/Fall left/FallLeftS4.avi | complete | 162 | 162 | 160 | 2 | 0.9876543209876543 | 1 |  |
| Subject.4/Fall right/FallRightS4.avi | complete | 150 | 150 | 150 | 0 | 1.0 | 0 |  |
| Subject.4/Fall sitting/FallSittingS4.avi | complete | 198 | 198 | 198 | 0 | 1.0 | 0 |  |
| Subject.4/Hop/HopS4.avi | complete | 144 | 144 | 135 | 9 | 0.9375 | 9 |  |
| Subject.4/Kneel/KneelS4.avi | complete | 222 | 222 | 222 | 0 | 1.0 | 0 |  |
| Subject.4/Pick up object/PickupobjectS4.avi | complete | 179 | 179 | 175 | 4 | 0.9776536312849162 | 2 |  |
| Subject.4/Sit down/SitDownS4.avi | complete | 199 | 199 | 199 | 0 | 1.0 | 0 |  |
| Subject.4/Walk/WalkS4.avi | complete | 240 | 240 | 222 | 18 | 0.925 | 18 |  |
| Subject.5/Fall backwards/FallBackwardsS5.avi | complete | 214 | 214 | 214 | 0 | 1.0 | 0 |  |
| Subject.5/Fall forward/FallForwardS5.avi | complete | 200 | 200 | 148 | 52 | 0.74 | 48 |  |
| Subject.5/Fall left/FallLeftS5.avi | complete | 247 | 247 | 222 | 25 | 0.8987854251012146 | 15 |  |
| Subject.5/Fall right/FallRightS5.avi | complete | 219 | 219 | 174 | 45 | 0.7945205479452054 | 45 |  |
| Subject.5/Fall sitting/FallSittingS5.avi | complete | 229 | 229 | 224 | 5 | 0.9781659388646288 | 5 |  |
| Subject.5/Hop/HopS5.avi | complete | 122 | 122 | 99 | 23 | 0.8114754098360656 | 23 |  |
| Subject.5/Kneel/KneelS5.avi | complete | 208 | 208 | 203 | 5 | 0.9759615384615384 | 5 |  |
| Subject.5/Pick up object/PickupobjectS5.avi | complete | 271 | 271 | 132 | 139 | 0.4870848708487085 | 73 |  |
| Subject.5/Sit down/SitDownS5.avi | complete | 260 | 260 | 259 | 1 | 0.9961538461538462 | 1 |  |
| Subject.5/Walk/WalkS5.avi | complete | 226 | 226 | 192 | 34 | 0.8495575221238938 | 34 |  |
| Subject.6/Fall backwards/FallBackwardsS6.avi | complete | 234 | 234 | 234 | 0 | 1.0 | 0 |  |
| Subject.6/Fall forward/FallForwardS6.avi | complete | 202 | 202 | 132 | 70 | 0.6534653465346535 | 62 |  |
| Subject.6/Fall left/FallLeftS6.avi | complete | 229 | 229 | 43 | 186 | 0.18777292576419213 | 185 |  |
| Subject.6/Fall right/FallRightS6.avi | complete | 229 | 229 | 229 | 0 | 1.0 | 0 |  |
| Subject.6/Fall sitting/FallSittingS6.avi | complete | 248 | 248 | 230 | 18 | 0.9274193548387096 | 18 |  |
| Subject.6/Hop/HopS6.avi | complete | 162 | 162 | 135 | 27 | 0.8333333333333334 | 27 |  |
| Subject.6/Kneel/KneelS6.avi | complete | 189 | 189 | 181 | 8 | 0.9576719576719577 | 5 |  |
| Subject.6/Pick up object/PickupobjectS6.avi | complete | 202 | 202 | 108 | 94 | 0.5346534653465347 | 94 |  |
| Subject.6/Sit down/SitDownS6.avi | complete | 189 | 189 | 189 | 0 | 1.0 | 0 |  |
| Subject.6/Walk/WalkS6.avi | complete | 266 | 266 | 265 | 1 | 0.9962406015037594 | 1 |  |
| Subject.7/Fall backwards/FallBackwardsS7.avi | complete | 252 | 252 | 252 | 0 | 1.0 | 0 |  |
| Subject.7/Fall forward/FallForwardS7.avi | complete | 162 | 162 | 157 | 5 | 0.9691358024691358 | 1 |  |
| Subject.7/Fall left/FallLeftS7.avi | complete | 212 | 212 | 207 | 5 | 0.9764150943396226 | 4 |  |
| Subject.7/Fall right/FallRightS7.avi | complete | 166 | 166 | 166 | 0 | 1.0 | 0 |  |
| Subject.7/Fall sitting/FallSittingS7.avi | complete | 207 | 207 | 203 | 4 | 0.9806763285024155 | 3 |  |
| Subject.7/Hop/HopS7.avi | complete | 198 | 198 | 184 | 14 | 0.9292929292929293 | 11 |  |
| Subject.7/Kneel/KneelS7.avi | complete | 202 | 202 | 156 | 46 | 0.7722772277227723 | 13 |  |
| Subject.7/Pick up object/PickupobjectS7.avi | complete | 153 | 153 | 152 | 1 | 0.9934640522875817 | 1 |  |
| Subject.7/Sit down/SitDownS7.avi | complete | 193 | 193 | 191 | 2 | 0.9896373056994818 | 1 |  |
| Subject.7/Walk/WalkS7.avi | complete | 193 | 193 | 191 | 2 | 0.9896373056994818 | 1 |  |
| Subject.8/Fall backwards/FallBackwardsS8.avi | complete | 212 | 212 | 211 | 1 | 0.9952830188679245 | 1 |  |
| Subject.8/Fall forward/FallForwardS8.avi | complete | 122 | 122 | 0 | 122 | 0.0 | 122 |  |
| Subject.8/Fall left/FallLeftS8.avi | complete | 202 | 202 | 73 | 129 | 0.3613861386138614 | 129 |  |
| Subject.8/Fall right/FallRightS8.avi | complete | 229 | 229 | 34 | 195 | 0.14847161572052403 | 152 |  |
| Subject.8/Fall sitting/FallSittingS8.avi | complete | 234 | 234 | 201 | 33 | 0.8589743589743589 | 32 |  |
| Subject.8/Hop/HopS8.avi | complete | 166 | 166 | 156 | 10 | 0.9397590361445783 | 10 |  |
| Subject.8/Kneel/KneelS8.avi | complete | 256 | 256 | 181 | 75 | 0.70703125 | 72 |  |
| Subject.8/Pick up object/PickupobjectS8.avi | complete | 266 | 266 | 245 | 21 | 0.9210526315789473 | 21 |  |
| Subject.8/Sit down/SitDownS8.avi | complete | 212 | 212 | 174 | 38 | 0.8207547169811321 | 38 |  |
| Subject.8/Walk/WalkS8.avi | complete | 216 | 216 | 216 | 0 | 1.0 | 0 |  |
| Subject.9/Fall backwards/FallBackwardsS9.avi | complete | 252 | 252 | 252 | 0 | 1.0 | 0 |  |
| Subject.9/Fall forward/FallForwardS9.avi | complete | 148 | 148 | 128 | 20 | 0.8648648648648649 | 19 |  |
| Subject.9/Fall left/FallLeftS9.avi | complete | 207 | 207 | 104 | 103 | 0.5024154589371981 | 15 |  |
| Subject.9/Fall right/FallRightS9.avi | complete | 189 | 189 | 180 | 9 | 0.9523809523809523 | 9 |  |
| Subject.9/Fall sitting/FallSittingS9.avi | complete | 216 | 216 | 215 | 1 | 0.9953703703703703 | 1 |  |
| Subject.9/Hop/HopS9.avi | complete | 212 | 212 | 210 | 2 | 0.9905660377358491 | 2 |  |
| Subject.9/Kneel/KneelS9.avi | complete | 216 | 216 | 203 | 13 | 0.9398148148148148 | 7 |  |
| Subject.9/Pick up object/PickupobjectS9.avi | complete | 220 | 220 | 212 | 8 | 0.9636363636363636 | 8 |  |
| Subject.9/Sit down/SitDownS9.avi | complete | 238 | 238 | 208 | 30 | 0.8739495798319328 | 13 |  |
| Subject.9/Walk/WalkS9.avi | complete | 234 | 234 | 199 | 35 | 0.8504273504273504 | 35 |  |
| Subject.10/Fall backwards/FallBackwardsS10.avi | complete | 274 | 274 | 266 | 8 | 0.9708029197080292 | 8 |  |
| Subject.10/Fall forward/FallForwardS10.avi | complete | 243 | 243 | 185 | 58 | 0.7613168724279835 | 21 |  |
| Subject.10/Fall left/FallLeftS10.avi | complete | 212 | 212 | 197 | 15 | 0.9292452830188679 | 12 |  |
| Subject.10/Fall right/FallRightS10.avi | complete | 202 | 202 | 202 | 0 | 1.0 | 0 |  |
| Subject.10/Fall sitting/FallSittingS10.avi | complete | 288 | 288 | 187 | 101 | 0.6493055555555556 | 47 |  |
| Subject.10/Hop/HopS10.avi | complete | 248 | 248 | 246 | 2 | 0.9919354838709677 | 2 |  |
| Subject.10/Kneel/KneelS10.avi | complete | 261 | 261 | 182 | 79 | 0.6973180076628352 | 21 |  |
| Subject.10/Pick up object/PickupobjectS10.avi | complete | 292 | 292 | 214 | 78 | 0.7328767123287672 | 34 |  |
| Subject.10/Sit down/SitDownS10.avi | complete | 234 | 234 | 220 | 14 | 0.9401709401709402 | 4 |  |
| Subject.10/Walk/WalkS10.avi | complete | 224 | 224 | 210 | 14 | 0.9375 | 4 |  |
