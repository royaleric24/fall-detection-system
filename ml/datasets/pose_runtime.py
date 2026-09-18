"""Shared MediaPipe VIDEO configuration and missing-interval accounting.

No dependency imports or experiment execution occur on import.
"""

from pathlib import Path
from typing import Any

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
MODEL_SHA256 = "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1"


def pose_options(model: Path, mp: Any, confidence: float = .5) -> Any:
    """Shared Stage 1.3/2.2 VIDEO/CPU setup; callers create a fresh tracker."""
    return mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model),
                                         delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=confidence, min_pose_presence_confidence=confidence,
        min_tracking_confidence=confidence, output_segmentation_masks=False)


def missing_runs(indices: list[int]) -> list[list[int]]:
    """Inclusive zero-based missing-frame intervals; no skeletons retained."""
    runs: list[list[int]] = []
    for index in indices:
        if runs and index == runs[-1][1] + 1:
            runs[-1][1] = index
        else:
            runs.append([index, index])
    return runs
