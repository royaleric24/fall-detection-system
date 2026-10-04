"""Deterministic Stage 3.3 image-plane normalization; no fitted statistics."""

from dataclasses import dataclass

import numpy as np

# Same MediaPipe indices as characterize_pose.LANDMARKS; no MediaPipe dependency.
LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 11, 12, 23, 24
FEATURE_DIM = 33 * 4  # Per joint: normalized x, y, original visibility, validity.


@dataclass(frozen=True)
class PreprocessingConfig:
    min_torso_scale: float = 0.001  # Image-height units, engineering floor, not tuned.

    def __post_init__(self) -> None:
        if not np.isfinite(self.min_torso_scale) or self.min_torso_scale <= 0:
            raise ValueError("Minimum torso scale must be positive and finite")


def normalize_pose_sequence(landmarks: np.ndarray, pose_detected: np.ndarray, *,
                            image_aspect: float,
                            config: PreprocessingConfig = PreprocessingConfig()
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Return float32 [T,132] and bool [T,33], without changing raw input.

    x is multiplied by width/height before geometry to give isotropic image-height
    units. z is excluded: MediaPipe z is not calibrated metric depth. Finite
    returned joints remain usable regardless of visibility; confidence is retained.
    One available hip/shoulder substitutes for that pair's midpoint. If no hip,
    no shoulder, or a degenerate torso exists, the whole frame is zero/invalid.
    Missing frames and unavailable joints are zero-filled; no interpolation.
    """
    raw = np.asarray(landmarks)
    detected = np.asarray(pose_detected)
    if (raw.ndim != 3 or raw.shape[1:] != (33, 4) or raw.shape[0] == 0
            or detected.shape != (raw.shape[0],) or detected.dtype != np.dtype(bool)):
        raise ValueError("Expected nonempty [T,33,4] landmarks and bool [T] detected")
    if not np.isfinite(image_aspect) or image_aspect <= 0:
        raise ValueError("Expected positive finite image aspect")
    xy = raw[:, :, :2].astype(np.float64, copy=True)
    xy[:, :, 0] *= image_aspect
    usable = np.isfinite(xy).all(axis=2) & np.isfinite(raw[:, :, 3]) & detected[:, None]
    result = np.zeros((len(raw), 33, 4), dtype=np.float32)
    mask = np.zeros((len(raw), 33), dtype=bool)
    for t in range(len(raw)):
        hips = [i for i in (LEFT_HIP, RIGHT_HIP) if usable[t, i]]
        shoulders = [i for i in (LEFT_SHOULDER, RIGHT_SHOULDER) if usable[t, i]]
        if not hips or not shoulders:
            continue
        center = xy[t, hips].mean(axis=0)
        scale = np.linalg.norm(xy[t, shoulders].mean(axis=0) - center)
        if not np.isfinite(scale) or scale < config.min_torso_scale:
            continue
        indices = np.flatnonzero(usable[t])
        with np.errstate(over="ignore", invalid="ignore"):
            normalized = ((xy[t, indices] - center) / scale).astype(np.float32)
        finite = np.isfinite(normalized).all(axis=1)
        indices, normalized = indices[finite], normalized[finite]
        result[t, indices, :2] = normalized
        result[t, indices, 2] = raw[t, indices, 3]
        result[t, indices, 3] = 1
        mask[t, indices] = True
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite preprocessing output")
    return result.reshape(len(raw), FEATURE_DIM), mask
