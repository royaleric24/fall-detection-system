"""Stage 3.3 causal missing handling and optional XYZ torso normalization."""

from dataclasses import dataclass

import numpy as np

# Existing MediaPipe mapping from characterize_pose.LANDMARKS.
LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 11, 12, 23, 24
FEATURE_DIM = 33 * 4  # Per joint: x, y, z, visibility. Masks are separate.


@dataclass(frozen=True)
class PreprocessingConfig:
    normalize_pose: bool = True
    max_forward_fill_frames: int = 5
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if type(self.normalize_pose) is not bool:
            raise ValueError("normalize_pose must be boolean")
        if type(self.max_forward_fill_frames) is not int or self.max_forward_fill_frames < 0:
            raise ValueError("Forward-fill limit must be a nonnegative integer")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("Scale epsilon must be positive and finite")


def handle_missing_pose(landmarks: np.ndarray, pose_detected: np.ndarray, *,
                        max_forward_fill_frames: int = 5
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Copy current poses, fill <= limit frames from last observed pose, else zero.

    Returns handled [T,33,4], available [T], imputed [T]. Leading gaps and frames
    after the limit are zero; filled poses do not reset the gap counter. No future
    frame is used. Caller inputs are never modified. Partial nonfinite detected
    joints remain unavailable to the transform; official raw schema rejects them.
    """
    raw, detected = np.asarray(landmarks), np.asarray(pose_detected)
    if (raw.ndim != 3 or raw.shape[1:] != (33, 4) or len(raw) == 0
            or detected.shape != (len(raw),) or detected.dtype != np.dtype(bool)):
        raise ValueError("Expected nonempty [T,33,4] landmarks and bool [T] detected")
    if type(max_forward_fill_frames) is not int or max_forward_fill_frames < 0:
        raise ValueError("Forward-fill limit must be a nonnegative integer")
    handled = np.zeros(raw.shape, dtype=np.float64)
    available = np.zeros(len(raw), dtype=bool)
    imputed = np.zeros(len(raw), dtype=bool)
    last = None
    gap = 0
    for t in range(len(raw)):
        if detected[t]:
            handled[t] = raw[t]
            available[t] = True
            last = raw[t].copy()
            gap = 0
        else:
            gap += 1
            if last is not None and gap <= max_forward_fill_frames:
                handled[t] = last
                available[t] = True
                imputed[t] = True
    return handled, available, imputed


def normalize_pose_sequence(landmarks: np.ndarray, pose_detected: np.ndarray, *,
                            config: PreprocessingConfig = PreprocessingConfig()
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Return float32 [T,132] XYZ/visibility and usable bool [T,33].

    Causal fill precedes geometry. Coordinates retain MediaPipe's original units
    and axes; this is relative XYZ, not calibrated metric 3-D. If enabled, subtract
    XYZ hip midpoint and divide by max(XYZ torso distance, epsilon). A single
    available hip/shoulder substitutes for its midpoint. Missing reference pairs
    produce a zero/invalid frame. Finite low-visibility joints remain observed;
    visibility is clipped to [0,1], never geometrically normalized. No added features.
    """
    handled, available, _ = handle_missing_pose(
        landmarks, pose_detected, max_forward_fill_frames=config.max_forward_fill_frames)
    usable = np.isfinite(handled).all(axis=2) & available[:, None]
    result = np.zeros(handled.shape, dtype=np.float32)
    mask = np.zeros(usable.shape, dtype=bool)
    for t in range(len(handled)):
        indices = np.flatnonzero(usable[t])
        if not len(indices):
            continue
        xyz = handled[t, indices, :3]
        if config.normalize_pose:
            hips = [i for i in (LEFT_HIP, RIGHT_HIP) if usable[t, i]]
            shoulders = [i for i in (LEFT_SHOULDER, RIGHT_SHOULDER) if usable[t, i]]
            if not hips or not shoulders:
                continue
            center = handled[t, hips, :3].mean(axis=0)
            scale = np.linalg.norm(handled[t, shoulders, :3].mean(axis=0) - center)
            if not np.isfinite(scale):
                continue
            xyz = (xyz - center) / max(float(scale), config.epsilon)
        with np.errstate(over="ignore", invalid="ignore"):
            xyz = xyz.astype(np.float32)
        finite = np.isfinite(xyz).all(axis=1)
        indices, xyz = indices[finite], xyz[finite]
        result[t, indices, :3] = xyz
        result[t, indices, 3] = np.clip(handled[t, indices, 3], 0, 1)
        mask[t, indices] = True
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite preprocessing output")
    return result.reshape(len(result), FEATURE_DIM), mask
