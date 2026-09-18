"""Pure NumPy pose_raw_v1 encoding, validation and validated NPZ publication."""

import math
import os
import tempfile
from zipfile import BadZipFile
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "pose_raw_v1"
TIMESTAMP_METHOD = "derived_from_frame_index_and_source_fps"
ARRAY_DTYPES = {"frame_index": np.dtype("int32"), "timestamp_ms": np.dtype("int64"),
                "pose_detected": np.dtype("bool"), "landmarks": np.dtype("float32")}


class ExtractionError(ValueError):
    """Known extraction failure; never an ordinary no-pose result."""

    def __init__(self, error_type: str, message: str, frame_index: int | None = None):
        super().__init__(message)
        self.error_type = error_type
        self.frame_index = frame_index


def timestamp_ms(frame_index: int, source_fps: float) -> int:
    """Clip-relative time, not wall-clock time; Python round uses ties to even."""
    if (isinstance(frame_index, (bool, np.bool_)) or not isinstance(frame_index, (int, np.integer))
            or frame_index < 0 or not math.isfinite(source_fps) or source_fps <= 0):
        raise ValueError("Expected nonnegative frame index and positive finite FPS")
    return round(frame_index * 1000 / source_fps)


def encode_pose(result: Any) -> tuple[bool, np.ndarray]:
    """Only an empty pose list is ordinary missingness; retain low visibility."""
    try:
        poses = result.pose_landmarks
        if not isinstance(poses, (list, tuple)):
            raise ValueError("Expected a pose list")
        if len(poses) == 0:
            return False, np.full((33, 4), np.nan, dtype=np.float32)
        if len(poses) != 1 or len(poses[0]) != 33:
            raise ValueError("Expected exactly one pose with 33 landmarks")
        values = np.array([(lm.x, lm.y, lm.z, lm.visibility) for lm in poses[0]],
                          dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            stored = values.astype(np.float32)
        if stored.shape != (33, 4):
            raise ValueError("Expected scalar x/y/z/visibility values")
        if not np.isfinite(values).all() or not np.isfinite(stored).all():
            raise ValueError("Nonfinite landmark before or after float32 conversion")
        return True, stored
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ExtractionError("malformed_pose_result", str(exc)) from exc


def validate_arrays(arrays: dict[str, np.ndarray], source_fps: float,
                    expected_frame_count: int) -> None:
    """Reject extra keys, wrong types/shapes, discontinuities and hidden missingness."""
    if set(arrays) != set(ARRAY_DTYPES):
        raise ValueError("Expected exactly the four pose_raw_v1 arrays")
    if (isinstance(expected_frame_count, (bool, np.bool_))
            or not isinstance(expected_frame_count, (int, np.integer))
            or not 0 < expected_frame_count <= np.iinfo(np.int32).max):
        raise ValueError("Expected positive int32-representable frame count")
    for name, dtype in ARRAY_DTYPES.items():
        value = arrays[name]
        shape = (expected_frame_count, 33, 4) if name == "landmarks" else (expected_frame_count,)
        if not isinstance(value, np.ndarray) or value.dtype != dtype or value.shape != shape:
            raise ValueError(f"{name}: expected {shape} {dtype}")
    indices = np.arange(expected_frame_count, dtype=np.int32)
    times = np.array([timestamp_ms(int(i), source_fps) for i in indices], dtype=np.int64)
    if not np.array_equal(arrays["frame_index"], indices):
        raise ValueError("frame_index must be contiguous from zero")
    if not np.array_equal(arrays["timestamp_ms"], times) or np.any(np.diff(times) <= 0):
        raise ValueError("timestamp_ms must follow the frozen strictly increasing formula")
    detected, landmarks = arrays["pose_detected"], arrays["landmarks"]
    if not np.isfinite(landmarks[detected]).all():
        raise ValueError("Detected poses must contain only finite values")
    if not np.isnan(landmarks[~detected]).all():
        raise ValueError("Missing poses must contain only NaN values")


def load_validated(path: Path, source_fps: float,
                   expected_frame_count: int) -> dict[str, np.ndarray]:
    """Reload without pickle and apply the same schema checks as in memory."""
    try:
        # Own the file handle even if NumPy rejects a truncated ZIP on opening.
        with path.open("rb") as handle:
            archive = np.load(handle, allow_pickle=False)
            if not isinstance(archive, np.lib.npyio.NpzFile):
                raise ValueError("Expected an NPZ archive, not a standalone NPY array")
            with archive:
                if len(archive.files) != 4 or set(archive.files) != set(ARRAY_DTYPES):
                    raise ValueError("NPZ must contain exactly four unique array names")
                arrays = {name: archive[name] for name in archive.files}
    except (EOFError, BadZipFile) as exc:
        raise ValueError(f"Invalid NPZ archive: {exc}") from exc
    validate_arrays(arrays, source_fps, expected_frame_count)
    return arrays


def publish_npz(path: Path, arrays: dict[str, np.ndarray], source_fps: float,
                expected_frame_count: int) -> dict[str, np.ndarray]:
    """Validate → temporary NPZ → reload/validate → atomic no-overwrite publish."""
    validate_arrays(arrays, source_fps, expected_frame_count)
    temporary = None
    try:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing existing output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pose-", suffix=".npz",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
        reloaded = load_validated(temporary, source_fps, expected_frame_count)
        # Same-filesystem link is atomic and fails if another writer created path.
        os.link(temporary, path)
        return reloaded
    except (OSError, ValueError) as exc:
        raise ExtractionError("output_persistence_failure", str(exc)) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
