"""
Frame Difference Service.
Determines whether a new frame is significantly different from the last one
to avoid processing redundant/identical frames.
"""
import cv2
import numpy as np

_last_frame_gray: np.ndarray | None = None
DIFF_THRESHOLD = 8.0       # mean absolute pixel difference threshold
MIN_CHANGED_RATIO = 0.03   # at least 3% of pixels must change


def is_significant(frame: np.ndarray) -> bool:
    """
    Returns True if this frame is meaningfully different from the previous one.
    Side-effect: updates internal state.
    """
    global _last_frame_gray

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (320, 240))  # normalise size for comparison

    if _last_frame_gray is None:
        _last_frame_gray = gray
        return True

    diff = cv2.absdiff(gray, _last_frame_gray)
    mean_diff = float(np.mean(diff))
    changed_pixels = float(np.count_nonzero(diff > 20)) / diff.size

    if mean_diff > DIFF_THRESHOLD or changed_pixels > MIN_CHANGED_RATIO:
        _last_frame_gray = gray
        return True

    return False


def reset():
    """Reset stored frame (e.g. on new WebSocket connection)."""
    global _last_frame_gray
    _last_frame_gray = None
