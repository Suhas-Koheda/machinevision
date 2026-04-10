import cv2
import numpy as np


def detect_speckle(image: np.ndarray) -> dict:
    """
    Detects speckle noise using local variance and multiplicative residual analysis.

    Speckle noise is multiplicative, so we compare each pixel against its local mean
    and measure how strong the normalized residual is inside local neighborhoods.
    """
    image_float = image.astype(np.float32)

    local_mean = cv2.blur(image_float, (7, 7))
    local_sq_mean = cv2.blur(image_float * image_float, (7, 7))
    local_variance_map = np.maximum(local_sq_mean - (local_mean * local_mean), 0.0)

    valid_mask = local_mean > 15.0
    if not np.any(valid_mask):
        valid_mask = np.ones_like(image_float, dtype=bool)

    normalized_residual = np.abs(image_float - local_mean) / (local_mean + 1.0)
    coefficient_of_variation = np.sqrt(local_variance_map) / (local_mean + 1.0)

    average_local_variance = float(np.mean(local_variance_map[valid_mask]))
    speckle_index = float(np.mean(normalized_residual[valid_mask]))
    average_coefficient = float(np.mean(coefficient_of_variation[valid_mask]))

    normalized_coefficient = min(average_coefficient / 0.35, 1.0)
    normalized_variance = min(average_local_variance / 1500.0, 1.0)

    if average_coefficient < 0.03 and speckle_index < 0.03:
        confidence = 0.0
        noise_ratio = 0.0
    else:
        confidence = min(
            0.55 * normalized_coefficient
            + 0.30 * min(speckle_index / 0.25, 1.0)
            + 0.15 * normalized_variance,
            1.0,
        )
        noise_ratio = min(max((average_coefficient - 0.03) / 0.22, 0.0), 1.0)

    return {
        "confidence": float(confidence),
        "noise_ratio": float(noise_ratio),
        "local_variance": float(average_local_variance),
        "speckle_index": float(speckle_index),
        "coefficient_of_variation": float(average_coefficient),
    }
