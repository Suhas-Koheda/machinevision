import cv2
import numpy as np


def select_kernel_size(noise_ratio: float) -> int:
    """
    Selects an odd window size based on the estimated speckle intensity.
    """
    if noise_ratio < 0.10:
        return 3
    if noise_ratio < 0.25:
        return 5
    return 7


def apply_lee_filter(image: np.ndarray, noise_ratio: float) -> np.ndarray:
    """
    Applies the Lee filter for multiplicative speckle noise reduction.
    """
    kernel_size = select_kernel_size(noise_ratio)
    image_float = image.astype(np.float32)

    local_mean = cv2.blur(image_float, (kernel_size, kernel_size))
    local_sq_mean = cv2.blur(image_float * image_float, (kernel_size, kernel_size))
    local_variance = np.maximum(local_sq_mean - (local_mean * local_mean), 0.0)
    noise_variance = float(np.mean(local_variance))

    weight = local_variance / (local_variance + noise_variance + 1e-6)
    filtered = local_mean + weight * (image_float - local_mean)

    return np.clip(filtered, 0, 255).astype(np.uint8)


def apply_frost_filter(image: np.ndarray, noise_ratio: float) -> np.ndarray:
    """
    Applies a Frost filter with exponential distance weighting.
    """
    kernel_size = select_kernel_size(noise_ratio)
    radius = kernel_size // 2
    image_float = image.astype(np.float32)
    padded = np.pad(image_float, radius, mode="reflect")
    filtered = np.zeros_like(image_float)

    y_coords, x_coords = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    distance = np.sqrt((x_coords * x_coords) + (y_coords * y_coords)).astype(np.float32)
    alpha = 1.0 + (3.0 * noise_ratio)

    for row in range(image_float.shape[0]):
        for col in range(image_float.shape[1]):
            window = padded[row : row + kernel_size, col : col + kernel_size]
            local_mean = float(np.mean(window))
            local_variance = float(np.var(window))
            damping = alpha * (local_variance / ((local_mean * local_mean) + 1e-6))
            weights = np.exp(-damping * distance)
            weights_sum = np.sum(weights)

            if weights_sum == 0:
                filtered[row, col] = image_float[row, col]
            else:
                filtered[row, col] = float(np.sum(window * weights) / weights_sum)

    return np.clip(filtered, 0, 255).astype(np.uint8)


def filter_speckle(image: np.ndarray, noise_ratio: float, method: str = "lee") -> np.ndarray:
    """
    Dispatches speckle denoising to Lee or Frost filters.
    """
    normalized_method = method.lower()
    if normalized_method == "frost":
        return apply_frost_filter(image, noise_ratio)
    return apply_lee_filter(image, noise_ratio)
