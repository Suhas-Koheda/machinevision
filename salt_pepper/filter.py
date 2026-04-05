import cv2
import numpy as np

def select_kernel_size(noise_ratio: float) -> int:
    """
    Selects the optimal kernel size based on noise intensity.
    - noise_ratio < 0.02 -> kernel = 3
    - 0.02 <= noise_ratio < 0.05 -> kernel = 5
    - noise_ratio >= 0.05 -> kernel = 7
    """
    if noise_ratio < 0.02:
        return 3
    elif 0.02 <= noise_ratio < 0.05:
        return 5
    else:
        return 7

def filter_salt_pepper(image: np.ndarray, noise_ratio: float) -> np.ndarray:
    """
    Applies adaptive Median Filter using OpenCV based on noise_ratio.
    Ensures kernel size is always odd (3, 5, or 7).
    """
    kernel_size = select_kernel_size(noise_ratio)
    # Median blur is highly effective for salt-and-pepper noise.
    return cv2.medianBlur(image, kernel_size)
