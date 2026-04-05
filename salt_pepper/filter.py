import cv2
import numpy as np

def filter_salt_pepper(image: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """
    Applies the Median Filter using OpenCV for Salt & Pepper noise removal.
    - kernel_size: The size of the sliding window for the filter.
    """
    # Median blur is highly effective for impulsive (salt-and-pepper) noise.
    return cv2.medianBlur(image, kernel_size)
