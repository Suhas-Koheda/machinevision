import cv2
import numpy as np

def denoise(image: np.ndarray, strength: int = 3) -> np.ndarray:
    """
    Modular denoising hook with adjustable strength.
    Uses a lighter denoising to preserve details.
    """
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    else:
        return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)