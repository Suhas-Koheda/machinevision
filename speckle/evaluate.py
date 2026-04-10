import math
import numpy as np


def calculate_mse(original_image: np.ndarray, denoised_image: np.ndarray) -> float:
    """
    Computes the Mean Squared Error (MSE) between original and denoised images.
    """
    diff = original_image.astype(np.float32) - denoised_image.astype(np.float32)
    return float(np.mean(diff ** 2))


def calculate_psnr(mse_value: float) -> float:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) from MSE.
    """
    if mse_value == 0:
        return 50.0

    return float(20 * math.log10(255.0 / math.sqrt(mse_value)))
