import numpy as np
import math

def calculate_mse(original_image: np.ndarray, denoised_image: np.ndarray) -> float:
    """
    Computes the Mean Squared Error (MSE) between two images.
    Lower MSE implies better quality.
    """
    # Cast to float for precision and to handle subtraction correctly.
    diff = original_image.astype(np.float32) - denoised_image.astype(np.float32)
    error = np.mean(diff ** 2)
    return float(error)

def calculate_psnr(mseValue: float) -> float:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) from the MSE.
    Higher PSNR implies better filter performance.
    """
    if mseValue == 0:
        return 100.0  # Theoretically infinite, but 100 serves as a solid peak.
    
    # 255 is the maximum intensity for an 8-bit image.
    max_pixel_value = 255.0
    psnr = 20 * math.log10(max_pixel_value / math.sqrt(mseValue))
    return float(psnr)
