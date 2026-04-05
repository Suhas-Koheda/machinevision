import numpy as np
import math

def calculate_mse(original_image: np.ndarray, denoised_image: np.ndarray) -> float:
    """
    Computes the Mean Squared Error (MSE) with float precision.
    Lower MSE implies better quality.
    """
    # Cast to float for precision to handle large squared sums correctly.
    diff = original_image.astype(np.float32) - denoised_image.astype(np.float32)
    mseValue = np.mean(diff ** 2)
    return float(mseValue)

def calculate_psnr(mseValue: float) -> float:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) from the MSE.
    - If MSE == 0: Returns a capped realistic value (50.0).
    - Uses the formula: PSNR = 20 * log10(255 / sqrt(MSE))
    """
    if mseValue == 0:
        # Avoid misleading infinite/perfect scores.
        return 50.0 
    
    # Standard PSNR formula
    max_pixel_value = 255.0
    psnr = 20 * math.log10(max_pixel_value / math.sqrt(mseValue))
    return float(psnr)
