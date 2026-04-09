import numpy as np
import math

def calculate_mse(original_image: np.ndarray, denoised_image: np.ndarray) -> float:
    """
    Computes the Mean Squared Error (MSE) between original and denoised images.
    
    MSE measures the average squared difference between pixel values.
    - Lower MSE = better denoising quality
    - MSE = 0 = identical images (unrealistic in practice)
    
    Args:
        original_image: Original grayscale image
        denoised_image: Denoised grayscale image
    
    Returns:
        MSE value as float
    """
    # Cast to float32 for precision to handle large squared sums correctly
    diff = original_image.astype(np.float32) - denoised_image.astype(np.float32)
    mse_value = np.mean(diff ** 2)
    return float(mse_value)

def calculate_psnr(mse_value: float) -> float:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) from MSE.
    
    PSNR measures the ratio of maximum possible signal power to degrading noise power.
    - Higher PSNR = better image quality (typical range: 20-50 dB)
    - PSNR > 30 dB = acceptable quality for most applications
    - PSNR > 40 dB = excellent quality
    
    Formula: PSNR = 20 * log10(MAX_PIXEL / sqrt(MSE))
    where MAX_PIXEL = 255 for 8-bit images
    
    Args:
        mse_value: Mean Squared Error value
    
    Returns:
        PSNR value in dB as float
    """
    if mse_value == 0:
        # Avoid misleading infinite/perfect scores
        # If images are identical, return a capped realistic value
        return 50.0
    
    # Standard PSNR formula for 8-bit images
    max_pixel_value = 255.0
    psnr = 20 * math.log10(max_pixel_value / math.sqrt(mse_value))
    return float(psnr)

def calculate_metrics(original_image: np.ndarray, denoised_image: np.ndarray) -> dict:
    """
    Calculates both MSE and PSNR metrics for image quality assessment.
    
    Args:
        original_image: Original grayscale image
        denoised_image: Denoised grayscale image
    
    Returns:
        dict: {
            "mse": float,
            "psnr": float
        }
    """
    mse = calculate_mse(original_image, denoised_image)
    psnr = calculate_psnr(mse)
    
    return {
        "mse": mse,
        "psnr": psnr
    }
