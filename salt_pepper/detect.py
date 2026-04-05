import numpy as np

def detect_salt_pepper(image: np.ndarray) -> dict:
    """
    Detects Salt & Pepper noise by checking for pixels with extreme values 0 (pepper) and 255 (salt).
    Returns a dictionary with the confidence score and the salt/pepper ratio.
    """
    # Count pixels that are near-zero or near-255 (handles slight compression/artifacts)
    pepper_count = np.sum(image <= 3)
    salt_count = np.sum(image >= 252)
    
    total_pixels = image.size
    noise_ratio = (pepper_count + salt_count) / total_pixels
    
    # If more than 0.5% of pixels are noisy, it's likely Salt & Pepper.
    # We use a multiplier to ensure it beats the 0.1 Gaussian placeholder.
    confidence = min(noise_ratio * 20, 1.0)
    
    return {
        "confidence": float(confidence),
        "salt_count": int(salt_count),
        "pepper_count": int(pepper_count),
        "noise_ratio": float(noise_ratio)
    }
