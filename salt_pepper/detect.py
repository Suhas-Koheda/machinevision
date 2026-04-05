import numpy as np

def detect_salt_pepper(image: np.ndarray) -> dict:
    """
    Detects Salt & Pepper noise by checking for pixels with extreme values 0 (pepper) and 255 (salt).
    Returns a dictionary with the confidence score and the salt/pepper ratio.
    """
    # Count pixels that are near-zero or near-255 (handles slight compression)
    pepper_count = np.sum(image <= 2)
    salt_count = np.sum(image >= 253)
    
    total_pixels = image.size
    noise_pixels = pepper_count + salt_count
    
    # Simple ratio of noisy pixels to total pixels
    noise_ratio = noise_pixels / total_pixels
    
    # Use the ratio as confidence. If > 2% of pixels are impulsive, it's likely S&P.
    confidence = min(noise_ratio * 20, 1.0)
    
    return {
        "confidence": float(confidence),
        "salt_count": int(salt_count),
        "pepper_count": int(pepper_count),
        "noise_ratio": float(noise_ratio)
    }
