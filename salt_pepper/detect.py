import numpy as np

def detect_salt_pepper(image: np.ndarray) -> dict:
    """
    Detects Salt & Pepper noise by checking for pixels with extreme values 0 (pepper) and 255 (salt).
    Returns a dictionary with the confidence score and the salt/pepper ratio.
    """
    # Count pixels that are exactly 0 or 255 (as per strict requirements)
    pepper_count = np.sum(image == 0)
    salt_count = np.sum(image == 255)
    
    total_pixels = image.size
    noise_ratio = (pepper_count + salt_count) / total_pixels
    
    # Simple ratio of noisy pixels to total pixels as confidence
    confidence = min(noise_ratio * 10, 1.0)
    
    return {
        "confidence": float(confidence),
        "salt_count": int(salt_count),
        "pepper_count": int(pepper_count),
        "noise_ratio": float(noise_ratio)
    }
