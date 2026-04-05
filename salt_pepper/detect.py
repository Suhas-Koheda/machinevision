import numpy as np

def detect_salt_pepper(image: np.ndarray) -> dict:
    """
    Advanced Salt & Pepper noise detection.
    Uses thresholds (pepper <= 10, salt >= 245) for robust identification.
    """
    # Threshold-based pixel counting
    pepper_pixels = np.sum(image <= 10)
    salt_pixels = np.sum(image >= 245)
    
    total_pixels = image.size
    noise_ratio = (pepper_pixels + salt_pixels) / total_pixels
    
    # Adaptive confidence logic
    # If noise_ratio < 1%, it's likely not impulsive noise
    if noise_ratio < 0.01:
        confidence = 0.0
    else:
        # Scale confidence based on intensity of noise (maxed at 1.0)
        confidence = min(noise_ratio * 10, 1.0)
        
    return {
        "confidence": float(confidence),
        "noise_ratio": float(noise_ratio),
        "pepper_count": int(pepper_pixels),
        "salt_count": int(salt_pixels)
    }
