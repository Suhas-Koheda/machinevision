import numpy as np
import cv2

def detect_gaussian(image: np.ndarray) -> dict:
    """
    Advanced Gaussian noise detection using statistical analysis.
    
    Gaussian noise is random noise with a normal (Gaussian) distribution.
    Detection approach:
    1. Analyze pixel mean and variance
    2. Check Laplacian variance (edge detection) for smoothness
    3. Compare with known noise thresholds
    
    Returns:
        dict: {
            "confidence": float (0.0-1.0),
            "noise_ratio": float (estimated),
            "mean": float (pixel mean),
            "variance": float (pixel variance),
            "laplacian_variance": float (edge strength)
        }
    """
    
    # 1. Statistical Analysis - Mean & Variance
    mean_pixel = np.mean(image)
    variance = np.var(image)
    
    # Standard deviation for normalized comparison
    std_dev = np.sqrt(variance)
    
    # 2. Laplacian Variance (measures edge strength/sharpness)
    # High Laplacian variance = sharp edges (less noise)
    # Low Laplacian variance = smooth image (more gaussian noise)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian_variance = np.var(laplacian)
    
    # 3. Histogram analysis - Gaussian noise affects histogram smoothness
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    histogram = histogram.astype(np.float32) / histogram.sum()
    
    # Entropy of histogram (higher = more noise)
    histogram_entropy = -np.sum(histogram * (np.log(histogram + 1e-10)))
    
    # 4. Confidence Calculation Logic
    # Gaussian noise characteristics:
    # - High variance (spread of pixel values)
    # - Low Laplacian variance (blurred edges)
    # - High entropy in histogram (uniform distribution)
    
    confidence = 0.0
    noise_ratio = 0.0
    
    # Normalize metrics for comparison
    normalized_variance = min(variance / 2500.0, 1.0)  # Normalize by typical max variance
    normalized_laplacian = 1.0 - min(laplacian_variance / 100.0, 1.0)  # Inverse: low is noise
    normalized_entropy = histogram_entropy / 8.0  # Max entropy for 256 bins
    
    # Weighted combination for confidence
    if variance > 100:  # Must have some variance
        # Higher variance + lower edge sharpness = likely Gaussian noise
        confidence = (
            0.4 * normalized_variance +
            0.3 * normalized_laplacian +
            0.3 * normalized_entropy
        )
        confidence = min(confidence, 1.0)
        
        # Estimate noise ratio from variance
        # Typical clean image variance: 100-500
        # Noisy image variance: 500-2500
        if variance > 500:
            noise_ratio = min((variance - 500) / 2000.0, 1.0)
        else:
            noise_ratio = 0.0
    
    return {
        "confidence": float(confidence),
        "noise_ratio": float(noise_ratio),
        "mean": float(mean_pixel),
        "variance": float(variance),
        "laplacian_variance": float(laplacian_variance),
        "entropy": float(histogram_entropy)
    }
