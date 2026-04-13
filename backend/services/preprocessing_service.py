import cv2
import numpy as np

def estimate_noise(frame: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Estimates noise level using Laplacian variance.
    Returns: score (float), noise_heatmap (np.ndarray)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Noise Score
    score = laplacian.var()
    
    # Heatmap: Normalize laplacian and apply color mapping
    norm_laplacian = cv2.normalize(np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm_laplacian, cv2.COLORMAP_JET)
    
    return float(score), heatmap

def apply_adaptive_filter(frame: np.ndarray, noise_score: float) -> np.ndarray:
    """
    Apply filtering based on noise score.
    """
    if noise_score > 500: # Threshold for 'high' noise
        return cv2.GaussianBlur(frame, (5, 5), 0)
    elif noise_score > 200:
        return cv2.medianBlur(frame, 3)
    return frame
