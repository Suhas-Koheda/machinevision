import cv2
import numpy as np

def select_gaussian_params(noise_ratio: float) -> tuple:
    """
    Selects optimal Gaussian Blur parameters based on noise intensity.
    
    Parameters returned: (kernel_size, sigma)
    - noise_ratio < 0.1 -> kernel = 3, sigma = 0.5 (light smoothing)
    - 0.1 <= noise_ratio < 0.3 -> kernel = 5, sigma = 1.0 (medium smoothing)
    - noise_ratio >= 0.3 -> kernel = 7, sigma = 1.5 (heavy smoothing)
    """
    if noise_ratio < 0.1:
        return 3, 0.5
    elif 0.1 <= noise_ratio < 0.3:
        return 5, 1.0
    else:
        return 7, 1.5

def select_bilateral_params(noise_ratio: float) -> tuple:
    """
    Selects optimal Bilateral Filter parameters based on noise intensity.
    
    Bilateral Filter: Reduces noise while preserving edges
    Parameters: (diameter, sigma_color, sigma_space)
    
    - noise_ratio < 0.1 -> d = 5, sc = 50, ss = 50 (light edge preservation)
    - 0.1 <= noise_ratio < 0.3 -> d = 7, sc = 75, ss = 75 (medium filtering)
    - noise_ratio >= 0.3 -> d = 9, sc = 100, ss = 100 (strong filtering)
    """
    if noise_ratio < 0.1:
        return 5, 50, 50
    elif 0.1 <= noise_ratio < 0.3:
        return 7, 75, 75
    else:
        return 9, 100, 100

def apply_gaussian_blur(image: np.ndarray, noise_ratio: float) -> np.ndarray:
    """
    Applies Gaussian Blur filter for Gaussian noise reduction.
    
    Gaussian Blur: Averages pixel values with a weighted kernel
    - Pro: Simple, effective for Gaussian noise
    - Con: Blurs edges uniformly
    
    Args:
        image: Input grayscale image
        noise_ratio: Estimated noise level (0.0-1.0)
    
    Returns:
        Denoised image with Gaussian Blur applied
    """
    kernel_size, sigma = select_gaussian_params(noise_ratio)
    # Ensure kernel size is odd
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return blurred

def apply_bilateral_filter(image: np.ndarray, noise_ratio: float) -> np.ndarray:
    """
    Applies Bilateral Filter for edge-preserving Gaussian noise reduction.
    
    Bilateral Filter: Averages pixels based on both intensity AND spatial distance
    - Pro: Preserves edges while smoothing
    - Con: Slightly slower than Gaussian blur
    - Best for: Images where edge preservation is critical
    
    Args:
        image: Input grayscale image
        noise_ratio: Estimated noise level (0.0-1.0)
    
    Returns:
        Denoised image with Bilateral Filter applied
    """
    diameter, sigma_color, sigma_space = select_bilateral_params(noise_ratio)
    
    # Convert to uint8 if needed (bilateral filter requirement for grayscale)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    filtered = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
    return filtered

def filter_gaussian(image: np.ndarray, noise_ratio: float, method: str = "bilateral") -> np.ndarray:
    """
    Main filtering function for Gaussian noise reduction.
    
    Args:
        image: Input grayscale image
        noise_ratio: Estimated noise level (0.0-1.0)
        method: Filtering method - "gaussian" or "bilateral" (default: "bilateral")
    
    Returns:
        Denoised image
    """
    if method == "gaussian":
        return apply_gaussian_blur(image, noise_ratio)
    elif method == "bilateral":
        return apply_bilateral_filter(image, noise_ratio)
    else:
        # Default to bilateral (better edge preservation)
        return apply_bilateral_filter(image, noise_ratio)
