import matplotlib.pyplot as plt
import io
import base64
import numpy as np

def generate_histogram_base64(image: np.ndarray, title: str) -> str:
    """
    Generates a histogram of the image intensities using Matplotlib
    and returns it as a base64-encoded PNG string.
    """
    plt.figure(figsize=(5, 4))
    plt.hist(image.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')
