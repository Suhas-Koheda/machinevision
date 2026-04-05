import numpy as np
import cv2
import base64

def decode_image(image_bytes: bytes) -> np.ndarray:
    """
    Decodes an image from bytes into a NumPy array using OpenCV.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    return image

def encode_image_base64(image: np.ndarray) -> str:
    """
    Encodes a NumPy array back into a base64-encoded PNG string.
    """
    success, encoded_image = cv2.imencode(".png", image)
    if not success:
        return ""
    
    return base64.b64encode(encoded_image.tobytes()).decode("utf-8")
