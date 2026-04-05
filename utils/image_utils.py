import numpy as np
import cv2
import base64

def decode_image(image_bytes: bytes) -> np.ndarray:
    """
    Decodes an image from bytes and converts it to Grayscale using OpenCV.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Decode to Color first, then convert explicitly to Grayscale
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return None
    # Explicit conversion to ensure OpenCV's grayscale logic is applied
    grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_img

def encode_image_base64(image: np.ndarray) -> str:
    """
    Encodes a grayscale NumPy array back into a base64-encoded PNG string.
    """
    # Using PNG as it's lossless for academic evaluation
    success, encoded_image = cv2.imencode(".png", image)
    if not success:
        return ""
    
    return base64.b64encode(encoded_image.tobytes()).decode("utf-8")
