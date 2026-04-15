"""
OCR Service using PaddleOCR.
Initialised once; exposes extract_text(frame) → str.
"""
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

_ocr = None


def get_ocr():
    global _ocr
    if _ocr is None:
        logger.info("Initialising PaddleOCR …")
        from paddleocr import PaddleOCR

        _ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        logger.info("PaddleOCR ready.")
    return _ocr



def extract_text(frame: np.ndarray, min_confidence: float = 0.4) -> str:
    """
    Run OCR on frame.
    Returns cleaned, deduplicated text string.
    """
    try:
        ocr = get_ocr()
        # PaddleOCR often performs better on raw BGR frames or simple grayscale
        # We'll try raw frame first as it preserves details Otsu might destroy
        result = ocr.ocr(frame, cls=True)

        if not result or not result[0]:
            return ""

        words = []
        seen = set()
        for line in result[0]:
            if not line or len(line) < 2:
                continue
            text_info = line[1]
            if not text_info:
                continue
            word, conf = text_info[0], text_info[1]
            word = word.strip()
            
            # Use a more relaxed confidence threshold
            if conf < min_confidence:
                continue
            
            # Basic cleanup: remove extra whitespace but keep special characters
            word = " ".join(word.split())
            if not word:
                continue
                
            key = word.lower()
            if key in seen:
                continue
            seen.add(key)
            words.append(word)

        return " ".join(words)

    except Exception as exc:
        logger.error("OCR error: %s", exc)
        return ""
