"""
OCR Service using PaddleOCR.
Initialised once; exposes extract_text(frame) → str.
"""
import logging
import re
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


def preprocess_for_ocr(frame: np.ndarray) -> np.ndarray:
    """Convert to grayscale + Otsu threshold for cleaner OCR."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def extract_text(frame: np.ndarray, min_confidence: float = 0.75) -> str:
    """
    Run OCR on frame.
    Returns cleaned, deduplicated text string.
    """
    try:
        ocr = get_ocr()
        preprocessed = preprocess_for_ocr(frame)
        result = ocr.ocr(preprocessed, cls=True)

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
            # filters
            if conf < min_confidence:
                continue
            if len(word) < 2:
                continue
            # Remove redundant internal spaces and junk characters
            word = re.sub(r'[^a-zA-Z0-9\s]', '', word)
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
