import numpy as np
import cv2
import torch
import os

class OCREngine:
    def __init__(self):
        """
        Initialize PaddleOCR with safe detection and error reporting.
        Compatible with PaddleOCR 2.x and 3.x.
        """
        self.ocr = None
        self.mode = "none"
        
        try:
            # 1. Try to verify paddle environment
            import paddle
            use_gpu = torch.cuda.is_available()
            
            # Use paddle.set_device if possible (for 3.x compatibility)
            if use_gpu:
                try:
                    paddle.set_device('gpu')
                except Exception:
                    use_gpu = False
            else:
                paddle.set_device('cpu')

            # 2. Import PaddleOCR
            from paddleocr import PaddleOCR
            
            # To avoid "name predict_system is not defined" or "Unknown argument"
            # we use the most basic initialization. PaddleOCR 2.x/3.x will handle 
            # the rest automatically.
            self.ocr = PaddleOCR(lang='en', use_angle_cls=False)
            
            self.mode = "paddle"
            print(f"✅ OCR: PaddleOCR engine ready (CUDA available={use_gpu}).")
            
        except Exception as e:
            print(f"❌ OCR: Initialization failed: {e}")
            self.mode = "none"

    def extract_text(self, image: np.ndarray):
        if self.mode == "none" or self.ocr is None:
            return [], ""
            
        try:
            result = self.ocr.ocr(image)
            extracted_items = []
            full_text = []
            
            if result and result[0]:
                for line in result[0]:
                    extracted_items.append({
                        "text": line[1][0],
                        "bbox": line[0],
                        "confidence": float(line[1][1])
                    })
                    full_text.append(line[1][0])
            
            return extracted_items, " ".join(full_text)
        except Exception as e:
            print(f"⚠️ OCR: Extraction error: {e}")
            return [], ""

    def draw_text(self, image: np.ndarray, text: str):
        """
        Draws OCR text at the bottom of the frame with a semi-transparent background.
        """
        if not text:
            return image
            
        annotated = image.copy()
        h, w = annotated.shape[:2]
        
        # Use a nice font and calculate size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Split text into lines if it's too long
        max_w = w - 40
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = " ".join(current_line + [word])
            (tw, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            if tw < max_w:
                current_line.append(word)
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
        if current_line:
            lines.append(" ".join(current_line))
        
        # Draw background for text
        line_h = 30
        bg_h = len(lines) * line_h + 20
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, h - bg_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
        
        # Draw lines
        for i, line in enumerate(lines):
            y = h - bg_h + 30 + (i * line_h)
            cv2.putText(annotated, line, (20, y), font, font_scale, (255, 255, 255), thickness)
            
        return annotated
