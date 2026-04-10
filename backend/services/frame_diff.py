import cv2
import numpy as np

class FrameDiffChecker:
    def __init__(self, threshold: float = 5.0):
        self.last_frame_gray = None
        self.threshold = threshold

    def is_significantly_different(self, frame: np.ndarray) -> bool:
        """
        Computes the mean absolute difference between the current frame and the last processed frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.last_frame_gray is None:
            self.last_frame_gray = gray
            return True
        
        diff = cv2.absdiff(self.last_frame_gray, gray)
        mean_diff = np.mean(diff)
        
        if mean_diff > self.threshold:
            self.last_frame_gray = gray
            return True
        
        return False

    def reset(self):
        self.last_frame_gray = None
