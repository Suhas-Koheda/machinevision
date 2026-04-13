import numpy as np
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class TrackerService:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.history = {} # track_id -> [trajectory points]

    def track(self, frame: np.ndarray) -> list[dict]:
        """
        Runs YOLO tracking on frame.
        Returns persistent detections with track_id.
        """
        results = self.model.track(frame, persist=True, verbose=False)[0]
        
        detections = []
        if results.boxes is None: return detections
        
        for box in results.boxes:
            track_id = int(box.id[0]) if box.id is not None else -1
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = results.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            det = {
                "track_id": track_id,
                "label": label,
                "confidence": round(conf, 3),
                "bbox": [round(x1), round(y1), round(x2), round(y2)]
            }
            detections.append(det)
            
            # Update trajectory
            if track_id != -1:
                center = ((x1 + x2)/2, (y1 + y2)/2)
                if track_id not in self.history: self.history[track_id] = []
                self.history[track_id].append(center)
                if len(self.history[track_id]) > 30: self.history[track_id].pop(0)

        return detections

    def get_trajectories(self):
        return self.history

    def draw_tracks(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        import cv2
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            tid = det["track_id"]
            label = f"ID:{tid} {det['label']}"
            color = (int((tid * 50) % 255), int((tid * 80) % 255), 255)
            
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw trail
            if tid in self.history:
                points = self.history[tid]
                for i in range(1, len(points)):
                    p1 = (int(points[i-1][0]), int(points[i-1][1]))
                    p2 = (int(points[i][0]), int(points[i][1]))
                    cv2.line(out, p1, p2, color, 1)
        return out

_tracker = None
def get_tracker():
    global _tracker
    if _tracker is None:
        _tracker = TrackerService()
    return _tracker
