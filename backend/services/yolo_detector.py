import cv2
from ultralytics import YOLO
import numpy as np
import torch

class YOLODetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path).to(device)
        self.seen_objects = {}
        self.seen_classes = set()

    def reset(self):
        self.seen_objects = {}
        self.seen_classes = set()

    def detect(self, image: np.ndarray):
        results = self.model(image, verbose=False)[0]
        detections = []
        teacher_detected = False
        unique_new = []
        
        for box in results.boxes:
            label = self.model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            
            box_key = f"{label}_{int(coords[0])//50}_{int(coords[1])//50}"
            
            if box_key not in self.seen_objects:
                self.seen_objects[box_key] = True
                unique_new.append(True)
            else:
                unique_new.append(False)
            
            detections.append({
                "label": label,
                "confidence": confidence,
                "bbox": coords,
                "is_new": unique_new[-1]
            })
            
            if label == "person":
                teacher_detected = True

        return detections, teacher_detected
    
    def draw_detections(self, image: np.ndarray, detections: list):
        annotated = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
            label = det["label"]
            is_new = det.get("is_new", True)
            
            if is_new:
                color = (0, 255, 0)
                thickness = 3
            else:
                color = (120, 120, 120)
                thickness = 1
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            text = f"{label} {int(det['confidence'] * 100)}%"
            
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - text_height - 8), (x1 + text_width + 4, y1), color, -1)
            cv2.putText(annotated, text, (x2 - text_width - 2, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated