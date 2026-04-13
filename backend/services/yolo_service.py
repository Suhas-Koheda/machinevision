"""
YOLO Detection Service using Ultralytics YOLOv8.
Loads model once and reuses it for every frame.
"""
import logging
from ultralytics import YOLO
import numpy as np

logger = logging.getLogger(__name__)

_model = None


def get_model() -> YOLO:
    global _model
    if _model is None:
        logger.info("Loading YOLOv8n model …")
        _model = YOLO("yolov8n.pt")   # downloads once, cached locally
        logger.info("YOLOv8n ready.")
    return _model


def detect_objects(frame: np.ndarray, conf_threshold: float = 0.45) -> list[dict]:
    """
    Run YOLOv8 inference on a BGR frame.
    Returns list of dicts: {label, confidence, bbox:[x1,y1,x2,y2]}
    """
    try:
        model = get_model()
        results = model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            cls_id = int(box.cls[0])
            label = results.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "label": label,
                    "confidence": round(conf, 3),
                    "bbox": [round(x1), round(y1), round(x2), round(y2)],
                }
            )
        return detections
    except Exception as exc:
        logger.error("YOLO error: %s", exc)
        return []


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes + labels onto the frame (BGR copy)."""
    import cv2

    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['label']} {det['confidence']:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 80), 2)
        cv2.putText(
            out,
            label,
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 80),
            2,
        )
    return out
