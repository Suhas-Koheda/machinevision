import cv2
import numpy as np

class FlowService:
    def __init__(self):
        self.prev_gray = None

    def compute_flow(self, frame: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
        """
        Computes dense optical flow.
        Returns: flow_vectors (raw), visualization (np.ndarray)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = np.zeros_like(frame)
        
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return None, vis


        # Dense flow
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Visualize flow arrows
        step = 16
        h, w = gray.shape
        y, x = np.mgrid[step // 2:h:step, step // 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2).astype(int)
        
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (_x2, _y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

        self.prev_gray = gray
        return flow, vis

_flow_svc = FlowService()
def get_flow_service():
    return _flow_svc
