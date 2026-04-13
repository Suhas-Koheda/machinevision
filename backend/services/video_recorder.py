"""
Video Recorder Service.
Handles saving processed frames into a video file.
"""
import cv2
import os
import datetime
import logging

logger = logging.getLogger(__name__)

class VideoRecorder:
    def __init__(self, output_dir="static/recordings"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.writer = None
        self.filename = None

    def start(self, width, height, fps=5):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"record_{timestamp}.mp4"
        filepath = os.path.join(self.output_dir, self.filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        logger.info(f"Started recording to {filepath}")

    def write(self, frame):
        if self.writer:
            self.writer.write(frame)

    def stop(self):
        if self.writer:
            self.writer.release()
            self.writer = None
            logger.info("Recording stopped")
            return f"/static/recordings/{self.filename}"
        return None

recorder = VideoRecorder()
