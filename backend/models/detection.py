"""
ORM model for storing vision detections.
"""
import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from backend.db.database import Base


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)         # Grouping key (UUID or timestamp)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    objects_json = Column(Text, default="[]")       # JSON list of detected objects
    text_extracted = Column(Text, default="")        # Cleaned OCR text
    image_path = Column(Text, nullable=True)        # Path to the saved frame image
    confidence_avg = Column(Float, default=0.0)      # Average YOLO confidence
    frame_index = Column(Integer, default=0)         # Frame sequence number

