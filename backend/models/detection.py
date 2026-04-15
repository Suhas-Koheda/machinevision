import datetime
import json
from sqlalchemy import Column, Integer, String, DateTime, Text, Float, LargeBinary
from backend.db.database import Base

class Detection(Base):
    __tablename__ = "detections_v2"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    frame_index = Column(Integer)
    
    # Store complex JSON data
    objects_data = Column(Text)       # Full track/det list
    motion_score = Column(Float)
    noise_score = Column(Float)
    semantic_data = Column(Text)      # CLIP output
    graph_data = Column(Text)         # Scene graph
    ocr_text = Column(Text)
    events_data = Column(Text)
    
    image_path = Column(String)
    embedding = Column(LargeBinary) # Stores CLIP feature vector (np.ndarray bytes)
