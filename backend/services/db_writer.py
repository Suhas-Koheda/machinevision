import asyncio
import datetime
import json
import logging
import os
import cv2

logger = logging.getLogger(__name__)

def _sync_insert_v2(data: dict, frame_img=None):
    from backend.db.database import SessionLocal
    from backend.models.detection import Detection

    img_path = None
    if frame_img is not None:
        sid = data.get("session_id", "unknown")
        fid = data.get("frame_index", 0)
        save_dir = f"static/frames/{sid}"
        os.makedirs(save_dir, exist_ok=True)
        filename = f"f_{fid}_{datetime.datetime.now().timestamp()}.jpg"
        img_path = f"/{save_dir}/{filename}"
        cv2.imwrite(os.path.join(save_dir, filename), frame_img)

    db = SessionLocal()
    try:
        record = Detection(
            session_id=data["session_id"],
            frame_index=data["frame_index"],
            objects_data=json.dumps(data.get("objects", [])),
            motion_score=data.get("motion_score", 0.0),
            noise_score=data.get("noise_score", 0.0),
            semantic_data=json.dumps(data.get("semantic", [])),
            graph_data=json.dumps(data.get("graph", [])),
            ocr_text=data.get("text", ""),
            events_data=json.dumps(data.get("events", [])),
            image_path=img_path
        )
        db.add(record)
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error("DB V2 insert failed: %s", exc)
    finally:
         db.close()

async def save_detection_v2(data: dict, frame_img=None):
    await asyncio.to_thread(_sync_insert_v2, data, frame_img)

def _sync_get_sessions():
    from backend.db.database import SessionLocal
    from backend.models.detection import Detection
    from sqlalchemy import func
    db = SessionLocal()
    try:
        res = db.query(
            Detection.session_id, 
            func.min(Detection.timestamp), 
            func.count(Detection.id)
        ).group_by(Detection.session_id).order_by(Detection.timestamp.desc()).all()
        return [{"id": r[0], "start": r[1].isoformat(), "count": r[2]} for r in res]
    finally:
        db.close()

def _sync_get_detections(session_id: str):
    from backend.db.database import SessionLocal
    from backend.models.detection import Detection
    db = SessionLocal()
    try:
        res = db.query(Detection).filter(Detection.session_id == session_id).order_by(Detection.frame_index.asc()).all()
        return [{
            "id": r.id, 
            "frame_index": r.frame_index,
            "objects": json.loads(r.objects_data),
            "text": r.ocr_text,
            "image_path": r.image_path
        } for r in res]
    finally:
        db.close()

async def delete_session(session_id: str):
    from backend.db.database import SessionLocal
    from backend.models.detection import Detection
    import shutil
    db = SessionLocal()
    try:
        db.query(Detection).filter(Detection.session_id == session_id).delete()
        db.commit()
        path = f"static/frames/{session_id}"
        if os.path.exists(path): shutil.rmtree(path)
        return {"status": "deleted"}
    finally:
        db.close()

async def delete_frame(frame_id: int):
    from backend.db.database import SessionLocal
    from backend.models.detection import Detection
    db = SessionLocal()
    try:
        f = db.query(Detection).filter(Detection.id == frame_id).first()
        if f:
            if f.image_path and os.path.exists(f.image_path.lstrip('/')): 
                os.remove(f.image_path.lstrip('/'))
            db.delete(f)
            db.commit()
        return {"status": "deleted"}
    finally:
        db.close()
