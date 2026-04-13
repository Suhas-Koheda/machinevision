"""
Non-blocking DB writer with frame persistence.
"""
import asyncio
import datetime
import json
import logging
import os
import cv2

logger = logging.getLogger(__name__)

def _sync_insert(objects: list[dict], text: str, frame_index: int, session_id: str, frame_img=None):
    from backend.db.database import SessionLocal
    from backend.models.detection import Detection

    if not objects and not text:
        return

    avg_conf = (
        sum(o.get("confidence", 0) for o in objects) / len(objects) if objects else 0.0
    )

    image_path = None
    if frame_img is not None:
        save_dir = f"static/frames/{session_id}"
        os.makedirs(save_dir, exist_ok=True)
        filename = f"frame_{frame_index}_{datetime.datetime.now().timestamp()}.jpg"
        image_path = f"/{save_dir}/{filename}"
        cv2.imwrite(os.path.join(save_dir, filename), frame_img)

    db = SessionLocal()
    try:
        record = Detection(
            session_id=session_id,
            timestamp=datetime.datetime.utcnow(),
            objects_json=json.dumps(objects),
            text_extracted=text,
            image_path=image_path,
            confidence_avg=round(avg_conf, 3),
            frame_index=frame_index,
        )
        db.add(record)
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error("DB insert failed: %s", exc)
    finally:
        db.close()

async def save_detection(objects: list[dict], text: str, frame_index: int, session_id: str, frame_img=None):
    await asyncio.to_thread(_sync_insert, objects, text, frame_index, session_id, frame_img)

def _sync_get_sessions():
    from backend.db.database import SessionLocal
    from backend.models.detection import Detection
    from sqlalchemy import func
    
    db = SessionLocal()
    try:
        results = db.query(
            Detection.session_id, 
            func.min(Detection.timestamp).label('start'),
            func.count(Detection.id).label('frame_count')
        ).group_by(Detection.session_id).order_by(func.min(Detection.timestamp).desc()).all()
        
        return [{"id": r[0], "start": r[1].isoformat(), "count": r[2]} for r in results]
    finally:
        db.close()

async def get_sessions():
    return await asyncio.to_thread(_sync_get_sessions)

def _sync_get_detections(session_id=None, limit=50):
    from backend.db.database import SessionLocal
    from backend.models.detection import Detection

    db = SessionLocal()
    try:
        query = db.query(Detection)
        if session_id:
            query = query.filter(Detection.session_id == session_id)
        rows = query.order_by(Detection.timestamp.desc()).limit(limit).all()
        return [
            {
                "id": r.id,
                "session_id": r.session_id,
                "timestamp": r.timestamp.isoformat(),
                "objects": json.loads(r.objects_json or "[]"),
                "text": r.text_extracted,
                "image_path": r.image_path,
                "confidence_avg": r.confidence_avg,
                "frame_index": r.frame_index,
            }
            for r in rows
        ]
    finally:
        db.close()

async def get_detections(session_id=None, limit=50):
    return await asyncio.to_thread(_sync_get_detections, session_id, limit)

def _sync_delete_frame(frame_id):
    from backend.db.database import SessionLocal
    from backend.models.detection import Detection
    db = SessionLocal()
    try:
        item = db.query(Detection).filter(Detection.id == frame_id).first()
        if item:
            if item.image_path:
                full_path = item.image_path.lstrip("/")
                if os.path.exists(full_path):
                    os.remove(full_path)
            db.delete(item)
            db.commit()
            return True
        return False
    finally:
        db.close()

async def delete_frame(frame_id):
    return await asyncio.to_thread(_sync_delete_frame, frame_id)

def _sync_delete_session(session_id):
    from backend.db.database import SessionLocal
    from backend.models.detection import Detection
    import shutil
    db = SessionLocal()
    try:
        db.query(Detection).filter(Detection.session_id == session_id).delete()
        db.commit()
        # Clean folder
        folder = f"static/frames/{session_id}"
        if os.path.exists(folder):
            shutil.rmtree(folder)
        return True
    finally:
        db.close()

async def delete_session(session_id):
    return await asyncio.to_thread(_sync_delete_session, session_id)
