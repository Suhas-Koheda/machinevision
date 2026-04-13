"""
WebSocket + REST routes for the vision pipeline.
Includes Sessions, Deletion, and Persistence.
"""
import asyncio
import base64
import datetime
import json
import logging
import os
import shutil
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse

from backend.services import frame_diff, text_engine
from backend.services.yolo_service import detect_objects, draw_detections
from backend.services.ocr_service import extract_text
from backend.services.db_writer import save_detection, get_detections, get_sessions, delete_frame, delete_session
from backend.services.video_recorder import recorder

logger = logging.getLogger(__name__)
router = APIRouter()

_frame_counter = 0
_current_session_id = ""
_is_recording = False

def _decode_frame(raw: bytes) -> np.ndarray | None:
    try:
        arr = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None

def _encode_frame_b64(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf).decode("utf-8")

async def _process_frame(frame_bytes: bytes, ws: WebSocket | None, save_to_recorder: bool, session_id: str):
    global _frame_counter

    frame = _decode_frame(frame_bytes) if isinstance(frame_bytes, bytes) else frame_bytes
    if frame is None: return None

    # Skip redundant frames only for live stream
    if ws and not frame_diff.is_significant(frame): return None

    _frame_counter += 1
    
    # YOLO (Async thread)
    objects = await asyncio.to_thread(detect_objects, frame)
    
    # OCR (Async thread, every 3 frames)
    raw_text = ""
    if _frame_counter % 3 == 0:
        raw_text = await asyncio.to_thread(extract_text, frame)
    
    incremental_text = text_engine.get_incremental(raw_text)

    # Annotate frame
    annotated = draw_detections(frame, objects)
    
    if save_to_recorder:
        recorder.write(annotated)

    img_b64 = await asyncio.to_thread(_encode_frame_b64, annotated)

    result = {
        "session_id": session_id,
        "frame_index": _frame_counter,
        "objects": objects,
        "text": incremental_text,
        "image": f"data:image/jpeg;base64,{img_b64}",
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

    if ws:
        try:
            await ws.send_json(result)
        except Exception:
            pass

    # Non-blocking DB + Image persistence
    if objects or incremental_text:
        asyncio.create_task(save_detection(objects, incremental_text, _frame_counter, session_id, annotated))
    
    return result

@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    global _frame_counter, _current_session_id, _is_recording
    _current_session_id = f"live_{datetime.datetime.now().strftime('%m%d_%H%M%S')}"
    _frame_counter = 0
    _is_recording = False
    
    frame_diff.reset()
    text_engine.reset()
    logger.info(f"Accepted WS connection. Session: {_current_session_id}")

    try:
        while True:
            # Check for disconnect before receiving
            data = await ws.receive()
            if "bytes" in data:
                logger.info(f"Frame received: {len(data['bytes'])} bytes")
                asyncio.create_task(_process_frame(data["bytes"], ws, _is_recording, _current_session_id))


            elif "text" in data:
                cmd = json.loads(data["text"])
                if cmd.get("type") == "start_recording":
                    _is_recording = True
                    recorder.start(640, 480)
                elif cmd.get("type") == "stop_recording":
                    _is_recording = False
                    url = recorder.stop()
                    await ws.send_json({"type": "recording_saved", "url": url})
    except (WebSocketDisconnect, RuntimeError):
        logger.info("WebSocket disconnected.")
    except Exception as exc:
        logger.error("WS error: %s", exc)

@router.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    session_id = f"file_{uuid.uuid4().hex[:8]}"
    temp_dir = "static/uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{session_id}_{file.filename}")
    
    with open(temp_path, "wb") as b: 
        shutil.copyfileobj(file.file, b)

    cap = cv2.VideoCapture(temp_path)
    res_list = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        # Sample every 15 frames for upload
        if count % 15 == 0:
            res = await _process_frame(frame, None, False, session_id)
            if res: res_list.append(res)
        count += 1
    cap.release()
    try:
        os.remove(temp_path)
    except:
        pass
    return JSONResponse(content={"session_id": session_id, "frames_processed": len(res_list)})

@router.get("/api/sessions")
async def list_sessions():
    return await get_sessions()

@router.get("/api/sessions/{session_id}/frames")
async def list_session_frames(session_id: str):
    return await get_detections(session_id=session_id)

@router.delete("/api/frames/{frame_id}")
async def remove_frame(frame_id: int):
    success = await delete_frame(frame_id)
    return {"status": "success" if success else "failed"}

@router.delete("/api/sessions/{session_id}")
async def remove_session(session_id: str):
    success = await delete_session(session_id)
    return {"status": "success" if success else "failed"}

@router.get("/api/detections")
async def all_detections(limit: int = 50):
    return await get_detections(limit=limit)
