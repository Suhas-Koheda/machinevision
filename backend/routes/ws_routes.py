"""
Master Vision System: Integrates Research Pipeline with Utility Features.
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
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import JSONResponse

# Services
from backend.services import preprocessing_service as pre
from backend.services.tracking_service import get_tracker
from backend.services.flow_service import get_flow_service
from backend.services.clip_service import get_clip_service
from backend.services.scene_graph_service import construct_scene_graph
from backend.services.event_engine import get_event_engine
from backend.services.ocr_service import extract_text
from backend.services.text_engine import get_incremental
from backend.services.db_writer import save_detection_v2
from backend.services.video_recorder import recorder

logger = logging.getLogger(__name__)
router = APIRouter()

_frame_counter = 0
_current_session_id = ""
_is_recording = False
active_websockets = set()


def _encode_b64(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return base64.b64encode(buf).decode("utf-8")

async def _process_perception_pipeline(frame: np.ndarray, ws: WebSocket | None, session_id: str, is_recording: bool, force_intel: bool = False):
    global _frame_counter
    _frame_counter += 1
    
    # 1. Preprocessing
    noise_score, noise_map = pre.estimate_noise(frame)
    clean_frame = pre.apply_adaptive_filter(frame, noise_score)
    
    # 2. Motion
    flow_svc = get_flow_service()
    flow_vectors, flow_vis = flow_svc.compute_flow(clean_frame)
    motion_score = np.abs(flow_vectors).mean() if flow_vectors is not None else 0.0
    
    # Adaptive Skip (Only for live streams)
    if not force_intel and motion_score < 0.4 and _frame_counter % 20 != 0 and ws:
        return None
        
    # 3. Tracking
    tracker = get_tracker()
    detections = await asyncio.to_thread(tracker.track, clean_frame)
    annotated = tracker.draw_tracks(clean_frame, detections)
    
    # 4. CLIP (Deep Semantics)
    semantic_tags = []
    if (force_intel or _frame_counter % 10 == 0) and detections:
        clip = get_clip_service()
        semantic_tags = await asyncio.to_thread(clip.analyze_crops, clean_frame, detections)
        
    # 5. Scene Graph
    graph = construct_scene_graph(detections, semantic_tags)
    
    # 6. OCR
    ocr_text = ""
    if force_intel or _frame_counter % 15 == 0:
        raw_ocr = await asyncio.to_thread(extract_text, clean_frame)
        ocr_text = get_incremental(raw_ocr)

        
    # 7. Events
    events = get_event_engine().update(detections)
    
    # Recording
    if is_recording:
        recorder.write(annotated)
    
    result = {
        "session_id": session_id,
        "frame_index": _frame_counter,
        "objects": detections,
        "semantic": semantic_tags,
        "graph": graph,
        "text": ocr_text,
        "events": events,
        "noise_score": round(noise_score, 1),
        "motion_score": round(float(motion_score), 2),
        "processed_img": f"data:image/jpeg;base64,{_encode_b64(annotated)}",
        "noise_map": f"data:image/jpeg;base64,{_encode_b64(noise_map)}",
        "flow_map": f"data:image/jpeg;base64,{_encode_b64(flow_vis)}",
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    
    if ws:
        await ws.send_json(result)
    elif active_websockets:
        # Broadcast to all if no specific WS provided (e.g. during upload)
        broadcast_tasks = [s.send_json(result) for s in active_websockets]
        await asyncio.gather(*broadcast_tasks, return_exceptions=True)
        
    # Persistent Storage
    if _frame_counter % 5 == 0:
        asyncio.create_task(save_detection_v2(result, annotated))
        
    return result

@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    active_websockets.add(ws)
    global _frame_counter, _current_session_id, _is_recording
    _current_session_id = f"live_{datetime.datetime.now().strftime('%m%d_%H%M')}"
    _frame_counter = 0
    _is_recording = False
    
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
                
            if "bytes" in msg:
                arr = np.frombuffer(msg["bytes"], np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    await _process_perception_pipeline(frame, ws, _current_session_id, _is_recording)
            elif "text" in msg:
                cmd = json.loads(msg["text"])
                if cmd.get("type") == "start_recording":
                    _is_recording = True
                    recorder.start(640, 480)
                elif cmd.get("type") == "stop_recording":
                    _is_recording = False
                    url = recorder.stop()
                    await ws.send_json({"type": "recording_saved", "url": url})
    except (WebSocketDisconnect, RuntimeError):
        logger.info("Connection Terminated.")
    finally:
        active_websockets.remove(ws)


@router.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    session_id = f"file_{uuid.uuid4().hex[:6]}"
    os.makedirs("static/uploads", exist_ok=True)
    temp_path = f"static/uploads/{session_id}_{file.filename}"
    with open(temp_path, "wb") as b: shutil.copyfileobj(file.file, b)

    cap = cv2.VideoCapture(temp_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if count % 20 == 0:
            await _process_perception_pipeline(frame, None, session_id, False, force_intel=True)

        count += 1
    cap.release()
    if os.path.exists(temp_path): os.remove(temp_path)
    return JSONResponse(content={"status": "complete", "session_id": session_id})

# Migration support for old UI
@router.get("/api/sessions")
async def list_sessions():
    from backend.services.db_writer import _sync_get_sessions
    return await asyncio.to_thread(_sync_get_sessions)

@router.get("/api/sessions/{session_id}/frames")
async def list_session_frames(session_id: str):
    from backend.services.db_writer import _sync_get_detections
    return await asyncio.to_thread(_sync_get_detections, session_id)

@router.delete("/api/sessions/{session_id}")
async def remove_session(session_id: str):
    from backend.services.db_writer import delete_session
    return await delete_session(session_id)

@router.delete("/api/frames/{frame_id}")
async def remove_frame(frame_id: int):
    from backend.services.db_writer import delete_frame
    return await delete_frame(frame_id)

