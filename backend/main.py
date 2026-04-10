import cv2
import numpy as np
import asyncio
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import uuid
import base64

from services.frame_diff import FrameDiffChecker
from services.denoise import denoise
from services.yolo_detector import YOLODetector
from services.ocr_engine import OCREngine
from services.similarity import TextSimilarityEngine
from db.database import save_result, get_results, clear_session, clear_all

app = FastAPI(title="Video Understanding API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
frame_checker = FrameDiffChecker(threshold=5.0)
yolo = YOLODetector()
ocr = OCREngine()
text_similarity = TextSimilarityEngine()

@app.get("/")
async def root():
    return {"message": "Video Understanding API is running"}

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    # In a real app, we'd save the file and process it in background
    # For now, let's acknowledge
    file_id = str(uuid.uuid4())
    return {"id": file_id, "filename": file.filename}

@app.get("/results")
async def fetch_results():
    results = await get_results()
    sessions = {}
    for r in results:
        sid = r.get("session_id", "unknown")
        if sid not in sessions:
            sessions[sid] = {"id": sid, "name": r.get("video_name", f"Video {sid[:8]}"), "frames": []}
        sessions[sid]["frames"].append(r)
    return {"results": results, "sessions": list(sessions.values())}

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    await clear_session(session_id)
    return {"message": "Session deleted"}

@app.delete("/clear")
async def clear_all_data():
    await clear_all()
    return {"message": "All data cleared"}

@app.post("/process")
async def process_video(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    
    session_id = str(uuid.uuid4())
    video_name = file.filename or "Uploaded Video"
    
    temp_filename = f"temp_{uuid.uuid4()}.mp4"
    with open(temp_filename, "wb") as f:
        f.write(contents)
    
    cap = cv2.VideoCapture(temp_filename)
    results = []
    frame_checker.reset()
    yolo.reset()
    text_similarity.reset()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        if frame_checker.is_significantly_different(frame):
            print(f"📸 Significant frame change detected at {timestamp:.2f}ms. Processing...")
            processed_frame = denoise(frame)
            detections, teacher_detected = yolo.detect(processed_frame)
            if detections:
                print(f"📦 YOLO: Detected {len(detections)} objects: {[d['label'] for d in detections]}")
            
            ocr_items, full_text = ocr.extract_text(processed_frame)
            if full_text:
                print(f"🔤 OCR: Extracted text: \"{full_text[:50]}...\"")
            
            is_new_text = text_similarity.is_semantically_different(full_text)
            
            if is_new_text:
                # Create annotated frame for the UI
                annotated_frame = yolo.draw_detections(processed_frame, detections)
                annotated_frame = ocr.draw_text(annotated_frame, full_text)
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                unique_labels = [d["label"] for d in detections if d.get("is_new", True)]
                
                result_entry = {
                    "timestamp": f"{timestamp:.2f}",
                    "teacher_detected": teacher_detected,
                    "detected_objects": detections,
                    "extracted_text": full_text,
                    "is_new_content": True,
                    "session_id": session_id,
                    "video_name": video_name,
                    "unique_objects": unique_labels,
                    "image_data": f"data:image/jpeg;base64,{img_base64}"
                }
                await save_result(result_entry)
                results.append(result_entry)
            else:
                print(f"⏭️ Skipping: Text is semantically similar to previous frame.")
                    
    cap.release()
    import os
    os.remove(temp_filename)
    
    return {"frames": results, "session_id": session_id, "video_name": video_name}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    frame_checker.reset()
    yolo.reset()
    text_similarity.reset()
    session_id = str(uuid.uuid4())
    video_name = "Live Stream"
    
    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
                
            if frame_checker.is_significantly_different(frame):
                print(f"📡 WebSocket: Received significant frame. Processing...")
                processed_frame = denoise(frame)
                detections, teacher_detected = yolo.detect(processed_frame)
                ocr_items, full_text = ocr.extract_text(processed_frame)
                
                # Always send the annotated frame for the live view
                annotated_frame = yolo.draw_detections(processed_frame, detections)
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                unique_labels = [d["label"] for d in detections if d.get("is_new", True)]
                
                result_entry = {
                    "timestamp": str(datetime.now().strftime("%H:%M:%S")),
                    "teacher_detected": teacher_detected,
                    "detected_objects": detections,
                    "extracted_text": full_text,
                    "is_new_content": True,
                    "session_id": session_id,
                    "video_name": video_name,
                    "unique_objects": unique_labels,
                    "image_data": f"data:image/jpeg;base64,{img_base64}"
                }
                await save_result(result_entry)
                await websocket.send_json(result_entry)
                print(f"✨ WebSocket: Sent frame with {len(detections)} objects")
            else:
                print(f"⏭️ WebSocket: Content similar, skipping notification.")
    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
