import sqlite3
import json
from datetime import datetime
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "video_understanding.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_frames (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            teacher_detected BOOLEAN,
            detected_objects TEXT,
            extracted_text TEXT,
            is_new_content BOOLEAN,
            session_id TEXT,
            video_name TEXT,
            unique_objects TEXT,
            image_data TEXT
        )
    ''')
    conn.commit()
    conn.close()

async def clear_session(session_id: str):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM processed_frames WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

async def clear_all():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM processed_frames")
    conn.commit()
    conn.close()

async def save_result(data: dict):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO processed_frames 
        (timestamp, teacher_detected, detected_objects, extracted_text, is_new_content, session_id, video_name, unique_objects, image_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        str(data.get("timestamp", datetime.now())),
        data.get("teacher_detected", False),
        json.dumps(data.get("detected_objects", [])),
        data.get("extracted_text", ""),
        data.get("is_new_content", True),
        data.get("session_id", ""),
        data.get("video_name", ""),
        json.dumps(data.get("unique_objects", [])),
        data.get("image_data", "")
    ))
    conn.commit()
    conn.close()

async def get_results():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM processed_frames ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    
    results = []
    for row in rows:
        results.append({
            "timestamp": row["timestamp"],
            "teacher_detected": bool(row["teacher_detected"]),
            "detected_objects": json.loads(row["detected_objects"]) if row["detected_objects"] else [],
            "extracted_text": row["extracted_text"],
            "is_new_content": bool(row["is_new_content"]),
            "session_id": row["session_id"],
            "video_name": row["video_name"] if "video_name" in row.keys() else "",
            "unique_objects": json.loads(row["unique_objects"]) if row["unique_objects"] else [],
            "image_data": row["image_data"]
        })
    conn.close()
    return results

init_db()