# 🎬 Video Understanding Web Application

An intelligent, state-of-the-art system that extracts meaningful information from video, images, or live streams without redundancy.

## 🚀 Features

- **Intelligent Frame Difference detection**: Processes ONLY significant visual changes (Mean Pixel Difference).
- **Custom Denoising Layer**: Modular hook for image cleanup.
- **YOLOv8 Object Detection**: Detects all objects, flags "person" (Teacher Detection).
- **PaddleOCR Engine**: Extracts structured text with high accuracy.
- **Semantic Text Deduplication**: Uses Sentence Transformers and Cosine Similarity (threshold > 0.9) to skip repeated text.
- **Local MongoDB Storage**: Persists only unique, meaningful entries.
- **Modern React Frontend**: Live stream support and timeline-based results.

## 🏗️ Tech Stack

- **Backend**: FastAPI, OpenCV, YOLOv8, PaddleOCR, Motor (MongoDB), sentence-transformers.
- **Frontend**: React (Vite), Modern CSS, WebSockets.
- **Database**: MongoDB (Local).

## 🛠️ Installation & Setup

### 1. Backend Setup
```bash
# Activate your environment
source ~/ML/.venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 2. Frontend Setup
```bash
cd frontend
npm install
```

### 3. Run the Application
Start the backend:
```bash
python backend/main.py
```

Start the frontend:
```bash
cd frontend
npm run dev
```

## 📊 Database Entry Structure
```json
{
  "timestamp": "ISO-8601",
  "teacher_detected": true/false,
  "detected_objects": [
    { "label": "person", "confidence": 0.95, "bbox": [x1, y1, x2, y2] }
  ],
  "extracted_text": "Cleaned OCR text content",
  "is_new_content": true
}
```
