# 🌌 VisionAICore // AI Vision Intelligence Dashboard

A high-performance, real-time video perception engine and intelligence archive designed for autonomous scene understanding, object persistence, and semantic reasoning. 

**VisionAICore** transforms raw video feeds into structured data by integrating state-of-the-art neural networks with a robust processing pipeline.

---

## 🛰️ CORE INTELLIGENCE PILLARS

### 1. Neural Tracking & Persistence
Powered by **YOLOv8** (You Only Look Once), the system provides real-time multi-object detection and tracking. 
- **Persistent ID assignment**: Every detected entity is assigned a unique tracking ID that persists across frames.
- **Trajectory Analysis**: Visual trails mapped for every moving object to visualize historical motion paths.

### 2. Deep Semantic Reasoning
Integrates **CLIP (Contrastive Language-Image Pre-training)** to perform zero-shot classification and semantic analysis of detected crops.
- **Environment Context**: Goes beyond simple labels (e.g., "Person") to describe semantics (e.g., "Person sitting on a chair").
- **Visual-Textual Alignment**: Analyzes the relationship between image fragments and semantic tags.

### 3. OCR Engine (Optical Character Recognition)
Leverages **PaddleOCR** for high-accuracy text extraction from video frames.
- **Adaptive Preprocessing**: Automatically applies Otsu thresholding and grayscale conversion to maximize character recognition.
- **Incremental Extraction**: Intelligent deduplication to prevent redundant text logging across consecutive frames.

### 4. Motion & Diagnostic Telemetry
Uses **Farneback Dense Optical Flow** to analyze physical movement within the frame.
- **Motion Scoring**: Quantifies temporal changes to detect anomalies or significant activity.
- **Noise Analysis**: Real-time Laplacian variance estimation with JET colormap heatmaps to monitor signal quality.
- **Adaptive Filtering**: Dynamically applies Gaussian or Median blurs based on detected noise intensity to preserve intelligence accuracy.

### 5. Relationship & Event Intelligence
Constructs a dynamic **Scene Graph** and monitors for logic-based triggers.
- **Predicate Linking**: Maps relationships between objects (e.g., `Subject ➔ Predicate ➔ Object`) using detection proximities.
- **Event Engine**: Monitoring for entry/exit events and persistent state changes within the visual field.
- **Intelligence Logging**: All events are timestamped and archived for historical audit.

---

## 🔬 NEURAL MODALITY SPECIFICATIONS

| Component | Model / Architecture | Purpose |
| :--- | :--- | :--- |
| **Detection** | YOLOv8n (Nano) | High-speed object localization & tracking |
| **Semantics** | CLIP Vit-Base-Patch32 | Zero-shot visual-semantic understanding |
| **Text Vision** | PaddleOCR (Angle-Cls) | Real-time text extraction & rectification |
| **Motion** | Farneback Dense Flow | Physics-based movement telemetry |

---

## 🎭 OPERATIONAL FEATURES

| Feature | Description |
| :--- | :--- |
| **Real-time Feed** | 20fps low-latency video streaming via binary WebSockets. |
| **Video Archival** | Backend-side recording of processed streams into high-compression containers. |
| **Voice Transcription** | Browser-integrated Speech-to-Text for logging verbal observations alongside visual intelligence. |
| **Intelligence Archive** | SQL-backed persistence for sessions, keyframes, and metadata. |
| **Local Uploads** | Batch process pre-recorded video files through the intelligence pipeline. |

---

## 🛠️ SETUP & DEPLOYMENT

### Prerequisites
- **Python 3.10+** (Recommended)
- **Node.js 18+** (For frontend development)
- **GPU (Optional)**: CUDA-capable GPU for accelerated YOLO and CLIP processing.

### 1. Clone & Environment
```bash
git clone https://github.com/suhas-koheda/machinevision.git
cd machinevision
python3 -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
```

### 2. Install Core Dependencies
The system requires several heavy-weight ML libraries:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Build/Run Frontend
```bash
cd frontend
npm install
npm run build
cd ..
```

### 4. Launch the Intelligence Hub
```bash
python main.py
```
Access the dashboard at **[http://localhost:8000](http://localhost:8000)**.

---

## 📂 ARCHITECTURAL OVERVIEW

```text
├── main.py                 # FastAPI Application Server & Lifecycle Management
├── backend/
│   ├── db/                 # SQL Persistence (Database initialization & queries)
│   ├── routes/             # WebSocket handlers & REST Endpoints
│   ├── services/           # The Engine Core
│   │   ├── tracking_service.py # YOLOv8 Tracking Implementation
│   │   ├── clip_service.py     # CLIP Semantic Analyzers
│   │   ├── ocr_service.py      # PaddleOCR Integration
│   │   ├── flow_service.py     # Optical Flow Telemetry
│   │   └── event_engine.py     # Rule-based Event Detection
├── frontend/               
│   ├── src/                # React/Vite UI Architecture
│   └── dist/               # Production Build Artifacts
└── static/                 # Storage for recordings, uploads, and cached frames
```

---

*Engineered for high-performance vision research by Antigravity.*