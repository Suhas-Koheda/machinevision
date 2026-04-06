# 🌌 Noise Characterization & Removal Production Hub

A lightweight, robust, and academic-grade system for image noise analysis and restoration. This toolkit features an adaptive filtering core that dynamically scales processing power based on detected noise intensity.

---

## 🚀 ADAPTIVE FILTERING STRATEGY

Unlike traditional static filters, this project implements a **load-balanced filtering approach** for Salt & Pepper noise:

| Noise Ratio | Detected Level | Kernel Size | Strength |
| :--- | :--- | :--- | :--- |
| < 2% | Minimal | 3x3 | Sensitive |
| 2% - 5% | Moderate | 5x5 | Balanced |
| > 5% | Heavy | 7x7 | Maximum |

**Fail-safe**: If Salt & Pepper noise is detected with less than **1%** ratio, the system treats it as "clean" and bypasses filtering to preserve detail.

---

## 🔬 NOISE DETECTION LOGIC

The detection engine uses **threshold-based intensity analysis** rather than strict equality (which often fails on digital images due to compression):
- **Pepper**: Identified if pixel value ≤ 10
- **Salt**: Identified if pixel value ≥ 245
- **Confidence**: Dynamically scaled based on the ratio of detected noisy pixels.

---

## 🧬 HONEST CORE METRICS

We prioritize mathematical accuracy over "perfect scores":
- **MSE (Float Precision)**: Calculated using NumPy over the full image intensity range.
- **PSNR (Capped)**: Standard PSNR returns misleading `100` or `Infinity` on perfect images; our system caps PSNR at **50.0 dB** to reflect a realistic threshold of human perception.
- **Placeholder Fail-safe**: If a placeholder module (Gaussian/Speckle) is chosen, the engine returns **-1.0** for all metrics to signal that no valid processing occurred.

---

## 🛠️ SETUP & EXECUTION

Follow these steps to get the production core running on your local machine.

### 1. Environment Setup
Create a dedicated virtual environment to keep dependencies isolated and stable.

**Windows:**
```powershell
# Create the environment
python -m venv venv

# Activate the environment
.\venv\Scripts\activate
```

**macOS / Linux:**
```bash
# Create the environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate
```

### 2. Dependency Installation
Once the environment is active, install the required academic-grade libraries:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch the Engine
Start the FastAPI server using the production-ready wrapper:
```bash
python main.py
```

### 4. Access Points
- **Interactive UI**: [http://localhost:8000](http://localhost:8000)
- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: `GET /` (Serves the frontend)

---

## 📂 PROJECT STRUCTURE

```text
├── main.py              # Application entry point & FastAPI routing
├── requirements.txt      # System dependencies
├── salt_pepper/         # Core logic for Salt & Pepper noise
│   ├── detect.py        # Intensity-based detection engine
│   ├── filter.py        # Adaptive median filtering logic
│   └── evaluate.py      # MSE & PSNR calculation
├── gaussian/            # Placeholder for Gaussian noise module
├── speckle/             # Placeholder for Speckle noise module
├── utils/               # Image processing helper functions
└── static/              # Frontend Web UI (HTML/CSS/JS)
```

---

## 🎭 SUPPORTED NOISE MODELS

| Noise Type | Status | Detection | Filtering |
| :--- | :--- | :--- | :--- |
| **Salt & Pepper** | ✅ Active | Adaptive Threshold | Multilevel Median |
| **Gaussian** | 🚧 Roadmap | Basic Analytics | Placeholder |
| **Speckle** | 🚧 Roadmap | Basic Analytics | Placeholder |

---

## 📡 API REFERENCE - `POST /denoise`

**Input Type**: `multipart/form-data` with `file` field.

**Sample JSON Result**:
```json
{
  "detected_noise": "salt_pepper",
  "confidence": 0.85,
  "mse": 14.82,
  "psnr": 36.42,
  "processed_image": "data:image/png;base64,..."
}
```

---

*Academic-grade implementation by Antigravity.*