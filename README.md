# 🌌 Noise Characterization & Removal Toolkit

A sophisticated, production-ready system built with **FastAPI**, **OpenCV**, and **NumPy** to detect and eliminate image artifacts. This toolkit features a modular architecture that automatically diagnoses noise types and applies targeted filtering algorithms.

---

## 🚀 Key Features

- **🧠 Intelligent Noise Diagnosis**: Automatically analyzes image histograms and pixel distributions to determine noise characteristics.
- **🧂 Salt & Pepper Module (High-Fidelity)**:
    - **Detection**: Advanced tolerance-aware diagnosis for impulsive noise.
    - **Filtering**: High-performance Median Filter (7x7 kernel) for deep noise suppression.
    - **Evaluation**: Integrated MSE and PSNR metrics for quality verification.
- **⚡ Fast-API Backend**: Asynchronous processing with hot-reloading support for rapid experimentation.
- **✨ Premium UI**: Modern glassmorphism interface with drag-and-drop support and real-time before/after comparison.
- **🏗️ Modular Architecture**: Highly scalable project structure designed for easy integration of additional noise modules (Gaussian/Speckle).

---

## 📂 Project Structure

```text
noise_toolkit/
├── main.py             # API Core & Orchestration
├── salt_pepper/        # Impulsive Noise Implementation
│   ├── detect.py       # Tolerance-aware SNR detection
│   ├── filter.py       # Median Blur processing
│   └── evaluate.py     # Metrics (MSE/PSNR)
├── gaussian/           # Placeholder for Additive Noise
├── speckle/            # Placeholder for Multiplicative Noise
├── utils/              # Image transformation utilities
└── static/             # Frontend assets (index.html)
```

---

## 🛠️ Setup & Execution

### 1. Installation
Install the heavy-duty computer vision and web dependencies:
```bash
pip install -r requirements.txt
```

### 2. Launch the Engine
Start the local server with hot-reload enabled:
```bash
python main.py
```

### 3. Usage
- **Web UI**: Open `http://localhost:8000` in your browser.
- **API Sandbox**: Visit `http://localhost:8000/docs` (Swagger UI).

---

## 📡 API Reference - `POST /denoise`

| Field | Type | Description |
| :--- | :--- | :--- |
| `file` | Multipart | The raw image file to process. |

**Sample Response**:
```json
{
  "detected_noise": "salt_pepper",
  "confidence": 0.98,
  "mse": 12.45,
  "psnr": 35.21,
  "processed_image": "data:image/png;base64,..."
}
```

---

## 🧬 Evaluation Metrics

*   **MSE (Mean Squared Error)**: Quantifies the average squared difference between pixels. Lower is better.
*   **PSNR (Peak Signal-to-Noise Ratio)**: Expresses the ratio between the maximum possible power of a signal and the power of corrupting noise. Higher is better.

---

*Powered by Antigravity AI Engine & OpenCV.*