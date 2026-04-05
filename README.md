# 🌉 Noise Characterization & Removal Toolkit

A sophisticated system for image noise diagnosis and suppression using **FastAPI**, **OpenCV**, **NumPy**, and **Matplotlib**.

## 🚀 FEATURE OVERVIEW

- **🧠 Auto-Noise Detection**: Analyzes pixel distributions to identify artifacts.
- **🧂 Salt & Pepper Module (FULL)**: 
    - **Detection**: Histogram-based outlier analysis (0 and 255 intensity peaks).
    - **Filtering**: OpenCV Median Filter (`cv2.medianBlur`).
    - **Evaluation**: Automatic MSE and PSNR scoring.
- **📊 Advanced Visualization**: 
    - Real-time Histogram generation (Before vs. After).
    - Side-by-side comparison of the denoising effect.
- **🏗️ Modular Architecture**: Separate modules for different noise types and utilities.

---

## 📂 PROJECT STRUCTURE

```text
noise_toolkit/
├── main.py             # FastAPI Core
├── salt_pepper/        # S&P Implementation
│   ├── detect.py       # Detection logic
│   ├── filter.py       # Median filtering
│   ├── evaluate.py     # Metrics (MSE/PSNR)
│   └── visualize.py    # Matplotlib histograms
├── gaussian/           # Placeholder
├── speckle/            # Placeholder
├── utils/              # Conversion helpers
└── static/             # Premium UI
```

---

## 🛠️ HOW TO RUN

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Server**:
   ```bash
   python main.py
   ```

3. **Open UI**:
   Navigate to `http://localhost:8000` in your browser.

---

## 📡 API REFERENCE

### `POST /denoise`
**Input**: Multipart `file` (image).
**Output**: 
- `detected_noise`: (string)
- `mse`/`psnr`: (float)
- `processed_image`: (base64 string)
- `histogram_original`: (base64 graph)
- `histogram_denoised`: (base64 graph)

---

*Powered by Antigravity AI & OpenCV.*