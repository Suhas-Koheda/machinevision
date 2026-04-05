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

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Core**:
   ```bash
   python main.py
   ```

3. **Access**:
   - Web UI: `http://localhost:8000`
   - API Docs: `http://localhost:8000/docs`

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