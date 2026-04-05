from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import os

# Internal module imports
from salt_pepper.detect import detect_salt_pepper
from salt_pepper.filter import filter_salt_pepper
from salt_pepper.evaluate import calculate_mse, calculate_psnr
from salt_pepper.visualize import generate_histogram_base64
from gaussian.placeholder import detect_gaussian, filter_gaussian
from speckle.placeholder import detect_speckle, filter_speckle
from utils.image_utils import decode_image, encode_image_base64

app = FastAPI(title="Noise Characterization and Removal Toolkit")

# Mount Static Files
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """
    Serve the main UI.
    """
    return FileResponse("static/index.html")

@app.post("/denoise")
async def denoise_image(file: UploadFile = File(...)):
    """
    Main endpoint for image denoising.
    Detects noise type, applies the best filter, and generates histograms.
    """
    try:
        # Load image from the uploaded file
        img_bytes = await file.read()
        image = decode_image(img_bytes)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image or could not be decoded.")
        
        # 1. Detection Phase
        sp_result = detect_salt_pepper(image)
        gauss_conf = detect_gaussian(image)
        speckle_conf = detect_speckle(image)
        
        # Compare confidences and pick the best candidate
        confidences = {
            "salt_pepper": sp_result["confidence"],
            "gaussian": gauss_conf,
            "speckle": speckle_conf
        }
        print(f"Debug - Detection Confidences: {confidences}")
        
        detected_noise = max(confidences, key=confidences.get)
        highest_confidence = confidences[detected_noise]
        
        # 2. Filtering Phase
        if detected_noise == "salt_pepper":
            # Only Salt & Pepper is fully implemented
            denoised_image = filter_salt_pepper(image)
        elif detected_noise == "gaussian":
            # Placeholder logic
            denoised_image = filter_gaussian(image)
        else: # speckle
            # Placeholder logic
            denoised_image = filter_speckle(image)
            
        # 3. Metrics (Comparing input image to the denoised output)
        mse = calculate_mse(image, denoised_image)
        psnr = calculate_psnr(mse)
        
        # 4. Visualization (Histograms)
        hist_before = ""
        hist_after = ""
        try:
            hist_before = generate_histogram_base64(image, "Original Photo Histogram")
            hist_after = generate_histogram_base64(denoised_image, "Cleaned Photo Histogram")
        except Exception as vis_err:
            print(f"Visualization error: {vis_err}")
        
        # 5. Result Packaging
        result_b64 = encode_image_base64(denoised_image)
        
        return {
            "detected_noise": detected_noise,
            "confidence": highest_confidence,
            "mse": mse,
            "psnr": psnr,
            "processed_image": f"data:image/png;base64,{result_b64}",
            "histogram_original": f"data:image/png;base64,{hist_before}" if hist_before else "",
            "histogram_denoised": f"data:image/png;base64,{hist_after}" if hist_after else ""
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"An error occurred: {str(e)}"})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
