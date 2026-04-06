from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import os

# Internal module imports (Follow project/ Project Structure)
from salt_pepper.detect import detect_salt_pepper
from salt_pepper.filter import filter_salt_pepper
from salt_pepper.evaluate import calculate_mse, calculate_psnr
from gaussian.placeholder import detect_gaussian, filter_gaussian
from speckle.placeholder import detect_speckle, filter_speckle
from utils.image_utils import decode_image, encode_image_base64

app = FastAPI(title="Noise Characterization & Removal Production Core")

# Mount Static Files (Serving index.html at root)
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
    Detects noise type and applies the best filter using adaptive logic.
    """
    try:
        # Load image and convert to grayscale using OpenCV logic
        img_bytes = await file.read()
        image = decode_image(img_bytes)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image or could not be decoded.")
        
        # 1. Detection Phase
        sp_result = detect_salt_pepper(image)
        gauss_conf = detect_gaussian(image)
        speckle_conf = detect_speckle(image)
        
        # Selection Logic (Strict salt_pepper validation)
        if sp_result["confidence"] > 0:
            detected_noise = "salt_pepper"
            highest_confidence = sp_result["confidence"]
        else:
            detected_noise = "none"
            highest_confidence = 0.0
        
        # Init results
        denoised_image = image
        mse, psnr = -1.0, -1.0 # Default/Fail-safe for placeholders
        
        # 2. Adaptive Filtering Phase
        if detected_noise == "salt_pepper" and sp_result["noise_ratio"] >= 0.01:
            # Fully implemented and adaptive module
            denoised_image = filter_salt_pepper(image, sp_result["noise_ratio"])
            
            # Accurate metrics calculation based on image transformation
            mse = calculate_mse(image, denoised_image)
            psnr = calculate_psnr(mse)
            
        else:
            # Noise below threshold or non-SP: Return original image with negative metrics
            denoised_image = image
            mse = -1.0
            psnr = -1.0
            
        # 3. Packaging
        result_b64 = encode_image_base64(denoised_image)
        
        return {
            "detected_noise": detected_noise,
            "confidence": highest_confidence,
            "mse": mse,
            "psnr": psnr,
            "processed_image": f"data:image/png;base64,{result_b64}"
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Internal process failure: {str(e)}"})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
