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
from gaussian.detect import detect_gaussian
from gaussian.filter import filter_gaussian
from gaussian.evaluate import calculate_mse as calculate_mse_gaussian, calculate_psnr as calculate_psnr_gaussian
from speckle.placeholder import detect_speckle, filter_speckle
from utils.image_utils import decode_image, encode_image_base64
import cv2

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
    AUTOMATED Denoising Endpoint
    
    Intelligent, fully automatic image denoising framework that:
    1. DETECTS the type of noise present (Salt & Pepper, Gaussian, or None)
    2. CHOOSES the best filtering technique automatically
    3. APPLIES optimal parameters based on noise intensity
    4. EVALUATES quality using MSE and PSNR metrics
    
    NO manual selection required - the framework decides everything!
    """
    try:
        # Load and decode image
        img_bytes = await file.read()
        image = decode_image(img_bytes)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image or could not be decoded.")
        
        # ====== DETECTION PHASE ======
        # Comprehensive noise type detection
        sp_result = detect_salt_pepper(image)
        gauss_result = detect_gaussian(image)
        
        sp_confidence = sp_result["confidence"]
        sp_noise_ratio = sp_result["noise_ratio"]
        gauss_confidence = gauss_result["confidence"]
        gauss_noise_ratio = gauss_result["noise_ratio"]
        
        # ====== DECISION LOGIC ======
        # Automatic noise type ranking and selection
        noise_scores = {
            "salt_pepper": sp_confidence if sp_noise_ratio >= 0.003 else 0.0,
            "gaussian": gauss_confidence,
            "none": 0.0
        }
        
        # Select the noise type with highest confidence
        detected_noise = max(noise_scores, key=noise_scores.get)
        highest_confidence = noise_scores[detected_noise]
        
        # Initialize results
        denoised_image = image
        mse, psnr = -1.0, -1.0
        filter_method = "none"
        
        # ====== FILTERING PHASE ======
        # Apply optimal filter based on detected noise type
        
        if detected_noise == "salt_pepper" and sp_noise_ratio >= 0.005:
            # Salt & Pepper noise detected
            denoised_image = filter_salt_pepper(image, sp_noise_ratio)
            mse = calculate_mse(image, denoised_image)
            psnr = calculate_psnr(mse)
            filter_method = "median_filter"
            
        elif detected_noise == "gaussian" and gauss_noise_ratio > 0.01:
            # Gaussian noise detected - use bilateral filter for best edge preservation
            denoised_image = filter_gaussian(image, gauss_noise_ratio, method="bilateral")
            mse = calculate_mse_gaussian(image, denoised_image)
            psnr = calculate_psnr_gaussian(mse)
            filter_method = "bilateral_filter"
            
        else:
            # No significant noise detected
            denoised_image = image
            mse = -1.0
            psnr = -1.0
            filter_method = "none"
        
        # ====== PACKAGING RESPONSE ======
        result_b64 = encode_image_base64(denoised_image)
        
        return {
            "detected_noise": detected_noise,
            "confidence": float(highest_confidence),
            "noise_details": {
                "salt_pepper": {
                    "confidence": float(sp_confidence),
                    "ratio": float(sp_noise_ratio),
                    "pepper_pixels": int(sp_result["pepper_count"]),
                    "salt_pixels": int(sp_result["salt_count"])
                },
                "gaussian": {
                    "confidence": float(gauss_confidence),
                    "ratio": float(gauss_noise_ratio),
                    "variance": float(gauss_result["variance"]),
                    "laplacian_variance": float(gauss_result["laplacian_variance"])
                }
            },
            "filter_applied": filter_method,
            "mse": float(mse),
            "psnr": float(psnr),
            "processed_image": f"data:image/png;base64,{result_b64}"
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Internal process failure: {str(e)}"})



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
