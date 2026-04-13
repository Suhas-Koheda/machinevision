import torch
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import logging

logger = logging.getLogger(__name__)

class CLIPService:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"*** INITIALIZING SEMANTIC ENGINE (CLIP) on {self.device} ***")
        # Forcing use_safetensors=False to avoid re-downloading 600MB if user already has .bin version
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=False).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.prompts = ["person sitting", "person standing", "person walking", "person using laptop", "empty chair"]

    def analyze_crops(self, frame, detections):
        """
        Crop detections and classify using CLIP.
        """
        results = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            if x2-x1 < 10 or y2-y1 < 10: continue
            
            crop = frame[y1:y2, x1:x2]
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            
            inputs = self.processor(text=self.prompts, images=pil_img, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model(**inputs)
            
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            best_idx = probs.argmax().item()
            results.append({
                "track_id": det.get("track_id", -1),
                "semantic": self.prompts[best_idx],
                "confidence": round(float(probs[0][best_idx].detach().cpu().item()), 3)
            })

        return results

_clip = None
def get_clip_service():
    global _clip
    if _clip is None:
        import cv2 # ensure cv2 is available for pil conversion
        _clip = CLIPService()
    return _clip
