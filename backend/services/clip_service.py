import os
import torch
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import logging

logger = logging.getLogger(__name__)

class CLIPService:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Set offline mode environment variables to prevent connection attempts
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

        logger.info(f"*** INITIALIZING SEMANTIC ENGINE (CLIP) on {self.device} ***")
        try:
            # Try loading locally first to avoid connection errors
            self.model = CLIPModel.from_pretrained(model_name, use_safetensors=False, local_files_only=True).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
        except Exception as e:
            logger.warning(f"Local load failed ({e}), attempting standard load with connection timeout safety.")
            # Fallback to standard if local fails (e.g. first run)
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

    def get_image_embedding(self, pil_img):
        """
        Generate CLIP embedding for a full image or crop.
        Returns normalized numpy array.
        """
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            # Some versions return objects, some return tensors
            if torch.is_tensor(outputs):
                features = outputs
            elif hasattr(outputs, "image_embeds"):
                features = outputs.image_embeds
            elif hasattr(outputs, "pooler_output"):
                features = outputs.pooler_output
            else:
                # Fallback to first element if it's a sequence
                features = outputs[0] if hasattr(outputs, "__getitem__") else outputs
        
        # Final safety check: ensure we have a tensor with .norm
        if not hasattr(features, "norm"):
             raise AttributeError(f"CLIP feature extraction failed to produce a tensor. Got {type(features)}")

        # Normalize
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

_clip = None
def get_clip_service():
    global _clip
    if _clip is None:
        import cv2 # ensure cv2 is available for pil conversion
        _clip = CLIPService()
    return _clip
