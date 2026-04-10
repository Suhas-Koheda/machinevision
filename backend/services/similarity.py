from sentence_transformers import SentenceTransformer, util
import torch

class TextSimilarityEngine:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.last_embedding = None
        self.last_text = ""

    def is_semantically_different(self, text: str, threshold: float = 0.9) -> bool:
        if not text.strip():
            return False
        
        current_embedding = self.model.encode(text, convert_to_tensor=True)
        
        if self.last_embedding is None:
            self.last_embedding = current_embedding
            self.last_text = text
            return True
        
        similarity = util.cos_sim(self.last_embedding, current_embedding).item()
        
        if similarity < threshold:
            self.last_embedding = current_embedding
            self.last_text = text
            return True
        
        return False

    def reset(self):
        self.last_embedding = None
        self.last_text = ""
