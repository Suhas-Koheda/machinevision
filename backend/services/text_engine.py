"""
Text Engine — Intelligent incremental text tracker and deduplicator.
Avoids repeating phrases or substrings in OCR output.
"""
import re
import logging
import torch
from difflib import SequenceMatcher
from transformers import pipeline

logger = logging.getLogger(__name__)

_last_text: str = ""
_history: list = []
_summarizer: 'TextSummarizer' = None
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

class TextSummarizer:
    def __init__(self, model_path="/home/ssp/ML/proejct/py/.model_cache/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"*** INITIALIZING EXTRACTIVE SUMMARIZER (MiniLM) on {self.device} ***")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_path, local_files_only=True).to(self.device)
        except Exception as e:
            logger.error(f"Failed to load local MiniLM: {e}. Summarization will be basic.")
            self.model = None

    def _get_embedding(self, text):
        if not self.model: return None
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean Pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return F.normalize(embeddings, p=2, dim=1)

    def summarize(self, text: str, max_sentences=3) -> str:
        if not text or len(text.split()) < 20:
            return text
        
        if not self.model:
            return text[:200] + "..."

        # 1. Split into sentences (simple heuristic)
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s for s in sentences if len(s.split()) > 3] # Filter short ones
        
        if len(sentences) <= max_sentences:
            return text

        # 2. Get embeddings for each sentence and the whole text
        sent_embeds = [self._get_embedding(s) for s in sentences]
        doc_embed = self._get_embedding(text)

        # 3. Calculate similarity scores (cosine similarity)
        scores = []
        for embed in sent_embeds:
            score = torch.mm(embed, doc_embed.transpose(0, 1)).item()
            scores.append(score)

        # 4. Pick top N sentences (preserving original order)
        top_indices = np.argsort(scores)[-max_sentences:]
        top_indices.sort()
        
        summary = " ".join([sentences[i] for i in top_indices])
        return summary

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = TextSummarizer()
    return _summarizer

def fuzzy_match(a, b):
    return SequenceMatcher(None, a, b).ratio()

def get_incremental(new_text: str) -> str:
    """
    Returns only the genuinely new portion of text compared to global state.
    Uses fuzzy matching and word-based deduplication.
    """
    global _last_text, _history

    if not new_text:
        return ""

    new_text = new_text.strip()
    
    # If the exact text was just seen, skip
    if new_text == _last_text:
        return ""

    # Check for direct overlap: 'Abc Def' + 'Def Ghi' -> 'Ghi'
    words = new_text.split()
    unique_new_words = []
    
    # Filter out words that were already present in the "last" composite view
    # to avoid the OCR "echo" effect.
    for word in words:
        if word.lower() not in _last_text.lower():
            unique_new_words.append(word)
    
    if not unique_new_words:
        return ""
    
    added_text = " ".join(unique_new_words)
    _last_text += " " + added_text
    _last_text = _last_text.strip()
    
    _history.append(added_text)
    
    return added_text

def get_full_history() -> str:
    return " ".join(_history)

def reset():
    global _last_text, _history
    _last_text = ""
    _history = []
