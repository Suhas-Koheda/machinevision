"""
Text Engine — Intelligent incremental text tracker and deduplicator.
Avoids repeating phrases or substrings in OCR output.
"""
import re
from difflib import SequenceMatcher

_last_text: str = ""
_history: list = []

def fuzzy_match(a, b):
    return SequenceMatcher(None, a, b).ratio()

def get_incremental(new_text: str) -> str:
    """
    Returns only the genuinely new portion of text compared to global state.
    Uses fuzzy matching and substring detection to avoid repeats like:
    'Suhas Koheda' -> 'Suhas Koheda 23BA' -> '23BA'
    """
    global _last_text

    if not new_text:
        return ""

    new_text = new_text.strip()
    
    # If the exact text was just seen, skip
    if new_text == _last_text:
        return ""

    # Check for direct overlap: 'Abc Def' + 'Def Ghi' -> 'Ghi'
    # Or 'Suhas Koheda' + 'Suhas Koheda 23BA' -> '23BA'
    
    # 1. Startswith check (standard incremental)
    if new_text.startswith(_last_text) and _last_text:
        added = new_text[len(_last_text):].strip()
        if added:
            _last_text = new_text
            return added
        return ""

    # 2. Overlap check (fuzzy/suffix)
    # Try to find the overlap between the end of history and start of new text
    best_overlap = 0
    words = new_text.split()
    
    # Check if the first few words of new_text already exist in the last_text
    # e.g. last: "Suhas Koheda" new: "Suhas Koheda 123"
    # we want to return "123"
    
    # Actually, a simpler way for the user's specific problem:
    # "Suhas Koheda 23BA11148 HOSTELLER" repeated.
    # We can use a rolling buffer of unique words/phrases.
    
    words = new_text.split()
    unique_new_words = []
    
    for word in words:
        if word.lower() not in _last_text.lower():
            unique_new_words.append(word)
    
    if not unique_new_words:
        return ""
    
    added_text = " ".join(unique_new_words)
    _last_text += " " + added_text
    _last_text = _last_text.strip()
    
    return added_text

def reset():
    global _last_text
    _last_text = ""
