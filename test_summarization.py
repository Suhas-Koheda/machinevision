import os
import sys

# Add the project root to sys.path for imports to work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.services.text_engine import get_incremental, get_summarizer, get_full_history, reset

def test_machine_vision_summarization():
    print("--- TESTING TEXT ENGINE & SUMMARIZATION ---")
    
    # 1. Simulate OCR stream (Longer, multi-sentence)
    sample_ocr_steps = [
        "Suhas Koheda is a student at VIT Chennai. He is staying at the hostel.",
        "The current room number is 302 in Block B.",
        "Today is 15-04-2026. The weather is sunny and clear.",
        "Security check completed at gate 1 at 16:12.",
        "User verified with biometric ID 23BAI1148.",
        "Lunch menu for the day includes Pav Bhaji and Rice.",
    ]
    
    print("\n[Step 1] Feeding OCR samples...")
    for text in sample_ocr_steps:
        # We manually append since get_incremental is for word-by-word diffing
        get_incremental(text)
        print(f"  + Logged: {text}")

    full_text = get_full_history()
    print(f"\n[Step 2] Full Accumulated History:\n{full_text}")

    # 2. Test Summarization
    print("\n[Step 3] Initializing Summarizer...")
    summarizer = get_summarizer()
    
    if summarizer.model:
        print("[Step 4] Running Extractive Summarization...")
        summary = summarizer.summarize(full_text, max_sentences=2)
        print(f"\nRESULT SUMMARY:\n{summary}")
    else:
        print("ERROR: Summarizer model could not be loaded. Check paths.")

if __name__ == "__main__":
    test_machine_vision_summarization()
