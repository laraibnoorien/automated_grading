"""
book_embedder_pipeline.py
--------------------------
Embeds OCR text files (Book / Student Answers / Reference Answers)
and stores them in a persistent library with subject, board, and class labels.

Now includes an interactive prompt for selecting which files to embed.
"""

import json
import os
import torch
from sentence_transformers import SentenceTransformer
from datetime import datetime

# ======= CONFIGURATION =======
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DB_PATH = "embeddings_library.pt"  # persistent store
# =============================

def load_json_text(json_path):
    """Load text content from either OCR-style or structured QA JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Case 1: OCR raw text
    if isinstance(data, dict) and "extracted_text" in data:
        text = data["extracted_text"].strip()

    # Case 2: structured QA list
    elif isinstance(data, list):
        lines = []
        for q in data:
            q_num = q.get("question_number", "")
            parts = q.get("parts", {})
            lines.append(f"Q{q_num}")
            for key, val in parts.items():
                if isinstance(val, dict):
                    for sub_k, sub_v in val.items():
                        lines.append(f"{key}{sub_k}: {sub_v}")
                else:
                    lines.append(f"{key}: {val}")
        text = "\n".join(lines)

    else:
        text = ""

    return text

def preprocess_text(raw_text):
    """Split text into lines or sentences."""
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    return lines

def generate_embeddings(sentences, model):
    """Generate tensor embeddings for given sentences."""
    return model.encode(sentences, convert_to_tensor=True, show_progress_bar=True)

def save_embeddings(subject, board, class_name, category, sentences, embeddings):
    """Append embeddings to a persistent library (torch file)."""
    if os.path.exists(EMBEDDING_DB_PATH):
        data = torch.load(EMBEDDING_DB_PATH)
    else:
        data = {}

    # Ensure hierarchical structure
    if subject not in data:
        data[subject] = {}
    if board not in data[subject]:
        data[subject][board] = {}
    if class_name not in data[subject][board]:
        data[subject][board][class_name] = {}
    if category not in data[subject][board][class_name]:
        data[subject][board][class_name][category] = []

    # Store new entry with metadata
    entry = {
        "timestamp": datetime.now().isoformat(),
        "sentences": sentences,
        "embeddings": embeddings,
    }

    data[subject][board][class_name][category].append(entry)

    torch.save(data, EMBEDDING_DB_PATH)
    print(f"[+] Saved {len(sentences)} {category} embeddings under {subject}/{board}/Class {class_name}")

def embed_file(json_path, subject, board, class_name, category):
    """Main embedding function."""
    model = SentenceTransformer(MODEL_NAME)
    print(f"[*] Embedding {category} from {json_path}")

    text = load_json_text(json_path)
    sentences = preprocess_text(text)

    embeddings = generate_embeddings(sentences, model)
    save_embeddings(subject, board, class_name, category, sentences, embeddings)


# ======= INTERACTIVE MAIN PIPELINE =======
if __name__ == "__main__":
    SUBJECT = input("Enter subject: ").strip()
    BOARD = input("Enter board: ").strip()
    CLASS = input("Enter class: ").strip()

    FILES = {
        "regrex_student_answer": "regrex_student_answer.json",
        "regrex_reference_answer": "regrex_reference_answer.json",
        "regrex_book": "regrex_book.json",
    }

    print("\nSelect what to embed:")
    print("1 → regrex_student_answer")
    print("2 → regrex_reference_answer")
    print("3 → regrex_book")
    print("4 → both 1 and 2")
    print("5 → both 1 and 3")

    choice = input("\nEnter your choice (1-5): ").strip()

    mapping = {
        "1": ["regrex_student_answer"],
        "2": ["regrex_reference_answer"],
        "3": ["regrex_book"],
        "4": ["regrex_student_answer", "regrex_reference_answer"],
        "5": ["regrex_student_answer", "regrex_book"],
    }

    selected = mapping.get(choice)

    if not selected:
        print("[-] Invalid choice. Exiting.")
        exit(1)

    for key in selected:
        path = FILES.get(key)
        if os.path.exists(path):
            embed_file(path, SUBJECT, BOARD, CLASS, key)
        else:
            print(f"[-] Missing file: {path}")
