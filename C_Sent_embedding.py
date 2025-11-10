"""
unified_embedder_pipeline.py
-----------------------------
Embeds OCR-derived text files (Book / Reference Answers / Student Answers)
and stores them appropriately.

- Book & Reference → persistent embedding library (.pt)
- Student Answers → temporary embeddings for evaluation only
"""

import os
import json
import torch
from datetime import datetime
from sentence_transformers import SentenceTransformer

# ======= CONFIG =======
MODEL_NAME = "intfloat/e5-base-v2"  # superior for semantic & QA embeddings
HDD_PATH = "/Volumes/New Volume/models/e5-base-v2"
EMBEDDING_DB_PATH = "embeddings_library.pt"
TEMP_STUDENT_EMB_PATH = "temp_student_embeddings.pt"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
# ======================

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using NVIDIA GPU (CUDA)")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU (no GPU detected)")

# ---------- LOAD MODEL ----------
try:
    model = SentenceTransformer(HDD_PATH, device=device)
    print("[*] Loaded E5 model from HDD cache")
except:
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.save(HDD_PATH)
    print("[*] Downloaded and saved E5 model to HDD")

# ---------- HELPERS ----------
def chunk_with_overlap(words, size=200, overlap=50):
    """Chunk words into overlapping segments."""
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def load_book_text(json_path):
    """Load OCR book JSON and split into overlapping chunks."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    topics_list = []
    for chapter in data:
        chapter_title = chapter.get("chapter_title", "")
        for topic in chapter.get("topics", []):
            content = topic.get("content", "")
            words = content.split()
            chunks = chunk_with_overlap(words, CHUNK_SIZE, CHUNK_OVERLAP)
            topics_list.append({
                "chapter_title": chapter_title,
                "topic_title": topic.get("topic_title", ""),
                "content": content,
                "chunks": chunks
            })
    return topics_list


def load_student_text(json_path):
    """Load and flatten student or reference answers."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentences = []
    for ans in data:
        parts = ans.get("answer_parts", ans.get("parts", {}))

        def flatten(p):
            if isinstance(p, str):
                return p
            elif isinstance(p, dict):
                return " ".join(flatten(v) for v in p.values())
            return ""

        text = flatten(parts)
        if text.strip():
            sentences.append(text.strip())

    return sentences


def generate_embeddings(sentences, model, category):
    """Encode sentences with appropriate prefix for E5 model."""
    if category == "student_answer":
        sentences = ["query: " + s for s in sentences]
    else:
        sentences = ["passage: " + s for s in sentences]

    print(f"[*] Encoding {len(sentences)} {category} sentences...")
    embeddings = model.encode(
        sentences,
        convert_to_tensor=True,
        device=device,
        show_progress_bar=True
    )
    return embeddings, sentences


# ---------- SAVE FUNCTIONS ----------
def save_to_library(school_type, subject, board, class_name, category, sentences, embeddings):
    """Save persistent embeddings (book/reference)."""
    if os.path.exists(EMBEDDING_DB_PATH):
        data = torch.load(EMBEDDING_DB_PATH)
    else:
        data = {}

    # Build nested structure
    data.setdefault(school_type, {}).setdefault(subject, {})
    if school_type == "school":
        data[school_type][subject].setdefault(board, {}).setdefault(class_name, {}).setdefault(category, [])
        entry_list = data[school_type][subject][board][class_name][category]
    else:
        data[school_type][subject].setdefault(category, [])
        entry_list = data[school_type][subject][category]

    # Avoid duplicates
    existing_sentences = set()
    for e in entry_list:
        existing_sentences.update(e["sentences"])

    new_sents, new_embeds = [], []
    for s, e in zip(sentences, embeddings):
        if s not in existing_sentences:
            new_sents.append(s)
            new_embeds.append(e)

    if not new_sents:
        print(f"[=] No new {category} data to embed.")
        return

    entry = {
        "timestamp": datetime.now().isoformat(),
        "sentences": new_sents,
        "embeddings": torch.stack(new_embeds)
    }

    entry_list.append(entry)
    torch.save(data, EMBEDDING_DB_PATH)
    print(f"[+] Added {len(new_sents)} new {category} entries to {school_type}/{subject}")


def save_temp_student_embeddings(sentences, embeddings):
    """Save temporary student embeddings for evaluation."""
    torch.save({"sentences": sentences, "embeddings": embeddings}, TEMP_STUDENT_EMB_PATH)
    print(f"[+] Temporary student embeddings saved → {TEMP_STUDENT_EMB_PATH}")


# ---------- MAIN PIPELINE ----------
def embed_file(json_path, school_type, subject, board, class_name, category):
    """Embed a given JSON file depending on category."""
    print(f"\n=== Embedding {category.upper()} ===")

    if category == "book":
        topics = load_book_text(json_path)
        sentences = [chunk for t in topics for chunk in t["chunks"]]
    else:
        sentences = load_student_text(json_path)

    if not sentences:
        print(f"[-] No valid text in {json_path}")
        return

    embeddings, sentences = generate_embeddings(sentences, model, category)

    if category == "student_answer":
        save_temp_student_embeddings(sentences, embeddings)
    else:
        save_to_library(school_type, subject, board, class_name, category, sentences, embeddings)


# ---------- INTERACTIVE ----------
if __name__ == "__main__":
    school_type = input("Enter institution type (school / college): ").strip().lower()

    if school_type == "school":
        board = input("Enter board: ").strip()
        class_name = input("Enter class: ").strip()
        subject = input("Enter subject: ").strip()
    elif school_type == "college":
        subject = input("Enter subject: ").strip()
        board = None
        class_name = None
    else:
        print("[-] Invalid institution type.")
        exit(1)

    FILES = {
        "student_answer": "regrex_student_answer.json",
        "reference_answer": "regrex_reference_answer.json",
        "book": "regrex_book.json",
    }

    print("\nSelect what to embed:")
    print("1 → Student Answers (temporary only)")
    print("2 → Reference Answers (persistent)")
    print("3 → Book (persistent)")
    print("4 → Reference + Book")

    choice = input("\nEnter choice (1–4): ").strip()
    mapping = {
        "1": ["student_answer"],
        "2": ["reference_answer"],
        "3": ["book"],
        "4": ["reference_answer", "book"],
    }

    selected = mapping.get(choice)
    if not selected:
        print("[-] Invalid choice.")
        exit(1)

    for key in selected:
        path = FILES[key]
        if os.path.exists(path):
            embed_file(path, school_type, subject, board, class_name, key)
        else:
            print(f"[-] Missing file: {path}")
