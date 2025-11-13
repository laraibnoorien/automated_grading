"""
C_sent_embedding.py
Embeds:
- Book + Reference → persistent library
- Student → temporary only
"""

import os
import json
import torch
from datetime import datetime
from sentence_transformers import SentenceTransformer
from B_normalize import full_clean

# -------- CONFIG --------
MODEL_NAME = "intfloat/e5-base-v2"
HDD_PATH = "/Volumes/New Volume/models/e5-base-v2"
EMBEDDING_DB_PATH = "embeddings_library.pt"
TEMP_STUDENT_EMB_PATH = "temp_student_embeddings.pt"
# ------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

try:
    model = SentenceTransformer(HDD_PATH, device=device)
except:
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.save(HDD_PATH)


def load_book_chunks(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = []
    for chapter in data:
        for topic in chapter.get("topics", []):
            for c in topic.get("chunks", []):
                if c.strip():
                    chunks.append(c.strip())
    return chunks


def flatten_answer_parts(parts):
    if isinstance(parts, str):
        return parts
    if isinstance(parts, dict):
        return " ".join(flatten_answer_parts(v) for v in parts.values())
    return ""


def load_ref_or_student(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = []
    for ans in data:
        p = ans.get("answer_parts", {})
        text = flatten_answer_parts(p)
        text = full_clean(text)
        if text:
            out.append(text)
    return out


def embed_text(sentences, category):
    if category == "student_answer":
        sentences = ["query: " + s for s in sentences]
    else:
        sentences = ["passage: " + s for s in sentences]

    emb = model.encode(sentences, convert_to_tensor=True, device=device, show_progress_bar=True)
    return emb, sentences


def save_to_library(school_type, subject, board, class_name, category, sentences, embeddings):
    if os.path.exists(EMBEDDING_DB_PATH):
        db = torch.load(EMBEDDING_DB_PATH)
    else:
        db = {}

    db.setdefault(school_type, {}).setdefault(subject, {})

    if school_type == "school":
        db[school_type][subject].setdefault(board, {}).setdefault(class_name, {}).setdefault(category, [])
        folder = db[school_type][subject][board][class_name][category]
    else:
        db[school_type][subject].setdefault(category, [])
        folder = db[school_type][subject][category]

    existing = set()
    for e in folder:
        existing.update(e["sentences"])

    new_sents, new_embeds = [], []
    for s, e in zip(sentences, embeddings):
        if s not in existing:
            new_sents.append(s)
            new_embeds.append(e)

    if not new_sents:
        print("[=] No new entries.")
        return

    entry = {
        "timestamp": datetime.now().isoformat(),
        "sentences": new_sents,
        "embeddings": torch.stack(new_embeds)
    }

    folder.append(entry)
    torch.save(db, EMBEDDING_DB_PATH)
    print(f"[+] Saved {len(new_sents)} new {category} items.")


def save_temp_student(sentences, embeddings):
    torch.save({"sentences": sentences, "embeddings": embeddings}, TEMP_STUDENT_EMB_PATH)
    print(f"[+] Student temp embeddings saved → {TEMP_STUDENT_EMB_PATH}")


def embed_file(json_path, school_type, subject, board, class_name, category):
    print(f"\n=== Embedding: {category.upper()} ===")

    if category == "book":
        sentences = load_book_chunks(json_path)
    else:
        sentences = load_ref_or_student(json_path)

    if not sentences:
        print("[-] Empty input text.")
        return

    embeddings, sentences = embed_text(sentences, category)

    if category == "student_answer":
        save_temp_student(sentences, embeddings)
    else:
        save_to_library(school_type, subject, board, class_name, category, sentences, embeddings)


if __name__ == "__main__":
    school_type = input("school / college: ").strip().lower()

    if school_type == "school":
        board = input("board: ").strip()
        class_name = input("class: ").strip()
        subject = input("subject: ").strip()
    else:
        board = None
        class_name = None
        subject = input("subject: ").strip()

    FILES = {
        "student_answer": "regrex_student_answer.json",
        "reference_answer": "regrex_reference_answer.json",
        "book": "regrex_book.json",
    }

    print("\n1 = Student\n2 = Reference\n3 = Book\n4 = Reference + Book\n")
    choice = input("choice: ").strip()

    mapping = {
        "1": ["student_answer"],
        "2": ["reference_answer"],
        "3": ["book"],
        "4": ["reference_answer", "student_answer"],
    }

    tasks = mapping.get(choice, [])
    if not tasks:
        print("Invalid.")
        exit()

    for key in tasks:
        path = FILES[key]
        if os.path.exists(path):
            embed_file(path, school_type, subject, board, class_name, key)
        else:
            print(f"Missing file: {path}")
