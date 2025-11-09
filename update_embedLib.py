import os
import json
import torch
from sentence_transformers import SentenceTransformer
from textwrap import wrap

# ========= CONFIG =========
DRIVE_PATH = "/Volumes/New Volume"  # Change if Windows, e.g. "E:/NC_squaree"
LIBRARY_PATH = os.path.join(DRIVE_PATH, "embeddings_library")
EMBEDDING_FILE = os.path.join(LIBRARY_PATH, "embeddings.pt")
METADATA_FILE = os.path.join(LIBRARY_PATH, "metadata.json")

os.makedirs(LIBRARY_PATH, exist_ok=True)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ========= FUNCTION: Add OCR Text to Embedding Library =========
def add_ocr_text_to_library(ocr_json_path, book_name, book_id, institution, level_type, board=None):
    with open(ocr_json_path, "r") as f:
        ocr_data = json.load(f)
    
    full_text = ocr_data["extracted_text"]
    
    # Step 1: Clean text
    clean_text = (
        full_text.replace("--- Page", "")
        .replace("#", "")
        .replace("\n\n", "\n")
        .strip()
    )
    
    # Step 2: Split into semantic chunks (~200–300 tokens)
    chunks = wrap(clean_text, 1000)  # you can tune 1000 for paragraph-size segments
    
    # Step 3: Embed
    embeddings = model.encode(chunks, convert_to_tensor=True)
    
    # Step 4: Append to existing library (if exists)
    if os.path.exists(EMBEDDING_FILE):
        saved = torch.load(EMBEDDING_FILE)
        all_embeddings = torch.cat((saved["embeddings"], embeddings))
        all_texts = saved["texts"] + chunks
    else:
        all_embeddings = embeddings
        all_texts = chunks

    torch.save({"texts": all_texts, "embeddings": all_embeddings}, EMBEDDING_FILE)

    # Step 5: Append metadata
    meta_entry = {
        "book_name": book_name,
        "book_id": book_id,
        "institution": institution,
        "level_type": level_type,  # e.g. "school" or "college"
        "board": board,
        "source_file": os.path.basename(ocr_json_path),
        "num_chunks": len(chunks)
    }

    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
    else:
        metadata = []

    metadata.append(meta_entry)
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Added {len(chunks)} chunks from {book_name} (Book ID: {book_id})")

# ========= USAGE =========
ocr_json_path = input("Enter the path to the OCR JSON file: ").strip()
book_name = input("Enter the name of the book: ").strip()
book_id = input("Enter the book ID: ").strip()
institution = input("Enter the institution: ").strip()
level_type = input("Enter the level type (school or college): ").strip()
board = input("Enter the board (if applicable): ").strip()
add_ocr_text_to_library(
    ocr_json_path,
    book_name,
    book_id,
    institution,
    level_type,
    board,
)
