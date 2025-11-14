import faiss
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Paths must match local_embed_faiss.py
FAISS_INDEX_PATH = "./faiss_index.index"
META_PATH = "./faiss_metadata.json"
MODEL_CACHE = "./model_cache"
EMBED_DIM = 768

STUDENT_REGREX_PATH = "./regrex_student_answer.json"


# -----------------------------
# Load model (same as embedder)
# -----------------------------
def load_model():
    try:
        if Path(MODEL_CACHE).exists() and any(Path(MODEL_CACHE).iterdir()):
            model = SentenceTransformer(MODEL_CACHE)
        else:
            model = SentenceTransformer("intfloat/e5-base-v2")
    except:
        model = SentenceTransformer("intfloat/e5-base-v2")

    # MPS / CUDA (optional)
    try:
        import torch
        if torch.backends.mps.is_available():
            model.to("mps")
        elif torch.cuda.is_available():
            model.to("cuda")
    except:
        pass

    return model


model = load_model()


# -----------------------------
# Normalize embeddings
# -----------------------------
def normalize(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1
    return v / n


# -----------------------------
# Safe FAISS loader (never crashes)
# -----------------------------
def load_faiss():
    if not Path(FAISS_INDEX_PATH).exists():
        print("[!] FAISS index missing → creating EMPTY index.")
        index = faiss.IndexHNSWFlat(EMBED_DIM, 32)
        faiss.write_index(index, FAISS_INDEX_PATH)
        return index

    try:
        return faiss.read_index(FAISS_INDEX_PATH)
    except:
        print("[!] FAISS index corrupted → recreating EMPTY index.")
        index = faiss.IndexHNSWFlat(EMBED_DIM, 32)
        faiss.write_index(index, FAISS_INDEX_PATH)
        return index


# -----------------------------
# Safe metadata loader (never crashes)
# -----------------------------
def load_meta():
    if not Path(META_PATH).exists():
        print("[!] Metadata missing → creating empty metadata list.")
        json.dump([], open(META_PATH, "w"))
        return []

    try:
        return json.load(open(META_PATH))
    except:
        print("[!] Metadata corrupted → resetting to empty list.")
        json.dump([], open(META_PATH, "w"))
        return []


# -----------------------------
# FAISS search wrapper
# -----------------------------
def search_student_answer(answer_text: str, k: int = 5):
    index = load_faiss()
    meta = load_meta()

    # If empty DB
    if index.ntotal == 0 or len(meta) == 0:
        print("[!] No book/reference embeddings found. Run local_embed_faiss.py first.")
        return []

    # If mismatch, safely truncate
    if index.ntotal != len(meta):
        print(f"[!] Warning: FAISS rows={index.ntotal} but metadata={len(meta)}.")
        print("    Using the smaller size for safe search.")
        min_len = min(index.ntotal, len(meta))
        meta = meta[:min_len]

    q = "query: " + answer_text.strip()

    emb = model.encode([q], convert_to_numpy=True)
    emb = normalize(emb.astype("float32"))

    distances, idxs = index.search(emb, min(k, index.ntotal))

    results = []
    for i, score in zip(idxs[0], distances[0]):
        if 0 <= i < len(meta):
            entry = meta[i].copy()
            entry["score"] = float(score)
            results.append(entry)

    return results


# -----------------------------
# Batch processing: Option B
# -----------------------------
def run_batch(top_k=5):
    if not Path(STUDENT_REGREX_PATH).exists():
        print(f"ERROR: {STUDENT_REGREX_PATH} not found.")
        return

    student_answers = json.load(open(STUDENT_REGREX_PATH, "r"))

    print(f"\nFound {len(student_answers)} student answers. Running FAISS search...\n")

    for ans in student_answers:
        qnum = ans.get("question_number", "?")
        text = ans.get("answer") or ans.get("raw_answer") or ""
        text = text.strip()

        print("\n" + "="*60)
        print(f"QUESTION {qnum} — Student Answer:")
        print(text[:400], "..." if len(text) > 400 else "")
        print("="*60)

        if not text:
            print("Empty answer — skipping.\n")
            continue

        # run FAISS search
        results = search_student_answer(text, k=top_k)

        if not results:
            print("No matches found.\n")
            continue

        print("\nTop Matches:")
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] score={r['score']:.4f}")
            print(f"FILE: {r['file']} | ROLE: {r['role']}")
            print(r["text"][:300], "...")



# -----------------------------
# CLI entrypoint
# -----------------------------
if __name__ == "__main__":
    print("Running FAISS search for ALL student answers in regrex_student_answer.json...")
    run_batch(top_k=5)
