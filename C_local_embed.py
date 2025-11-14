#!/usr/bin/env python3
"""
local_embed_faiss.py (improved)

- Book + Reference -> permanent FAISS (faiss_index.index + faiss_metadata.json)
- Student -> temporary embeddings (in-memory; optional save)
- Safe for macOS (IndexHNSWFlat, omp single-thread)
- Uses normalize_book_text (B_normalize_book) for books and full_clean (B_normalize_answer) for answers
- Provides search helper for quick testing
"""

import os
import json
import faiss
import numpy as np
import hashlib
from pathlib import Path
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import nltk
import traceback

# ensure punkt
try:
    _ = sent_tokenize("hello")
except LookupError:
    nltk.download("punkt")

# -------------------
# Config
# -------------------
MODEL_NAME = "intfloat/e5-base-v2"
MODEL_CACHE = "./model_cache"

FAISS_INDEX_PATH = "./faiss_index.index"
META_PATH = "./faiss_metadata.json"
TEMP_STUDENT_EMB_PATH = "./temp_student_embeddings.json"

EMBED_DIM = 768
BATCH = 16

# chunk sizes tuned for good retrieval
CHUNK_MAX_WORDS = 45
CHUNK_OVERLAP = 10

# macOS safety
os.environ["TOKENIZERS_PARALLELISM"] = "false"
faiss.omp_set_num_threads(1)

Path(MODEL_CACHE).mkdir(parents=True, exist_ok=True)

# -------------------
# Normalizers — external modules you already have
# -------------------
try:
    from B_normalize_book import normalize_book_text
except Exception:
    def normalize_book_text(x): return x  # noop fallback

try:
    from B_normalize_answer import full_clean
except Exception:
    try:
        from B_normalize_book import full_clean
    except Exception:
        def full_clean(x): return x  # noop fallback

# -------------------
# Model loader
# -------------------
def load_model():
    if Path(MODEL_CACHE).exists() and any(Path(MODEL_CACHE).iterdir()):
        model = SentenceTransformer(MODEL_CACHE)
    else:
        model = SentenceTransformer(MODEL_NAME)
        try:
            model.save(MODEL_CACHE)
        except Exception:
            pass
    # try to move to GPU/MPS if available (sentence-transformers may ignore .to())
    try:
        import torch
        if torch.backends.mps.is_available():
            model.to("mps")
        elif torch.cuda.is_available():
            model.to("cuda")
    except Exception:
        pass
    return model

model = load_model()

# -------------------
# Utilities
# -------------------
def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def chunk_text(text: str, max_words=CHUNK_MAX_WORDS, overlap=CHUNK_OVERLAP):
    text = text.strip()
    if not text:
        return []
    sents = sent_tokenize(text)
    chunks, curr, count = [], [], 0
    for s in sents:
        words = s.split()
        ln = len(words)
        if count + ln > max_words and curr:
            chunks.append(" ".join(curr).strip())
            tail = " ".join(curr).split()[-overlap:]
            curr = tail[:] if tail else []
            count = len(curr)
        curr.append(s)
        count += ln
    if curr:
        chunks.append(" ".join(curr).strip())
    return [c for c in chunks if c]

def normalize_emb(arr: np.ndarray):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms

# -------------------
# FAISS helpers
# -------------------
def create_empty_index():
    idx = faiss.IndexHNSWFlat(EMBED_DIM, 32)  # safe on macOS
    idx.hnsw.efConstruction = 40
    idx.hnsw.efSearch = 32
    return idx

def load_faiss_index():
    if Path(FAISS_INDEX_PATH).exists():
        try:
            idx = faiss.read_index(FAISS_INDEX_PATH)
            return idx
        except Exception:
            print("[!] Failed to read FAISS index file (corrupt?). Recreating.")
    return create_empty_index()

def save_faiss_index(idx):
    faiss.write_index(idx, FAISS_INDEX_PATH)

def load_meta():
    if Path(META_PATH).exists():
        return json.load(open(META_PATH, "r", encoding="utf-8"))
    return []

def save_meta(meta):
    json.dump(meta, open(META_PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

def meta_index_consistent(idx, meta):
    try:
        n_idx = int(idx.ntotal)
    except Exception:
        n_idx = 0
    return n_idx == len(meta)

def rebuild_index_from_meta(meta):
    # Re-create the FAISS index from stored metadata by re-embedding (safer) OR load stored embeddings if you saved them.
    # Here we re-embed text (slower) to guarantee compatibility.
    print("[*] Rebuilding FAISS index from metadata by re-embedding passages (this may take a while)...")
    idx = create_empty_index()
    texts = [m["text"] for m in meta]
    # embed in batches
    emb_batches = []
    for i in range(0, len(texts), BATCH):
        batch = ["passage: " + t for t in texts[i:i+BATCH]]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        emb_batches.append(emb)
    emb_np = np.vstack(emb_batches).astype("float32")
    emb_np = normalize_emb(emb_np)
    idx.add(emb_np)
    save_faiss_index(idx)
    print("[+] Rebuild complete.")
    return idx

# -------------------
# Loaders (book/ref/answers)
# -------------------
def load_book_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    chunks = []
    for chap in data:
        for topic in chap.get("topics", []):
            content = topic.get("content", "")
            # NOTE: content should already be run through normalize_book_text upstream
            for c in topic.get("chunks", []) or chunk_text(content):
                t = c.strip()
                if t:
                    chunks.append(t)
    return chunks

def load_answers_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = []
    for obj in data:
        # prefer 'answer' cleaned, fallback to raw_answer
        ans = obj.get("answer") or obj.get("raw_answer") or ""
        if not isinstance(ans, str):
            # if structure present, flatten
            if isinstance(ans, dict):
                parts = []
                for v in ans.values():
                    if isinstance(v, str):
                        parts.append(v)
                    elif isinstance(v, dict):
                        parts.extend([vv for vv in v.values() if isinstance(vv, str)])
                ans = " ".join(parts)
            else:
                ans = ""
        ans = full_clean(ans)
        if ans:
            chunks.extend(chunk_text(ans))
    return chunks

# -------------------
# Embedding + Upsert
# -------------------
def embed_texts(texts, role="passage"):
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype="float32"), []
    prefix = "query: " if role == "query" else "passage: "
    payload = [prefix + t for t in texts]
    batch_embs = []
    for i in range(0, len(payload), BATCH):
        batch = payload[i:i+BATCH]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        batch_embs.append(emb)
    emb_np = np.vstack(batch_embs).astype("float32")
    emb_np = normalize_emb(emb_np)
    # ids are sha1 of prefixed text
    ids = [sha1(p) for p in payload]
    return emb_np, ids

def add_to_faiss(texts, role, source_file):
    if not texts:
        print("[-] No texts to add.")
        return

    idx = load_faiss_index()
    meta = load_meta()

    emb_np, ids = embed_texts(texts, role=role)

    existing_ids = {m["id"] for m in meta}
    to_add_vecs = []
    to_add_meta = []

    for vec, _id, raw_text in zip(emb_np, ids, texts):
        if _id in existing_ids:
            continue
        to_add_vecs.append(vec)
        to_add_meta.append({"id": _id, "role": role, "file": source_file, "text": raw_text})

    if not to_add_vecs:
        print("[=] Nothing new to add (deduped).")
        return

    arr = np.vstack(to_add_vecs).astype("float32")
    idx.add(arr)
    meta.extend(to_add_meta)
    save_faiss_index(idx)
    save_meta(meta)
    print(f"[+] Added {len(to_add_vecs)} new entries (source={source_file}).")

# -------------------
# Search
# -------------------
def search_faiss(query_text, k=5):
    # ensure index + meta exist and are consistent
    idx = load_faiss_index()
    meta = load_meta()
    if not meta and idx.ntotal == 0:
        print("[-] No data in index.")
        return []

    if not meta_index_consistent(idx, meta):
        print("[!] FAISS index and metadata out of sync. Attempting rebuild...")
        try:
            idx = rebuild_index_from_meta(meta)
        except Exception as e:
            print("[!] Rebuild failed:", e)
            return []

    q_emb, q_ids = embed_texts([query_text], role="query")
    if q_emb.shape[0] == 0:
        return []

    D, I = idx.search(q_emb.astype("float32"), k)
    results = []
    for score, idx_pos in zip(D[0], I[0]):
        if idx_pos < 0 or idx_pos >= len(meta):
            continue
        entry = meta[idx_pos].copy()
        entry["score"] = float(score)  # cosine-like similarity as FAISS returns inner prod on normalized vectors
        results.append(entry)
    return results

# -------------------
# Temp student embeddings (optional)
# -------------------
def save_temp_student_embeddings(texts, emb_np, ids):
    # emb_np is numpy array; convert to lists for JSON (heavy)
    data = []
    for t, id_, vec in zip(texts, ids, emb_np.tolist()):
        data.append({"id": id_, "text": t, "vector": vec})
    json.dump(data, open(TEMP_STUDENT_EMB_PATH, "w", encoding="utf-8"), indent=2)
    print(f"[+] Saved {len(data)} temp student embeddings → {TEMP_STUDENT_EMB_PATH}")

# -------------------
# CLI
# -------------------
def cli():
    print("\nOptions:")
    print("1 = Embed student answers  (regrex_student_answer.json)  [temporary]")
    print("2 = Embed reference answers (regrex_reference_answer.json)")
    print("3 = Embed book chunks     (regrex_book.json)")
    print("4 = Search (interactive)")
    print("5 = Rebuild index from metadata")
    print("6 = Inspect index/meta counts")
    ch = input("choice: ").strip()

    try:
        if ch == "1":
            path = "regrex_student_answer.json"
            if not Path(path).exists():
                print("Missing", path); return
            texts = load_answers_chunks(path)
            # student embeddings are temporary: we do NOT add them to FAISS index
            emb_np, ids = embed_texts(texts, role="query")
            # optional: save temp student embeddings for debugging; disabled by default
            save_temp_student_embeddings(texts, emb_np, ids)
            print(f"[+] Generated {len(ids)} student embeddings (temporary).")
        elif ch == "2":
            path = "regrex_reference_answer.json"
            if not Path(path).exists():
                print("Missing", path); return
            texts = load_answers_chunks(path)
            add_to_faiss(texts, role="passage", source_file=path)
        elif ch == "3":
            path = "regrex_book.json"
            if not Path(path).exists():
                print("Missing", path); return
            # ensure book was normalized upstream
            texts = load_book_chunks(path)
            add_to_faiss(texts, role="passage", source_file=path)
        elif ch == "4":
            q = input("Enter query text: ").strip()
            k = int(input("k (top results): ").strip() or 5)
            res = search_faiss(q, k=k)
            for i, r in enumerate(res, 1):
                print(f"\n[{i}] score={r['score']:.4f} file={r.get('file')} id={r['id']}")
                print(r['text'][:400])
        elif ch == "5":
            meta = load_meta()
            if not meta:
                print("No metadata to rebuild from.")
                return
            rebuild_index_from_meta(meta)
        elif ch == "6":
            idx = load_faiss_index()
            meta = load_meta()
            print("FAISS ntotal:", int(idx.ntotal))
            print("Metadata length:", len(meta))
            print("Consistent:", meta_index_consistent(idx, meta))
        else:
            print("invalid")
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    cli()
