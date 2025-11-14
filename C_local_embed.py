# local_embed_faiss.py
import os
import json
import faiss
import numpy as np
import hashlib
from pathlib import Path
from sentence_transformers import SentenceTransformer
from datetime import datetime
import traceback

from B_normalize_answer import normalize_text, NORMALIZE_VERSION
from B_Regrex_Answers import sentence_group_chunk

# -------------------
# Config
# -------------------
MODEL_NAME = "intfloat/e5-base-v2"   # retrieval model (fast; good for FAISS)
EMBED_DIM = 768
BATCH = 16

# index & meta files (separate for book and reference)
DATA_DIR = Path("./faiss_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

FAISS_BOOK_PATH = DATA_DIR / "faiss_book.index"
META_BOOK_PATH = DATA_DIR / "faiss_book_meta.json"

FAISS_REF_PATH = DATA_DIR / "faiss_ref.index"
META_REF_PATH = DATA_DIR / "faiss_ref_meta.json"

# ensure reproducible threading
os.environ["TOKENIZERS_PARALLELISM"] = "false"
faiss.omp_set_num_threads(1)

# -------------------
# Model loader
# -------------------
MODEL_CACHE = "./model_cache"
Path(MODEL_CACHE).mkdir(exist_ok=True)

def load_model():
    try:
        model = SentenceTransformer(MODEL_CACHE) if Path(MODEL_CACHE).exists() and any(Path(MODEL_CACHE).iterdir()) else SentenceTransformer(MODEL_NAME)
    except Exception:
        model = SentenceTransformer(MODEL_NAME)
    # try to move to GPU/MPS if available
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

def normalize_emb(arr: np.ndarray):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms

# -------------------
# Index helpers (two separate indices)
# -------------------
def create_hnsw_index():
    idx = faiss.IndexHNSWFlat(EMBED_DIM, 32)
    idx.hnsw.efConstruction = 40
    idx.hnsw.efSearch = 32
    return idx

def load_index(path: Path):
    if path.exists():
        try:
            idx = faiss.read_index(str(path))
            return idx
        except Exception:
            print(f"[!] Failed to read FAISS index at {path}. Recreating.")
    return create_hnsw_index()

def save_index(idx, path: Path):
    faiss.write_index(idx, str(path))

def load_meta(path: Path):
    if path.exists():
        return json.load(open(path, "r", encoding="utf-8"))
    return []

def save_meta(meta, path: Path):
    json.dump(meta, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

def meta_consistent(idx, meta):
    try:
        n_idx = int(idx.ntotal)
    except Exception:
        n_idx = 0
    return n_idx == len(meta)

# -------------------
# Embedding
# -------------------
def embed_texts(texts, role="passage"):
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype="float32"), []
    prefix = "query: " if role == "query" else "passage: "
    payload = [prefix + t for t in texts]
    embs = []
    for i in range(0, len(payload), BATCH):
        batch = payload[i:i+BATCH]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embs.append(emb)
    emb_np = np.vstack(embs).astype("float32")
    emb_np = normalize_emb(emb_np)
    ids = [sha1(p) for p in payload]
    return emb_np, ids

# -------------------
# Add to index (book or reference)
# -------------------
def add_texts_to_index(texts, source_file: str, source_type: str = "book"):
    """
    source_type: 'book' or 'reference'
    texts: list of raw passages (strings); they should be pre-normalized for the type upstream ideally
    """
    if not texts:
        print("[-] No texts to add.")
        return

    if source_type == "reference":
        idx = load_index(FAISS_REF_PATH)
        meta_path = META_REF_PATH
        meta = load_meta(meta_path)
    else:
        idx = load_index(FAISS_BOOK_PATH)
        meta_path = META_BOOK_PATH
        meta = load_meta(meta_path)

    emb_np, ids = embed_texts(texts, role="passage")
    existing_ids = {m["id"] for m in meta}

    to_add_vecs = []
    to_add_meta = []
    now = datetime.utcnow().isoformat() + "Z"

    for chunk_index, (vec, _id, raw_text) in enumerate(zip(emb_np, ids, texts)):
        # dedupe by id + normalize_version + embedding_model
        # if same id exists for same model and normalize version skip
        already = False
        for m in meta:
            if m.get("id") == _id and m.get("embedding_model") == MODEL_NAME and m.get("normalize_version") == NORMALIZE_VERSION:
                already = True
                break
        if already:
            continue

        to_add_vecs.append(vec)
        to_add_meta.append({
            "id": _id,
            "role": "passage",
            "source_type": source_type,
            "file": source_file,
            "chunk_index": chunk_index,
            "text": raw_text,
            "cleaned_text": normalize_text(raw_text, mode=("book" if source_type=="book" else "answer"))["normalized"],
            "embedding_model": MODEL_NAME,
            "normalize_version": NORMALIZE_VERSION,
            "added_ts": now
        })

    if not to_add_vecs:
        print("[=] Nothing new to add (deduped).")
        return

    arr = np.vstack(to_add_vecs).astype("float32")
    idx.add(arr)
    meta.extend(to_add_meta)
    # save index + meta
    save_index(idx, FAISS_REF_PATH if source_type == "reference" else FAISS_BOOK_PATH)
    save_meta(meta, meta_path)
    print(f"[+] Added {len(to_add_meta)} entries to {source_type} index (source={source_file}).")

# -------------------
# Search helpers
# -------------------
def _search_index(idx_path: Path, meta_path: Path, query_text: str, k=5):
    if not idx_path.exists() or not meta_path.exists():
        return []
    idx = load_index(idx_path)
    meta = load_meta(meta_path)
    if not meta and idx.ntotal == 0:
        return []

    if not meta_consistent(idx, meta):
        print("[!] Index and metadata out-of-sync for", idx_path.name, "- attempting rebuild (re-embed).")
        # best-effort rebuild
        try:
            rebuild_index(idx_path, meta_path)
            idx = load_index(idx_path)
            meta = load_meta(meta_path)
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
        entry["score"] = float(score)
        results.append(entry)
    return results

def search_book(query_text: str, k=5):
    return _search_index(FAISS_BOOK_PATH, META_BOOK_PATH, query_text, k=k)

def search_reference(query_text: str, k=5):
    return _search_index(FAISS_REF_PATH, META_REF_PATH, query_text, k=k)

def search_combined(query_text: str, k_book=3, k_ref=5):
    """
    Search both indices and return merged list ordered by score with source marker.
    """
    ref_res = search_reference(query_text, k=k_ref) or []
    book_res = search_book(query_text, k=k_book) or []
    # simple merge by score
    merged = ref_res + book_res
    merged.sort(key=lambda x: x.get("score", 0), reverse=True)
    return merged

# -------------------
# Rebuild index from meta (re-embed cleaned_text)
# -------------------
def rebuild_index(idx_path: Path, meta_path: Path):
    meta = load_meta(meta_path)
    if not meta:
        print("[!] No metadata to rebuild from.")
        return
    print("[*] Rebuilding index from metadata by re-embedding cleaned_text (this may take a while)...")
    idx = create_hnsw_index()
    texts = [m.get("cleaned_text", m.get("text", "")) for m in meta]
    # batch re-embed
    emb_batches = []
    for i in range(0, len(texts), BATCH):
        batch = ["passage: " + t for t in texts[i:i+BATCH]]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        emb_batches.append(emb)
    emb_np = np.vstack(emb_batches).astype("float32")
    emb_np = normalize_emb(emb_np)
    idx.add(emb_np)
    save_index(idx, idx_path)
    print("[+] Rebuild complete for", idx_path.name)

# -------------------
# Quick CLI for testing
# -------------------
def cli():
    print("\nOptions:")
    print("1 = Add reference answers (regrex_reference_answer.json)")
    print("2 = Add book chunks (regrex_book.json)")
    print("3 = Search combined")
    print("4 = Rebuild both indices")
    print("5 = Inspect counts")
    ch = input("choice: ").strip()
    try:
        if ch == "1":
            path = "regrex_reference_answer.json"
            if not Path(path).exists():
                print("Missing", path); return
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # data already contains 'chunks' for answers in our new regrex_output
            texts = []
            for obj in data:
                # push each chunk (answers may have 'chunks')
                if isinstance(obj.get("chunks"), list) and obj["chunks"]:
                    texts.extend(obj["chunks"])
                else:
                    texts.append(obj.get("answer") or obj.get("raw_answer") or "")
            add_texts_to_index(texts, source_file=path, source_type="reference")
        elif ch == "2":
            path = "regrex_book.json"
            if not Path(path).exists():
                print("Missing", path); return
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            texts = []
            # support both structure you had (chapters->topics) or simple list
            for chap in (data if isinstance(data, list) else [data]):
                for topic in chap.get("topics", []):
                    # topic has 'chunks' if produced by parse_book_ocr_text
                    if topic.get("chunks"):
                        texts.extend(topic["chunks"])
                    else:
                        # fallback chunk by sentence groups
                        texts.extend(sentence_group_chunk(topic.get("content",""), group_size=3, overlap=1))
            add_texts_to_index(texts, source_file=path, source_type="book")
        elif ch == "3":
            q = input("Enter query text: ").strip()
            res = search_combined(q, k_book=3, k_ref=5)
            for i, r in enumerate(res, 1):
                print(f"\n[{i}] score={r['score']:.4f} source={r['source_type']} file={r.get('file')} id={r['id']}")
                print(r['cleaned_text'][:400])
        elif ch == "4":
            if META_BOOK_PATH.exists():
                rebuild_index(FAISS_BOOK_PATH, META_BOOK_PATH)
            if META_REF_PATH.exists():
                rebuild_index(FAISS_REF_PATH, META_REF_PATH)
        elif ch == "5":
            idx_b = load_index(FAISS_BOOK_PATH); meta_b = load_meta(META_BOOK_PATH)
            idx_r = load_index(FAISS_REF_PATH); meta_r = load_meta(META_REF_PATH)
            print("Book ntotal:", int(idx_b.ntotal), "meta:", len(meta_b), "consistent:", meta_consistent(idx_b, meta_b))
            print("Ref ntotal:", int(idx_r.ntotal), "meta:", len(meta_r), "consistent:", meta_consistent(idx_r, meta_r))
        else:
            print("invalid")
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    cli()
