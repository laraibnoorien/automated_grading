#!/usr/bin/env python3
"""
nli_filter.py
Automatically runs NLI on:
- Every student answer (from regrex_student_answer.json)
- Their FAISS top matches (using local_faiss_search.py search logic)
"""

import json
from typing import List, Dict
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

# -----------------------------
# Paths (same as FAISS system)
# -----------------------------
FAISS_INDEX_PATH = "./faiss_index.index"
META_PATH = "./faiss_metadata.json"
STUDENT_REGREX_PATH = "./regrex_student_answer.json"
MODEL_CACHE = "./model_cache"
EMBED_DIM = 768


# -----------------------------
# Device
# -----------------------------
DEVICE = "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"


# -----------------------------
# Load NLI model
# -----------------------------
print(f"[+] Loading NLI model ({DEVICE})...")
NLI_MODEL = "facebook/bart-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(DEVICE)
model.eval()


# -----------------------------
# Load embedding model (same as FAISS embedder)
# -----------------------------
def load_embedder():
    if Path(MODEL_CACHE).exists() and any(Path(MODEL_CACHE).iterdir()):
        m = SentenceTransformer(MODEL_CACHE)
    else:
        m = SentenceTransformer("intfloat/e5-base-v2")
    return m

embedder = load_embedder()


# -----------------------------
# Normalize vectors
# -----------------------------
def normalize(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1
    return v / n


# -----------------------------
# Load FAISS + metadata safely
# -----------------------------
def load_faiss():
    if not Path(FAISS_INDEX_PATH).exists():
        raise FileNotFoundError("FAISS index missing. Run local_embed_faiss.py first.")
    return faiss.read_index(FAISS_INDEX_PATH)

def load_meta():
    if not Path(META_PATH).exists():
        raise FileNotFoundError("faiss_metadata.json missing.")
    return json.load(open(META_PATH))


# -----------------------------
# FAISS search for a single student answer
# -----------------------------
def faiss_search_one(text: str, k=5):
    index = load_faiss()
    meta = load_meta()

    if index.ntotal == 0:
        return []

    q = "query: " + text.strip()
    emb = embedder.encode([q], convert_to_numpy=True)
    emb = normalize(emb.astype("float32"))

    distances, idxs = index.search(emb, min(k, index.ntotal))

    out = []
    for i, score in zip(idxs[0], distances[0]):
        if i >= len(meta):
            continue
        m = meta[i].copy()
        m["score"] = float(score)
        out.append(m)

    return out


# -----------------------------
# NLI scoring
# -----------------------------
def nli_score(student_answer: str, passage: str):
    """Return entailment / neutral / contradiction prob."""
    
    inputs = tokenizer(
        student_answer,
        passage,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    labels = ["contradiction", "neutral", "entailment"]

    return {
        "result": labels[np.argmax(probs)],
        "contradiction": float(probs[0]),
        "neutral": float(probs[1]),
        "entailment": float(probs[2]),
    }


# -----------------------------
# Run NLI on FAISS results
# -----------------------------
def filter_with_nli(student_text: str, faiss_results: List[Dict]):
    out = []

    for r in faiss_results:
        nli = nli_score(student_text, r["text"])
        r2 = r.copy()
        r2["nli"] = nli
        out.append(r2)

    # Sort: highest entailment first
    out = sorted(out, key=lambda x: x["nli"]["entailment"], reverse=True)
    return out


# -----------------------------
# MAIN: automatically run for all student answers
# -----------------------------
def run_nli_for_all(top_k=5):
    if not Path(STUDENT_REGREX_PATH).exists():
        print("❌ regrex_student_answer.json not found.")
        return

    students = json.load(open(STUDENT_REGREX_PATH))

    final_output = []

    for ans in students:
        qnum = ans.get("question_number")
        text = ans.get("answer") or ans.get("raw_answer") or ""
        text = text.strip()

        print("\n" + "="*70)
        print(f"QUESTION {qnum} — Running FAISS + NLI")
        print("="*70)

        faiss_hits = faiss_search_one(text, k=top_k)

        if not faiss_hits:
            print("No FAISS matches found.\n")
            continue

        ranked = filter_with_nli(text, faiss_hits)

        final_output.append({
            "question_number": qnum,
            "student_answer": text,
            "faiss_nli_ranked": ranked,
        })

        # Print top result
        best = ranked[0]
        print(f"\nBest match (entailment={best['nli']['entailment']:.3f}):")
        print(best["text"][:300], "...")

    # Save final NLI output
    json.dump(final_output, open("nli_output.json", "w"), indent=4)

    print("\n\n[✔] NLI results saved to nli_output.json")


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    run_nli_for_all(top_k=5)
