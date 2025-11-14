#!/usr/bin/env python3
"""
nli_filter.py (batch NLI + retrieval per-chunk)
- Processes regrex_student_answer.json (student answers must contain "chunks")
- Uses local_embed_faiss.search_combined() to retrieve candidates (book+reference)
- Runs NLI in batches for all chunk-candidate pairs
- Produces 'nli_output.json' (detailed per-chunk candidates with NLI scores)
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple
import math
import time

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Local modules (from the new FAISS system you already added)
from C_local_embed import search_combined
from B_normalize_answer import normalize_text

# -------------------------
# Config
# -------------------------
STUDENT_REGREX_PATH = Path("regrex_student_answer.json")
NLI_OUTPUT_PATH = Path("nli_output.json")

NLI_MODEL = "facebook/bart-large-mnli"
BATCH_SIZE = 32  # batch size for NLI
DEVICE = "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"

# retrieval sizes (user choice)
BOOK_K = 10
REF_K = 5

# -------------------------
# Load model
# -------------------------
print(f"[+] Loading NLI model ({NLI_MODEL}) on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(DEVICE)
nli_model.eval()

# -------------------------
# Helpers
# -------------------------
LABELS = ["contradiction", "neutral", "entailment"]

def batched_nli(premises: List[str], hypotheses: List[str], batch_size: int = 32) -> List[Dict]:
    """
    premises: list of student_chunk strings (hypothesis position in M-NLI will be student? We'll follow: premise=passage, hypothesis=student)
    hypotheses: list of passage strings
    returns list of dicts: {"contradiction":..., "neutral":..., "entailment":..., "result": label}
    NOTE: We use premise=passage, hypothesis=student to match prior convention? Our score uses entailment probability (passage entails student).
    """
    results = []
    assert len(premises) == len(hypotheses)
    for i in range(0, len(premises), batch_size):
        batch_p = premises[i:i+batch_size]
        batch_h = hypotheses[i:i+batch_size]
        # tokenizer accepts two lists for pair encoding
        enc = tokenizer(batch_p, batch_h, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            logits = nli_model(**enc).logits  # (batch, 3)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for p in probs:
            label_idx = int(np.argmax(p))
            out = {
                "contradiction": float(p[0]),
                "neutral": float(p[1]),
                "entailment": float(p[2]),
                "result": LABELS[label_idx]
            }
            results.append(out)
    return results

# -------------------------
# Main runner
# -------------------------
def run_nli_for_all(book_k:int=BOOK_K, ref_k:int=REF_K):
    if not STUDENT_REGREX_PATH.exists():
        raise FileNotFoundError("regrex_student_answer.json not found. Run regrex_answer.py first.")

    students = json.load(open(STUDENT_REGREX_PATH, "r", encoding="utf-8"))
    full_output = []

    # We'll batch NLI over all chunk-candidate pairs per student to be efficient.
    for ans in students:
        qnum = ans.get("question_number")
        # prefer 'chunks' if present; else build one chunk from 'answer'
        chunks = ans.get("chunks")
        if not chunks:
            raw = ans.get("answer") or ans.get("raw_answer") or ""
            raw_norm = normalize_text(raw, mode="answer")["normalized"]
            chunks = [raw_norm] if raw_norm.strip() else []

        student_entry = {
            "question_number": qnum,
            "student_chunks": [],
            "chunk_count": len(chunks)
        }

        print(f"[Q{qnum}] processing {len(chunks)} chunks...")

        for ci, chunk in enumerate(chunks):
            chunk_text = chunk.strip()
            if not chunk_text:
                continue

            # retrieve combined results (reference + book)
            candidates = search_combined(chunk_text, k_book=book_k, k_ref=ref_k)
            # dedupe candidates by id while keeping order
            seen = set()
            deduped = []
            for c in candidates:
                cid = c.get("id")
                if cid in seen:
                    continue
                seen.add(cid)
                deduped.append(c)

            # prepare NLI batch pairs: premise=passage_text, hypothesis=student_chunk
            premises = [c.get("cleaned_text", c.get("text","")) for c in deduped]
            hypotheses = [chunk_text for _ in premises]

            if not premises:
                student_entry["student_chunks"].append({
                    "chunk_index": ci,
                    "chunk_text": chunk_text,
                    "candidates": [],
                    "note": "no candidates"
                })
                continue

            # run NLI in batches
            nli_results = batched_nli(premises, hypotheses, batch_size=BATCH_SIZE)

            # attach NLI scores to candidates
            candidates_with_nli = []
            for c, nli in zip(deduped, nli_results):
                c2 = c.copy()
                c2["nli"] = nli
                candidates_with_nli.append(c2)

            # sort by entailment prob desc
            candidates_with_nli.sort(key=lambda x: x["nli"]["entailment"], reverse=True)

            student_entry["student_chunks"].append({
                "chunk_index": ci,
                "chunk_text": chunk_text,
                "candidates": candidates_with_nli
            })

        full_output.append(student_entry)

    # save
    json.dump(full_output, open(NLI_OUTPUT_PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[+] NLI output saved to {NLI_OUTPUT_PATH}")
    return full_output


if __name__ == "__main__":
    run_nli_for_all()
