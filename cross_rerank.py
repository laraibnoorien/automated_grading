#!/usr/bin/env python3
"""
cross_rerank.py (batched cross-encoder reranking)
- Reads nli_output.json
- For each student chunk, takes top-N candidates by entailment and runs CrossEncoder in batch
- Produces cross_rerank_output.json with CE scores attached
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple
import math
import time

import torch
from sentence_transformers import CrossEncoder

NLI_OUTPUT_PATH = Path("nli_output.json")
CE_OUTPUT_PATH = Path("cross_rerank_output.json")

CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_BATCH = 64

DEVICE = "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"

print(f"[+] Loading CrossEncoder ({CE_MODEL_NAME}) on {DEVICE} ...")
ce_model = CrossEncoder(CE_MODEL_NAME, device=DEVICE)

# how many top candidates (after entailment) to run CE on per chunk
TOP_FOR_CE = 5

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def run_rerank():
    if not NLI_OUTPUT_PATH.exists():
        raise FileNotFoundError("nli_output.json not found. Run nli_filter.py first.")

    nli_data = json.load(open(NLI_OUTPUT_PATH, "r", encoding="utf-8"))
    final_output = []

    for item in nli_data:
        qnum = item.get("question_number")
        student_chunks = item.get("student_chunks", [])

        out_chunks = []
        for ch in student_chunks:
            chunk_text = ch.get("chunk_text")
            candidates = ch.get("candidates", []) or []

            if not candidates:
                out_chunks.append({
                    "chunk_index": ch.get("chunk_index"),
                    "chunk_text": chunk_text,
                    "reranked": []
                })
                continue

            # take top-K by entailment (fast filter)
            topk = candidates[:TOP_FOR_CE]

            # prepare pairs for CE
            pairs = [(chunk_text, c.get("cleaned_text", c.get("text",""))) for c in topk]
            # run CE in batch (CrossEncoder handles batching internally, but to control memory we can chunk)
            ce_scores = ce_model.predict(pairs, batch_size=CE_BATCH).tolist()

            # attach CE scores
            for c, s in zip(topk, ce_scores):
                c["ce_score"] = float(s)

            # sort by CE score (desc), fallback to entailment
            topk_sorted = sorted(topk, key=lambda x: (x.get("ce_score", 0), x.get("nli", {}).get("entailment", 0)), reverse=True)

            out_chunks.append({
                "chunk_index": ch.get("chunk_index"),
                "chunk_text": chunk_text,
                "reranked": topk_sorted
            })

        final_output.append({
            "question_number": qnum,
            "student_chunks": out_chunks
        })

    json.dump(final_output, open(CE_OUTPUT_PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[+] Cross-encoder rerank output saved to {CE_OUTPUT_PATH}")
    return final_output

if __name__ == "__main__":
    run_rerank()
