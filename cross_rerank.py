#!/usr/bin/env python3
"""
cross_rerank.py
Automatically reranks NLI-filtered results using a cross-encoder.
Input  : nli_output.json
Output : cross_rerank_output.json
"""

import json
from pathlib import Path
from typing import List, Dict

from sentence_transformers import CrossEncoder
import torch


# -----------------------------
# DEVICE CONFIG
# -----------------------------
DEVICE = "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"


# -----------------------------
# Load CrossEncoder model
# -----------------------------
MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
print(f"[+] Loading CrossEncoder model on {DEVICE}...")
model = CrossEncoder(MODEL, device=DEVICE)


# -----------------------------
# RERANK FUNCTION
# -----------------------------
def rerank(student_answer: str, candidates: List[Dict]):
    """
    candidate entry:
    {
        "text": "...",
        "score": float,
        "nli": {
            "entailment": float
        }
    }
    """

    if not candidates:
        return []

    # CE input
    pairs = [(student_answer, c["text"]) for c in candidates]

    # model predict
    ce_scores = model.predict(pairs).tolist()

    # attach scores
    for c, s in zip(candidates, ce_scores):
        c["ce_score"] = float(s)

    # final sorting
    candidates = sorted(
        candidates,
        key=lambda x: (x["ce_score"], x["nli"]["entailment"]),
        reverse=True
    )

    return candidates


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def run_cross_encoder():
    NLI_FILE = "nli_output.json"

    if not Path(NLI_FILE).exists():
        print("❌ nli_output.json not found. Run nli_filter.py first.")
        return

    print(f"[+] Loading NLI results from {NLI_FILE}...")
    nli_data = json.load(open(NLI_FILE, "r"))

    final_output = []

    print("\n[+] Running CrossEncoder reranking for all student answers...")

    for item in nli_data:
        qnum = item["question_number"]
        student_ans = item["student_answer"]
        candidates = item["faiss_nli_ranked"]

        print(f"\n======================================================")
        print(f"Reranking Q{qnum} ...")
        print("======================================================")

        ranked = rerank(student_ans, candidates)

        if ranked:
            best = ranked[0]
            print(f"[BEST MATCH] CE Score={best['ce_score']:.3f} | Entail={best['nli']['entailment']:.3f}")
            print(best["text"][:300], "...")
        else:
            print("No candidates after reranking.")

        final_output.append({
            "question_number": qnum,
            "student_answer": student_ans,
            "reranked_results": ranked,
        })

    # save
    json.dump(final_output, open("cross_rerank_output.json", "w"), indent=4)

    print("\n[✔] Cross-encoder results saved to cross_rerank_output.json")


# -----------------------------
# ENTRYPOINT
# -----------------------------
if __name__ == "__main__":
    run_cross_encoder()
