#!/usr/bin/env python3
"""
marks_generator.py
- Reads cross_rerank_output.json
- For each student chunk, computes a semantic score from FAISS similarity / NLI entailment / CE score
- Computes keyword score (30 points) using reference if available, otherwise best passage
- Aggregates per-chunk scores into per-question final marks (0-100)
- Outputs:
    - full_pipeline_output.json (detailed per-chunk candidates + scores)
    - final_scores.json (summary per question)
"""

import json
import math
import re
from pathlib import Path
from typing import List, Dict

CE_OUTPUT_PATH = Path("cross_rerank_output.json")
FULL_OUTPUT_PATH = Path("full_pipeline_output.json")
FINAL_SCORES_PATH = Path("final_scores.json")

# We follow your requested weight: 70% semantic, 30% keywords
SEMANTIC_MAX = 70.0
KEYWORD_MAX = 30.0

# semantic composition weights (internal): adjust if needed
# these sum to 1.0 for the semantic component
W_FAISS = 0.4
W_CE = 0.3
W_NLI = 0.3

# helper transforms
def logistic(x: float) -> float:
    """map real number to (0,1) via logistic (sigmoid)"""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def faiss_score_to_01(s: float) -> float:
    """
    FAISS with normalized vectors usually returns inner-prod in [-1,1] or [0,1].
    Map it into [0,1] defensively.
    """
    if s is None:
        return 0.0
    # clamp
    s_clamped = max(min(s, 1.0), -1.0)
    return (s_clamped + 1.0) / 2.0

def extract_keywords(text: str) -> List[str]:
    text = (text or "").lower()
    # naive noun-ish words (3+ letters)
    words = re.findall(r"[a-zA-Z]{3,}", text)
    stop = {
        "the", "and", "are", "this", "that", "from", "into", "with",
        "your", "they", "have", "has", "was", "for", "but", "can",
        "will", "their", "them", "which", "what"
    }
    kws = [w for w in words if w not in stop]
    return list(dict.fromkeys(kws))  # preserve order, unique

def compute_keyword_score(student_text: str, reference_text: str) -> float:
    """
    returns value 0..KEYWORD_MAX
    """
    stud_k = set(extract_keywords(student_text))
    ref_k = set(extract_keywords(reference_text))
    if not ref_k:
        return 0.0
    matched = len(stud_k.intersection(ref_k))
    ratio = matched / max(len(ref_k), 1)
    return round(ratio * KEYWORD_MAX, 2)

def compute_semantic_score(candidate: Dict) -> float:
    """
    candidate is expected to have fields:
      - score: FAISS similarity (float)  -- may be inner product
      - ce_score: cross-encoder score (float) -- real-valued
      - nli: { 'entailment': float }
    Returns semantic score in 0..SEMANTIC_MAX
    """
    faiss_raw = candidate.get("score", 0.0)
    faiss_01 = faiss_score_to_01(faiss_raw)

    ce_raw = candidate.get("ce_score", 0.0)
    ce_01 = logistic(ce_raw)  # map CE to 0..1

    ent = candidate.get("nli", {}).get("entailment", 0.0)
    # combine
    semantic_comp = (W_FAISS * faiss_01) + (W_CE * ce_01) + (W_NLI * ent)
    semantic_score = round(semantic_comp * SEMANTIC_MAX, 2)
    return semantic_score

# Aggregation across chunks: simple average of chunk scores (weighted by chunk length optionally)
def aggregate_question_scores(chunk_scores: List[Dict]) -> Dict:
    """
    chunk_scores: list of dicts [{ 'chunk_index':..., 'semantic_score':..., 'keyword_score':..., 'final_chunk_score':... }]
    returns aggregate dict with final numeric score
    """
    if not chunk_scores:
        return {"semantic_total": 0.0, "keyword_total": 0.0, "final_score": 0.0}

    # simple average across chunks
    sems = [c["semantic_score"] for c in chunk_scores]
    kws = [c["keyword_score"] for c in chunk_scores]
    sem_avg = sum(sems) / len(sems)
    kw_avg = sum(kws) / len(kws)
    final = round(sem_avg + kw_avg, 2)  # sem already scaled to SEMANTIC_MAX, kw to KEYWORD_MAX
    return {"semantic_total": round(sem_avg,2), "keyword_total": round(kw_avg,2), "final_score": final}

# Main
def generate_marks():
    if not CE_OUTPUT_PATH.exists():
        raise FileNotFoundError("cross_rerank_output.json not found. Run cross_rerank.py first.")

    data = json.load(open(CE_OUTPUT_PATH, "r", encoding="utf-8"))
    full_out = []
    summary = []

    for item in data:
        qnum = item.get("question_number")
        student_chunks = item.get("student_chunks", [])

        per_chunk_results = []

        for ch in student_chunks:
            chunk_text = ch.get("chunk_text", "")
            reranked = ch.get("reranked", []) or []

            if not reranked:
                per_chunk_results.append({
                    "chunk_index": ch.get("chunk_index"),
                    "chunk_text": chunk_text,
                    "semantic_score": 0.0,
                    "keyword_score": 0.0,
                    "final_chunk_score": 0.0,
                    "best_candidate": None
                })
                continue

            # best candidate is the top reranked one
            best = reranked[0]

            # compute semantic score (0..70)
            semantic_score = compute_semantic_score(best)

            # determine reference for keyword scoring:
            # if any candidate from reference exists at top, prefer its text as keyword source
            ref_candidate = None
            for c in reranked:
                if c.get("source_type") == "reference":
                    ref_candidate = c
                    break
            ref_text_for_kw = (ref_candidate or best).get("cleaned_text", best.get("text",""))

            keyword_score = compute_keyword_score(chunk_text, ref_text_for_kw)

            final_chunk = round(semantic_score + keyword_score, 2)

            per_chunk_results.append({
                "chunk_index": ch.get("chunk_index"),
                "chunk_text": chunk_text,
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "final_chunk_score": final_chunk,
                "best_candidate": {
                    "id": best.get("id"),
                    "source_type": best.get("source_type"),
                    "file": best.get("file"),
                    "chunk_index": best.get("chunk_index"),
                    "text": best.get("cleaned_text", best.get("text","")),
                    "faiss_score": best.get("score"),
                    "ce_score": best.get("ce_score"),
                    "nli": best.get("nli")
                }
            })

        # aggregate per question
        agg = aggregate_question_scores(per_chunk_results)

        full_out.append({
            "question_number": qnum,
            "chunks": per_chunk_results,
            "aggregate": agg
        })

        summary.append({
            "question_number": qnum,
            "semantic_total": agg["semantic_total"],
            "keyword_total": agg["keyword_total"],
            "final_score": agg["final_score"]
        })

    # save both outputs
    json.dump(full_out, open(FULL_OUTPUT_PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(summary, open(FINAL_SCORES_PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"[+] Full pipeline output -> {FULL_OUTPUT_PATH}")
    print(f"[+] Final scores -> {FINAL_SCORES_PATH}")
    return full_out, summary

if __name__ == "__main__":
    generate_marks()
