#!/usr/bin/env python3
"""
marks_generator.py
Evaluates answers out of 100% using:
- Semantic similarity (NLI + CE)
- Keyword matching
"""

import json
import re
from pathlib import Path

INPUT_FILE = "cross_rerank_output.json"
OUTPUT_FILE = "final_marks.json"


# -----------------------------
# Extract keywords (scientific terms, nouns)
# -----------------------------
def extract_keywords(text: str):
    text = text.lower()
    words = re.findall(r"[a-zA-Z]{3,}", text)  # drop 1–2 letter words
    ignore = {
        "the", "and", "are", "this", "that",
        "from", "into", "with", "your", "they",
        "have", "has", "was", "for", "but",
        "can", "will", "their", "them"
    }
    return [w for w in words if w not in ignore]


# -----------------------------
# Main Scoring Logic
# -----------------------------
def score_answer(student: str, best_passage: dict):
    """
    Calculates:
    - semantic_score (0–70)
    - keyword_score  (0–30)
    """

    # ============ SEMANTIC SCORE ============
    ce = best_passage.get("ce_score", 0)
    ent = best_passage.get("nli", {}).get("entailment", 0)

    # normalize CE score (MiniLM usually between -5 to +5)
    ce_norm = max(min((ce + 5) / 10, 1), 0)

    semantic_score = (ce_norm * 40) + (ent * 30)
    semantic_score = max(0, min(70, semantic_score))

    # ============ KEYWORD SCORE ============
    stud_kw = set(extract_keywords(student))
    pass_kw = set(extract_keywords(best_passage.get("text", "")))

    if len(pass_kw) == 0:
        keyword_score = 0
    else:
        matched = len(stud_kw.intersection(pass_kw))
        keyword_score = (matched / len(pass_kw)) * 30

    keyword_score = max(0, min(30, keyword_score))

    # ============ FINAL SCORE ============
    final_score = semantic_score + keyword_score
    final_score = round(final_score, 2)

    return {
        "semantic_score": round(semantic_score, 2),
        "keyword_score": round(keyword_score, 2),
        "final_score": final_score
    }


# -----------------------------
# PROCESS ALL ANSWERS
# -----------------------------
def generate_marks():
    if not Path(INPUT_FILE).exists():
        print("❌ cross_rerank_output.json not found. Run cross_rerank.py first.")
        return

    data = json.load(open(INPUT_FILE, "r"))

    final_output = []

    for item in data:
        q = item["question_number"]
        student_answer = item["student_answer"]
        ranked = item["reranked_results"]

        if not ranked:
            # no matches → give minimal score
            final_output.append({
                "question_number": q,
                "student_answer": student_answer,
                "semantic_score": 0,
                "keyword_score": 0,
                "final_score": 0
            })
            continue

        best = ranked[0]  # top-ranked chunk
        marks = score_answer(student_answer, best)

        final_output.append({
            "question_number": q,
            "student_answer": student_answer,
            "best_passage_text": best["text"],
            **marks
        })

    json.dump(final_output, open(OUTPUT_FILE, "w"), indent=4)
    print(f"\n[✔] Final marks saved to {OUTPUT_FILE}")

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    generate_marks()
