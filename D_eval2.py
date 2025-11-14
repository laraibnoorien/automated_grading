"""
evaluation_v2.py — FIXED WITH AUTO BOOK DETECTION
"""

import os
import time
import json
from collections import defaultdict
from typing import List, Tuple, Dict

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import sent_tokenize

from B_normalize_answer import full_clean

# CONFIG
EMBEDDING_DB_PATH = "large_embeddings_library.pt"
STUDENT_JSON_PATH = "regrex_student_answer.json"

SEMANTIC_MODEL = "intfloat/e5-large-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-electra-base"
NLI_MODEL = "roberta-large-mnli"

TOP_K = 20
CROSS_BATCH = 32
NLI_BATCH = 16
MAX_CHUNK_WORDS = 80
SCORE_CLIP = (-10.0, 50.0)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

# ---------------------------
# LOAD MODELS
# ---------------------------
def load_models():
    try:
        sem = SentenceTransformer(SEMANTIC_MODEL, device=device)
    except:
        sem = SentenceTransformer("intfloat/e5-base-v2", device=device)

    cross = CrossEncoder(CROSS_ENCODER_MODEL, device=device)
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(device)

    return sem, cross, nli_tokenizer, nli_model


# ---------------------------
# LOAD STUDENT ANSWERS
# ---------------------------
def load_student_structured():
    with open(STUDENT_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------
# CHUNK STUDENT ANSWERS
# ---------------------------
def chunk_student_answers_for_search(structured):
    chunks = []
    map_idx = []

    for i, ans in enumerate(structured):
        parts = ans.get("answer_parts", ans.get("answer", ""))


        if isinstance(parts, dict):

            def flatten(v):
                if isinstance(v, str): return v
                if isinstance(v, dict): return " ".join(flatten(x) for x in v.values())
                return ""

            text = flatten(parts).strip()
        else:
            text = str(parts).strip()

        text = full_clean(text)
        if not text:
            continue

        words = text.split()

        if len(words) <= MAX_CHUNK_WORDS:
            chunks.append(text)
            map_idx.append(i)
            continue

        sents = sent_tokenize(text)
        cur, cur_w = [], 0

        for s in sents:
            w = len(s.split())

            if cur_w + w > MAX_CHUNK_WORDS and cur:
                chunks.append(" ".join(cur))
                map_idx.append(i)

                cur = [s]
                cur_w = w
            else:
                cur.append(s)
                cur_w += w

        if cur:
            chunks.append(" ".join(cur))
            map_idx.append(i)

    return chunks, map_idx


# ---------------------------
# AUTO-DETECT CATEGORY
# ---------------------------
def load_library_embeddings(school_type, subject, board, class_name, category=None):
    db = torch.load(EMBEDDING_DB_PATH)

    if school_type == "school":
        node = db.get(school_type, {}).get(subject, {}).get(board, {}).get(class_name, {})
    else:
        node = db.get(school_type, {}).get(subject, {})

    priority = ["reference_answer", "reference", "book", "library"]

    if category:
        priority = [category] + [x for x in priority if x != category]

    valid_key = next((k for k in priority if k in node and node[k]), None)
    if not valid_key:
        raise KeyError("No embeddings found")

    entry = node[valid_key][-1]

    print(f"[INFO] Using embeddings category: {valid_key}")

    return entry["sentences"], entry["embeddings"]


# ---------------------------
# SEMANTIC SEARCH
# ---------------------------
def semantic_retrieval(sem, chunks, lib_embeds, lib_sents):
    chunk_embeds = sem.encode(chunks, convert_to_tensor=True, device=device)

    hits = util.semantic_search(chunk_embeds, lib_embeds.to(device), top_k=TOP_K)

    pairs = []
    for q_idx, qhits in enumerate(hits):
        for h in qhits:
            pairs.append({
                "chunk_idx": q_idx,
                "student_chunk": chunks[q_idx],
                "library_sentence": lib_sents[h["corpus_id"]],
                "semantic_score": float(h["score"])
            })

    return pairs


# ---------------------------
# CROSS ENCODER RERANKING
# ---------------------------
def _clean_ce(text):
    t = text.replace("**", "").replace("`", "").replace("#", "")
    return " ".join(t.split()).strip()


def rerank_cross(pairs, cross):
    if not pairs: return []

    inputs = [
        (_clean_ce(p["student_chunk"]), _clean_ce(p["library_sentence"]))
        for p in pairs
    ]

    scores = cross.predict(inputs, show_progress_bar=True, batch_size=CROSS_BATCH)

    for i, s in enumerate(scores):
        pairs[i]["cross_score"] = float(s)

    return sorted(pairs, key=lambda x: x["cross_score"], reverse=True)


# ---------------------------
# NLI CHECK
# ---------------------------
def run_nli(pairs, tok, model):
    out = []

    for i in range(0, len(pairs), NLI_BATCH):
        batch = pairs[i:i+NLI_BATCH]

        prem = [full_clean(p["library_sentence"]) for p in batch]
        hyp = [full_clean(p["student_chunk"]) for p in batch]

        enc = tok(prem, hyp, truncation=True, padding=True,
                  return_tensors="pt", max_length=512).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=1).cpu().numpy()

        for j, p in enumerate(batch):
            contr, neut, ent = probs[j]

            out.append({
                **p,
                "nli_probs": {
                    "contradiction": round(float(contr), 4),
                    "neutral": round(float(neut), 4),
                    "entailment": round(float(ent), 4)
                }
            })

    return out


# ---------------------------
# SCORE & AGGREGATE
# ---------------------------
def score(merged, map_idx, total_ans):
    def sig(x): return 1 / (1 + torch.exp(torch.tensor(-x)).item())

    ans_scores = [0.0] * total_ans
    ans_meta = [[] for _ in range(total_ans)]

    for m in merged:
        idx = map_idx[m["chunk_idx"]]
        ent = m["nli_probs"]["entailment"]
        neut = m["nli_probs"]["neutral"]

        cs = max(min(m["cross_score"], SCORE_CLIP[1]), SCORE_CLIP[0])
        chunk_score = sig(cs) * (ent + 0.6 * neut)

        ans_scores[idx] = max(ans_scores[idx], chunk_score)
        ans_meta[idx].append(m)

    total = sum(ans_scores)
    percent = round((total / total_ans) * 100, 2)

    return ans_scores, ans_meta, total, percent


# ---------------------------
# MAIN
# ---------------------------
def evaluate_v2(school_type, subject, board=None, class_name=None, category=None):
    t0 = time.time()

    sem, cross, tok, nli_model = load_models()
    struct = load_student_structured()
    chunks, cmap = chunk_student_answers_for_search(struct)

    if not chunks:
        print("No student chunks.")
        return

    lib_sents, lib_embeds = load_library_embeddings(school_type, subject, board, class_name, category)

    pairs = semantic_retrieval(sem, chunks, lib_embeds, lib_sents)
    pairs = rerank_cross(pairs, cross)
    pairs = run_nli(pairs, tok, nli_model)

    num_answers = len(struct)
    scores, meta, total, percent = score(pairs, cmap, num_answers)

    graded = {
        "total_marks": round(total, 2),
        "max_marks": num_answers,
        "percentage": percent,
        "answers": meta
    }

    with open("graded_results_v2.json", "w") as f:
        json.dump(graded, f, indent=4)

    print(f"Done in {time.time()-t0:.2f}s — Score: {total}/{num_answers} ({percent}%)")

    return graded


if __name__ == "__main__":
    st = input("school/college: ").strip().lower()
    sub = input("subject: ").strip()
    if st == "school":
        brd = input("board: ").strip()
        cls = input("class: ").strip()
    else:
        brd = None
        cls = None

    evaluate_v2(st, sub, brd, cls, category=None)
