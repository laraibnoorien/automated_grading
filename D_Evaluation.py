import os
import time
import json
from collections import defaultdict

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from B_normalize import full_clean

EMBEDDING_DB_PATH = "embeddings_library.pt"
TEMP_STUDENT_PATH = "temp_student_embeddings.pt"
STUDENT_JSON_PATH = "regrex_student_answer.json"

SEMANTIC_MODEL = "intfloat/e5-base-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL = "roberta-large-mnli"

TOP_K = 5
SCORE_THRESHOLD = 0.0    # keep low, rely on reranker + NLI
NLI_BATCH_SIZE = 16

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def load_student_data():
    data = torch.load(TEMP_STUDENT_PATH)
    student_sents = data["sentences"]
    student_embeds = data["embeddings"]
    with open(STUDENT_JSON_PATH, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    word_counts = []
    for item in json_data:
        wc = item.get("metadata", {}).get("word_count")
        if wc is None:
            # fallback: flatten answer parts length estimate
            parts = item.get("answer_parts", "")
            if isinstance(parts, dict):
                text = " ".join(str(v) for v in parts.values())
            else:
                text = str(parts)
            wc = len(text.split())
        word_counts.append(wc)
    return student_sents, student_embeds, word_counts


def load_library_embeddings(school_type, subject, board=None, class_name=None, category="book"):
    db = torch.load(EMBEDDING_DB_PATH)
    if school_type == "school":
        node = db.get(school_type, {}).get(subject, {}).get(board, {}).get(class_name, {})
    else:
        node = db.get(school_type, {}).get(subject, {})

    possible_keys = [category, "reference", "reference_answer", "book", "teacher", "library"]
    valid_key = next((k for k in possible_keys if k in node and node[k]), None)
    if not valid_key:
        raise KeyError(f"No embeddings found for subject '{subject}' (checked {possible_keys})")
    entries = node[valid_key]
    latest = entries[-1]
    sentences = latest["sentences"]
    embeds = latest["embeddings"]
    return sentences, embeds


def semantic_search(student_embeds, library_embeds, student_sents, library_sents, top_k=TOP_K):
    student_embeds = student_embeds.to(device) if isinstance(student_embeds, torch.Tensor) else student_embeds
    library_embeds = library_embeds.to(device) if isinstance(library_embeds, torch.Tensor) else library_embeds
    hits = util.semantic_search(student_embeds, library_embeds, top_k=top_k)
    pairs = []
    for q_idx, q_hits in enumerate(hits):
        for h in q_hits:
            score = float(h["score"])
            if score >= SCORE_THRESHOLD:
                pairs.append({
                    "student_idx": q_idx,
                    "library_idx": h["corpus_id"],
                    "student_sentence": student_sents[q_idx],
                    "library_sentence": library_sents[h["corpus_id"]],
                    "semantic_score": round(score, 4)
                })
    return pairs


import re

def _clean_for_cross_encoder(text: str) -> str:
    # remove e5 prefixes and simple markdown/inline code artifacts
    text = re.sub(r'\b(query|passage)\s*:\s*', '', text, flags=re.IGNORECASE)
    text = text.replace("**", "")
    text = text.replace("```", "")
    text = re.sub(r'`+', "", text)
    # remove leftover weird punctuation at ends
    text = re.sub(r'^[\s\-\:\#\*]+', '', text)
    text = re.sub(r'[\s\-\:\#\*]+$', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def rerank_with_crossencoder(pairs, cross_encoder, batch_size: int = 16):
    """
    Clean pairs (strip 'query:'/'passage:' and markdown) then re-rank with cross-encoder.
    Returns same list with 'cross_score' added and sorted descending.
    """
    if not pairs:
        return []

    cleaned_pairs = [
        (_clean_for_cross_encoder(p["student_sentence"]),
         _clean_for_cross_encoder(p["library_sentence"]))
        for p in pairs
    ]

    # predict scores (cross_encoder expects list of (query, passage) tuples)
    scores = cross_encoder.predict(cleaned_pairs, show_progress_bar=True, batch_size=batch_size)

    for i, s in enumerate(scores):
        pairs[i]["cross_score"] = float(s)

    pairs.sort(key=lambda x: x["cross_score"], reverse=True)
    return pairs


def nli_batch_check(pairs, tokenizer, model, batch_size=NLI_BATCH_SIZE):
    model.to(device)
    model.eval()
    mapped = []
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        premises = [full_clean(p["library_sentence"]) for p in batch]
        hypotheses = [full_clean(p["student_sentence"]) for p in batch]
        enc = tokenizer(premises, hypotheses, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        for j, p in enumerate(batch):
            prob_vec = probs[j].tolist()
            # map index->label using id2label
            idx = int(torch.argmax(torch.tensor(prob_vec)))
            raw_label = id2label.get(idx, "neutral")
            # normalize probabilities to dict with keys entailment/neutral/contradiction
            label_probs = {}
            # find which id corresponds to standard labels
            # model.config.id2label might be like {0: 'CONTRADICTION',1:'NEUTRAL',2:'ENTAILMENT'}
            for lid, lab in id2label.items():
                label_probs[lab] = round(float(prob_vec[int(lid)]), 4)
            # ensure keys exist
            entail = label_probs.get("entailment", label_probs.get("entails", 0.0))
            neut = label_probs.get("neutral", 0.0)
            contra = label_probs.get("contradiction", 0.0)
            mapped.append({
                **p,
                "nli_label": raw_label,
                "nli_probs": {
                    "entailment": round(entail, 4),
                    "neutral": round(neut, 4),
                    "contradiction": round(contra, 4)
                }
            })
    return mapped


def calculate_marks(hybrid_results, word_counts):
    grouped = defaultdict(list)
    for r in hybrid_results:
        grouped[r["student_sentence"]].append(r)

    marked = []
    total_score = 0.0
    max_marks = len(grouped)

    for idx, (student_ans, matches) in enumerate(grouped.items()):
        best = max(matches, key=lambda x: x.get("cross_score", x.get("semantic_score", 0.0)))
        raw = best.get("cross_score", best.get("semantic_score", 0.0))
        nli_label = best.get("nli_label", "neutral")
        wc = word_counts[idx] if idx < len(word_counts) else 0

        # normalize raw to [0,1]
        if raw < 0 or raw > 1:
            base = 1 / (1 + torch.exp(torch.tensor(-raw))).item()  # sigmoid
        else:
            base = float(raw)
        base = max(0.0, min(1.0, round(base, 4)))

        if nli_label == "entailment" or nli_label == "entails":
            marks = base
        elif nli_label == "neutral":
            marks = base * 0.8
        else:
            marks = base * 0.3

        if wc < 5:
            marks *= 0.5
        elif wc > 40:
            marks = min(1.0, marks + 0.1)

        marks = round(max(0.0, marks), 2)
        total_score += marks

        marked.append({
            **best,
            "raw_score": round(float(raw), 4),
            "normalized_score": base,
            "word_count": wc,
            "marks_awarded": marks
        })

    total_score = round(total_score, 2)
    percentage = round((total_score / max_marks) * 100, 2) if max_marks else 0.0

    graded = {
        "total_marks": total_score,
        "max_marks": max_marks,
        "percentage": percentage,
        "answers": marked
    }
    return graded


def evaluate_answers(school_type, subject, board=None, class_name=None, category="book"):
    t0 = time.time()
    sem_model = SentenceTransformer(SEMANTIC_MODEL, device=device)
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device=device)
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)

    student_sents, student_embeds, word_counts = load_student_data()
    library_sents, library_embeds = load_library_embeddings(school_type, subject, board, class_name, category)

    if not isinstance(student_embeds, torch.Tensor):
        student_embeds = torch.tensor(student_embeds)
    if not isinstance(library_embeds, torch.Tensor):
        library_embeds = torch.tensor(library_embeds)

    student_embeds = student_embeds.to(device)
    library_embeds = library_embeds.to(device)

    pairs = semantic_search(student_embeds, library_embeds, student_sents, library_sents, top_k=TOP_K)
    if not pairs:
        print("No semantic matches found.")
        return {}

    pairs = rerank_with_crossencoder(pairs, cross_encoder)
    hybrid = nli_batch_check(pairs, nli_tokenizer, nli_model, NLI_BATCH_SIZE)

    with open("hybrid_results.json", "w", encoding="utf-8") as f:
        json.dump(hybrid, f, indent=4, ensure_ascii=False)

    graded = calculate_marks(hybrid, word_counts)
    with open("graded_results.json", "w", encoding="utf-8") as f:
        json.dump(graded, f, indent=4, ensure_ascii=False)

    print(f"Done in {time.time()-t0:.2f}s — Score: {graded['total_marks']}/{graded['max_marks']} ({graded['percentage']}%)")
    return graded


if __name__ == "__main__":
    school_type = input("school/college: ").strip().lower()
    subject = input("subject: ").strip()
    if school_type == "school":
        board = input("board: ").strip()
        class_name = input("class: ").strip()
    else:
        board = None
        class_name = None

    evaluate_answers(school_type, subject, board, class_name)
