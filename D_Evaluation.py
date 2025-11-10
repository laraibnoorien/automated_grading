"""
evaluation_pipeline_crossrerank.py
----------------------------------
Enhanced evaluation with Cross-Encoder re-ranking for maximum accuracy.

Pipeline:
1️⃣  Bi-encoder (E5) → fast semantic retrieval  
2️⃣  Cross-encoder (MiniLM) → precise re-ranking of top-K  
3️⃣  NLI (Roberta) → entailment/neutral/contradiction check  
4️⃣  Smooth mark calculation  
5️⃣  Outputs:
      - hybrid_results.json  (semantic + NLI + rerank)
      - graded_results.json  (final marks)
"""

import torch
import json
import time
from collections import defaultdict
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ========== CONFIG ==========
EMBEDDING_DB_PATH = "embeddings_library.pt"
TEMP_STUDENT_PATH = "temp_student_embeddings.pt"
STUDENT_JSON_PATH = "regrex_student_answer.json"

SEMANTIC_MODEL = "intfloat/e5-base-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
#NLI_MODEL = "microsoft/deberta-v3-base-mnli"
NLI_MODEL = "roberta-large-mnli"


TOP_K = 5
RE_RANK_TOP_N = 1
SCORE_THRESHOLD = 0.55
NLI_BATCH_SIZE = 16
# ============================

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using NVIDIA GPU (CUDA)")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU (no GPU detected)")
# ---------- LOADERS ----------
def load_student_data():
    emb_data = torch.load(TEMP_STUDENT_PATH)
    with open(STUDENT_JSON_PATH, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    student_sents = emb_data["sentences"]
    student_embeds = emb_data["embeddings"]
    word_counts = [item.get("metadata", {}).get("word_count", len(item.get("answer_parts", "").split())) for item in json_data]
    return student_sents, student_embeds, word_counts


def load_library_embeddings(school_type, subject, board=None, class_name=None, category="book"):
    data = torch.load(EMBEDDING_DB_PATH)
    if school_type == "school":
        entries = data[school_type][subject][board][class_name][category]
    else:
        entries = data[school_type][subject][category]
    latest = entries[-1]
    return latest["sentences"], latest["embeddings"]

# ---------- SEMANTIC SEARCH ----------
def semantic_search(student_embeds, library_embeds, student_sents, library_sents, top_k=TOP_K):
    hits = util.semantic_search(student_embeds, library_embeds, top_k=top_k)
    results = []
    for i, query_hits in enumerate(hits):
        for hit in query_hits:
            if hit['score'] >= SCORE_THRESHOLD:
                results.append({
                    "student_sentence": student_sents[i],
                    "library_sentence": library_sents[hit['corpus_id']],
                    "semantic_score": round(float(hit['score']), 3)
                })
    return results

# ---------- CROSS-ENCODER RE-RANKING ----------
def rerank_with_crossencoder(matches, cross_encoder):
    """Re-rank retrieved pairs with a cross-encoder for precision."""
    print("[*] Re-ranking with Cross-Encoder...")
    unique_pairs = [(m["student_sentence"], m["library_sentence"]) for m in matches]
    scores = cross_encoder.predict(unique_pairs, show_progress_bar=True, batch_size=16)
    for i, s in enumerate(scores):
        matches[i]["cross_score"] = round(float(s), 3)
    # Sort descending by cross_score
    matches.sort(key=lambda x: x["cross_score"], reverse=True)
    return matches

# ---------- NLI ----------
def nli_batch_check(pairs, tokenizer, model, batch_size=NLI_BATCH_SIZE):
    model.to(device)
    model.eval()
    results = []
    labels = ['entailment', 'neutral', 'contradiction']

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        premises = [p["library_sentence"] for p in batch]
        hypotheses = [p["student_sentence"] for p in batch]

        inputs = tokenizer(premises, hypotheses, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu()

        for j, p in enumerate(batch):
            prob_vec = probs[j].tolist()
            label = labels[int(torch.argmax(probs[j]))]
            results.append({
                **p,
                "nli_label": label,
                "nli_probs": {
                    "entailment": round(prob_vec[0], 3),
                    "neutral": round(prob_vec[1], 3),
                    "contradiction": round(prob_vec[2], 3)
                }
            })
    return results

# ---------- MARKING ----------
def calculate_marks(hybrid_results, word_counts):
    """
    Compute marks per answer safely.
    Ensures no negatives, scales all model scores to [0,1].
    """
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in hybrid_results:
        grouped[r["student_sentence"]].append(r)

    marked = []
    total_score = 0.0
    max_marks = len(grouped)

    for idx, (student_ans, matches) in enumerate(grouped.items()):
        # Pick the best match (cross-encoder preferred)
        best = max(matches, key=lambda x: x.get("cross_score", x.get("semantic_score", 0)))
        raw_score = best.get("cross_score", best.get("semantic_score", 0))
        nli = best.get("nli_label", "neutral")
        wc = word_counts[idx] if idx < len(word_counts) else 0

        # --- Normalization ---
        # Convert any [-1,1] or arbitrary range to [0,1]
        base = (raw_score + 1) / 2 if raw_score < 0 or raw_score > 1 else raw_score
        base = max(0.0, min(1.0, round(base, 3)))

        # --- Weighted scoring by NLI ---
        if nli == "entailment":
            marks = base
        elif nli == "neutral":
            marks = base * 0.8
        else:  # contradiction or uncertain
            marks = base * 0.3

        # --- Word count penalty / bonus ---
        if wc < 5:
            marks *= 0.5  # too short
        elif wc > 40:
            marks = min(1.0, marks + 0.1)  # small bonus

        marks = round(max(0.0, marks), 2)
        total_score += marks

        marked.append({
            **best,
            "raw_score": round(float(raw_score), 3),
            "normalized_score": base,
            "word_count": wc,
            "marks_awarded": marks
        })

    # --- Final summary ---
    total_score = round(total_score, 2)
    percentage = round((total_score / max_marks) * 100, 2) if max_marks else 0.0

    graded_output = {
        "total_marks": total_score,
        "max_marks": max_marks,
        "percentage": percentage,
        "answers": marked
    }

    return graded_output["answers"], total_score, max_marks

# ---------- MAIN ----------
def evaluate_answers(school_type, subject, board=None, class_name=None, category="book"):
    print("\n=== Evaluation with Cross-Encoder Re-ranking ===")
    t0 = time.time()

    # Load models
    sem_model = SentenceTransformer(SEMANTIC_MODEL, device=device)
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device=device)
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)

    # Load embeddings/data
    student_sents, student_embeds, word_counts = load_student_data()
    library_sents, library_embeds = load_library_embeddings(school_type, subject, board, class_name, category)

    if not isinstance(student_embeds, torch.Tensor):
        student_embeds = torch.tensor(student_embeds)
    if not isinstance(library_embeds, torch.Tensor):
        library_embeds = torch.tensor(library_embeds)

    # Step 1: Semantic retrieval
    print("[*] Semantic search (bi-encoder)...")
    matches = semantic_search(student_embeds, library_embeds, student_sents, library_sents, top_k=TOP_K)
    print(f" → {len(matches)} candidate pairs")

    # Step 2: Re-rank with Cross-Encoder
    matches = rerank_with_crossencoder(matches, cross_encoder)

    # Step 3: NLI Verification
    print(f"[*] Running NLI on {len(matches)} pairs...")
    hybrid_results = nli_batch_check(matches, nli_tokenizer, nli_model, NLI_BATCH_SIZE)

    # Save intermediate
    with open("hybrid_results.json", "w", encoding="utf-8") as f:
        json.dump(hybrid_results, f, indent=4, ensure_ascii=False)
    print("✅ hybrid_results.json saved")

    # Step 4: Mark Calculation
    marked_results, total_score, max_marks = calculate_marks(hybrid_results, word_counts)
    graded_output = {
        "total_marks": total_score,
        "max_marks": max_marks,
        "percentage": round((total_score / max_marks) * 100, 2) if max_marks else 0.0,
        "answers": marked_results
    }

    with open("graded_results.json", "w", encoding="utf-8") as f:
        json.dump(graded_output, f, indent=4, ensure_ascii=False)
    print("✅ graded_results.json saved")

    print(f"\n🏁 Done in {time.time()-t0:.2f}s — Final Score: {total_score}/{max_marks} ({graded_output['percentage']}%)")
    return graded_output


# ---------- CLI ----------
if __name__ == "__main__":
    school_type = input("Enter institution type (school/college): ").strip().lower()
    subject = input("Enter subject: ").strip()
    if school_type == "school":
        board = input("Enter board: ").strip()
        class_name = input("Enter class: ").strip()
    else:
        board = None
        class_name = None

    evaluate_answers(school_type, subject, board, class_name)
