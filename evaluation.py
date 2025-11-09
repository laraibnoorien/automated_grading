import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json

# ====== CONFIG ======
EMBEDDING_DB_PATH = "embeddings_library.pt"
SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL = "roberta-large-mnli"  # accurate NLI model
# ====================

# Load embeddings
def load_embeddings(subject, board, class_name, category):
    data = torch.load(EMBEDDING_DB_PATH)
    entries = data[subject][board][class_name][category]
    latest = entries[-1]
    return latest["sentences"], latest["embeddings"]

# Semantic Search
def semantic_search(student_embeds, book_embeds, student_sents, book_sents, top_k=2):
    hits = util.semantic_search(student_embeds, book_embeds, top_k=top_k)
    results = []
    for i, query_hits in enumerate(hits):
        top_matches = []
        for hit in query_hits:
            top_matches.append({
                "student_sentence": student_sents[i],
                "book_sentence": book_sents[hit['corpus_id']],
                "score": round(hit['score'], 3)
            })
        results.append(top_matches)
    return results

# NLI Checker
def nli_check(premise, hypothesis, tokenizer, model):
    inputs = tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]
    labels = ['entailment', 'neutral', 'contradiction']
    pred_label = labels[torch.argmax(probs)]
    return pred_label, probs.tolist()

# Hybrid Evaluation
def hybrid_check(subject="physics", board="cbse", class_name="10"):
    # Load models
    sem_model = SentenceTransformer(SEMANTIC_MODEL)
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)

    # Load embeddings
    student_sents, student_embeds = load_embeddings(subject, board, class_name, "regrex_student_answer")
    book_sents, book_embeds = load_embeddings(subject, board, class_name, "regrex_book")

    # Convert embeddings to tensors if needed
    if not isinstance(student_embeds, torch.Tensor):
        student_embeds = torch.tensor(student_embeds)
    if not isinstance(book_embeds, torch.Tensor):
        book_embeds = torch.tensor(book_embeds)

    # Perform semantic search
    print("[*] Running semantic similarity search...")
    matches = semantic_search(student_embeds, book_embeds, student_sents, book_sents)

    # Run NLI on top matches
    print("[*] Running NLI verification...")
    results = []
    for pair_list in matches:
        for pair in pair_list:
            label, probs = nli_check(pair["book_sentence"], pair["student_sentence"], nli_tokenizer, nli_model)
            results.append({
                **pair,
                "nli_label": label,
                "nli_probs": {
                    "entailment": round(probs[0], 3),
                    "neutral": round(probs[1], 3),
                    "contradiction": round(probs[2], 3)
                }
            })

    # Save results
    with open("hybrid_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Hybrid results saved to hybrid_results.json")
    return results
if __name__ == "__main__":
    print("\n=== Hybrid Evaluation Stage ===")
    # Ask user for metadata
    subject = input("Enter subject: ").strip()
    board = input("Enter board: ").strip()
    class_name = input("Enter class: ").strip()

    # Optional: ask which embeddings to use
    print("\nSelect categories to evaluate:")
    print("1 → regrex_student_answer")
    print("2 → regrex_reference_answer")
    print("3 → regrex_book")
    print("4 → both 1 and 2")
    print("5 → both 1 and 3")
    choice = input("\nEnter your choice (1-5): ").strip()

    mapping = {
        "1": ["regrex_student_answer"],
        "2": ["regrex_reference_answer"],
        "3": ["regrex_book"],
        "4": ["regrex_student_answer", "regrex_reference_answer"],
        "5": ["regrex_student_answer", "regrex_book"]
    }

    selected_categories = mapping.get(choice, ["regrex_student_answer", "regrex_book"])

    # Run hybrid check
    results = hybrid_check(subject=subject, board=board, class_name=class_name)
