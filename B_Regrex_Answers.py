# regrex_answer.py
import re
import json
import os
from typing import Union, Dict, Any, List
from B_normalize_answer import normalize_text
from nltk.tokenize import sent_tokenize

# --- chunker: sentence-group chunking (configurable) -----------------------
def sentence_group_chunk(text: str, group_size: int = 3, overlap: int = 1) -> List[str]:
    if not text:
        return []
    sents = sent_tokenize(text)
    if not sents:
        return []
    chunks = []
    i = 0
    n = len(sents)
    while i < n:
        chunk = " ".join(sents[i:i+group_size]).strip()
        if chunk:
            chunks.append(chunk)
        i += max(1, group_size - overlap)
    return chunks

# --- simple helpers -------------------------------------------------------
def clean_text_whitespace(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()

def remove_headers(text: str) -> str:
    patterns = [
        r"name[:\s].*",
        r"university.*",
        r"program.*",
        r"batch.*",
        r"course code.*",
        r"course title.*",
        r"semester.*",
        r"invigilator.*",
        r"date of examination.*",
        r"here is the (extracted|text).*"
    ]
    for p in patterns:
        text = re.sub(p, " ", text, flags=re.IGNORECASE)
    return text

def unify_answer_numbers(text: str) -> str:
    t = text
    # convert common forms to "Answer N "
    t = re.sub(r"(?m)^\s*(q|question)\s*[:\.\-]?\s*(\d+)\b", r"Answer \2 ", t, flags=re.IGNORECASE)
    t = re.sub(r"(?m)^\s*(\d+)\s*\.\s+", r"Answer \1 ", t)
    t = re.sub(r"(?m)^\s*(\d+)\s*\)\s+", r"Answer \1 ", t)
    t = re.sub(r"(?m)^\s*\(\s*(\d+)\s*\)\s+", r"Answer \1 ", t)
    return t

# parse multi-level (simple) — keep structure but don't deeply normalize nested roman parsing
def parse_multilevel(answer_text: str) -> Union[str, Dict[str, Any]]:
    # try letter groups (a), (b) etc.
    answer_text = answer_text.strip()
    letter_splits = re.split(r"(?=\(\s*[a-z]\s*\))", answer_text)
    if len(letter_splits) > 1:
        out = {}
        for part in letter_splits:
            part = part.strip()
            if not part:
                continue
            m = re.match(r"\(\s*([a-z])\s*\)\s*(.*)", part, flags=re.DOTALL)
            if m:
                out_key = f"({m.group(1)})"
                out[out_key] = normalize_text(m.group(2).strip(), mode="answer")["normalized"]
        return out

    # fallback single string normalized
    return normalize_text(answer_text, mode="answer")["normalized"]

# extract answers blocks
def extract_answers(unified_text: str):
    results = []
    pattern = re.compile(r"Answer\s+(\d+)\s+(.*?)(?=Answer\s+\d+|$)", flags=re.DOTALL | re.IGNORECASE)
    for m in pattern.finditer(unified_text):
        qnum = m.group(1)
        body = m.group(2).strip()
        if not body:
            continue
        raw_answer = clean_text_whitespace(body)
        parts = parse_multilevel(body)
        if isinstance(parts, dict):
            # flatten for search: combine parts
            answer_for_search = " ".join([v for v in parts.values() if isinstance(v, str)])
            answer_clean = normalize_text(answer_for_search, mode="answer")["normalized"]
            answer_parts = parts
        else:
            answer_clean = parts
            answer_parts = parts

        wc = len(answer_clean.split())
        # chunk the answer for retrieval (sentence-group chunking)
        chunks = sentence_group_chunk(answer_clean, group_size=3, overlap=1)
        results.append({
            "question_number": str(qnum),
            "raw_answer": raw_answer,
            "answer": answer_clean,
            "answer_parts": answer_parts,
            "chunks": chunks,
            "metadata": {"word_count": wc, "char_count": len(answer_clean)}
        })
    return results

def parse_file(in_path: str, out_path: str):
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("extracted_text", "")
    if not raw:
        print("No 'extracted_text' found in", in_path)
        return
    raw = remove_headers(raw)
    unified = unify_answer_numbers(raw)
    answers = extract_answers(unified)
    # fallback: if nothing found, heuristically split by numbered lines
    if not answers:
        fallback_pattern = re.split(r"(?m)^\s*\d+\s*(?:\)|\.)\s+", raw)
        if len(fallback_pattern) > 1:
            nums = re.findall(r"(?m)^\s*(\d+)\s*(?:\)|\.)", raw)
            bodies = fallback_pattern[1:]
            possible = []
            for n, b in zip(nums, bodies):
                b_clean = clean_text_whitespace(b)
                parts = parse_multilevel(b_clean)
                if isinstance(parts, dict):
                    answer_clean = " ".join([v for v in parts.values() if isinstance(v, str)])
                else:
                    answer_clean = parts
                answer_clean = normalize_text(answer_clean, mode="answer")["normalized"]
                chunks = sentence_group_chunk(answer_clean, group_size=3, overlap=1)
                possible.append({
                    "question_number": str(n),
                    "raw_answer": b_clean,
                    "answer": answer_clean,
                    "answer_parts": parts,
                    "chunks": chunks,
                    "metadata": {"word_count": len(answer_clean.split()), "char_count": len(answer_clean)}
                })
            answers = possible
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2, ensure_ascii=False)
    print(f"✔ Parsed {len(answers)} answers → {out_path}")

# CLI
def main():
    print("\n1 = Student\n2 = Reference\n3 = Exit")
    ch = input("Choice: ").strip()
    if ch == "1":
        if os.path.exists("OCR_student_answer.json"):
            parse_file("OCR_student_answer.json", "regrex_student_answer.json")
        else:
            print("Missing OCR_student_answer.json")
    elif ch == "2":
        if os.path.exists("OCR_reference_answer.json"):
            parse_file("OCR_reference_answer.json", "regrex_reference_answer.json")
        else:
            print("Missing OCR_reference_answer.json")
    else:
        return

if __name__ == "__main__":
    main()
