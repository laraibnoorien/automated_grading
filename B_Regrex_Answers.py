# regrex_answer.py
import re
import json
import os
from typing import Union, Dict, Any

# Try both possible normalizer module names for compatibility
try:
    from B_normalize_answer import full_clean
except Exception:
    try:
        from B_normalize_book import full_clean
    except Exception:
        raise ImportError("Cannot find full_clean in B_normalize_answer.py or B_normalize.py")

# --- helpers --------------------------------------------------------------
def clean_text(t: str) -> str:
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
    """
    Insert a canonical marker 'Answer N' for a variety of question numbering styles.
    """
    t = text
    # common: "1.", "1 )", "1)"
    t = re.sub(r"(?m)^\s*(\d+)\s*\.\s+", r"Answer \1 ", t)
    t = re.sub(r"(?m)^\s*(\d+)\s*\)\s+",  r"Answer \1 ", t)
    # inline like "Q1:" or "Q 1."
    t = re.sub(r"\b[qQ][\s\.\-]*(\d+)\b", r"Answer \1 ", t)
    # parentheses (1)
    t = re.sub(r"(?m)^\s*\(\s*(\d+)\s*\)\s+", r"Answer \1 ", t)
    # ensure "Answer N" tokens are consistently spaced
    t = re.sub(r"\bAnswer\s+0+", "Answer ", t)
    return t

# --- parsing multi-level answer parts -------------------------------------
def parse_multilevel(answer_text: str) -> Union[str, Dict[str, Any]]:
    """
    Parse nested answer parts:
    - First try roman numerals (i), (ii) etc.
    - Then letters (a), (b), ...
    - Return nested dict if parts found, else return cleaned string
    """
    answer_text = answer_text.strip()

    # try roman numerals as top-level
    roman_splits = re.split(r"(?=\(\s*[ivx]+\s*\))", answer_text, flags=re.IGNORECASE)
    if len(roman_splits) > 1:
        out = {}
        for seg in roman_splits:
            seg = seg.strip()
            if not seg:
                continue
            m = re.match(r"\(\s*([ivx]+)\s*\)\s*(.*)", seg, flags=re.IGNORECASE | re.DOTALL)
            if m:
                key = f"({m.group(1)})"
                body = m.group(2).strip()
                # inside roman, try letter parts
                letter_splits = re.split(r"(?=\(\s*[a-z]\s*\))", body)
                if len(letter_splits) > 1:
                    sub = {}
                    for part in letter_splits:
                        part = part.strip()
                        mm = re.match(r"\(\s*([a-z])\s*\)\s*(.*)", part, flags=re.DOTALL)
                        if mm:
                            sub_key = f"({mm.group(1)})"
                            sub[sub_key] = full_clean(mm.group(2).strip())
                    out[key] = sub if sub else full_clean(body)
                else:
                    out[key] = full_clean(body)
        return out

    # try letter-level directly
    letter_splits = re.split(r"(?=\(\s*[a-z]\s*\))", answer_text)
    if len(letter_splits) > 1:
        out = {}
        for part in letter_splits:
            part = part.strip()
            if not part:
                continue
            m = re.match(r"\(\s*([a-z])\s*\)\s*(.*)", part, flags=re.DOTALL)
            if m:
                out[f"({m.group(1)})"] = full_clean(m.group(2).strip())
        return out

    # fallback: single answer string (cleaned)
    return full_clean(answer_text)

# --- extract answers from text --------------------------------------------
def extract_answers(unified_text: str):
    """
    Find all 'Answer N' blocks and return list of dicts with:
      - question_number
      - raw_answer (original body)
      - answer (cleaned string)
      - answer_parts (structured dict or cleaned string)
      - metadata (word_count, char_count)
    """
    results = []
    pattern = re.compile(r"Answer\s+(\d+)\s+(.*?)(?=Answer\s+\d+|$)", flags=re.DOTALL | re.IGNORECASE)
    for m in pattern.finditer(unified_text):
        qnum = m.group(1)
        body = m.group(2).strip()
        if not body:
            continue

        raw_answer = clean_text(body)
        parts = parse_multilevel(body)
        # If parse returns dict, keep as is; if string, store both 'answer' and 'answer_parts' accordingly
        if isinstance(parts, dict):
            answer_for_search = " ".join([v if isinstance(v, str) else " ".join(v.values()) for v in parts.values()])
            answer_clean = full_clean(answer_for_search)
            answer_parts = parts
        else:
            answer_clean = parts
            answer_parts = parts

        wc = len(answer_clean.split())
        results.append({
            "question_number": str(qnum),
            "raw_answer": raw_answer,
            "answer": answer_clean,
            "answer_parts": answer_parts,
            "metadata": {"word_count": wc, "char_count": len(answer_clean)}
        })
    return results

# --- main file parse ------------------------------------------------------
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

    # If no answers found, try a fallback: split by lines that look like '1.' or '1)'
    if not answers:
        fallback_pattern = re.split(r"(?m)^\s*\d+\s*(?:\)|\.)\s+", raw)
        if len(fallback_pattern) > 1:
            # first item probably preface; pair numbers with chunks heuristically
            possible = []
            nums = re.findall(r"(?m)^\s*(\d+)\s*(?:\)|\.)", raw)
            bodies = fallback_pattern[1:]
            for n, b in zip(nums, bodies):
                b = clean_text(b)
                parts = parse_multilevel(b)
                if isinstance(parts, dict):
                    answer_clean = full_clean(" ".join([v if isinstance(v, str) else " ".join(v.values()) for v in parts.values()]))
                else:
                    answer_clean = parts
                possible.append({
                    "question_number": str(n),
                    "raw_answer": b,
                    "answer": answer_clean,
                    "answer_parts": parts,
                    "metadata": {"word_count": len(answer_clean.split()), "char_count": len(answer_clean)}
                })
            answers = possible

    # save
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=4, ensure_ascii=False)

    print(f"✔ Parsed {len(answers)} answers → {out_path}")

# --- CLI ------------------------------------------------------------------
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
