import re
import json
import os
from B_normalize import full_clean

def clean_text(t):
    return re.sub(r"\s+", " ", t).strip()

def remove_headers(text):
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

def unify_answer_numbers(text):
    text = re.sub(r"\b(\d+)\s*\.\s*", r" Answer \1 ", text)
    text = re.sub(r"\b(\d+)\s*\)\s*", r" Answer \1 ", text)
    text = re.sub(r"\(\s*(\d+)\s*\)", r" Answer \1 ", text)
    text = re.sub(r"\bq[\s\.\-]*(\d+)", r" Answer \1 ", text, flags=re.IGNORECASE)
    return text

def parse_multilevel(answer):
    parts = {}

    roman = re.split(r"(?=\([ivx]+\))", answer, flags=re.IGNORECASE)
    if len(roman) > 1:
        for r in roman:
            r = r.strip()
            if not r:
                continue
            m = re.match(r"\(([ivx]+)\)\s*(.*)", r, flags=re.IGNORECASE | re.DOTALL)
            if m:
                key = f"({m.group(1)})"
                body = m.group(2).strip()
                letters = re.split(r"(?=\([a-z]\))", body)
                if len(letters) > 1:
                    sub = {}
                    for l in letters:
                        l = l.strip()
                        mm = re.match(r"\(([a-z])\)\s*(.*)", l, re.DOTALL)
                        if mm:
                            sub[f"({mm.group(1)})"] = clean_text(mm.group(2))
                    parts[key] = sub
                else:
                    parts[key] = clean_text(body)
        return parts

    letters = re.split(r"(?=\([a-z]\))", answer)
    if len(letters) > 1:
        for l in letters:
            l = l.strip()
            m = re.match(r"\(([a-z])\)\s*(.*)", l, re.DOTALL)
            if m:
                parts[f"({m.group(1)})"] = clean_text(m.group(2))
        return parts

    return clean_text(answer)

def extract_answers(text):
    answers = []
    pattern = re.compile(r"Answer\s+(\d+)\s+(.*?)(?=Answer\s+\d+|$)", flags=re.DOTALL)
    for m in pattern.finditer(text):
        num = m.group(1)
        body = clean_text(m.group(2))
        structured = parse_multilevel(body)
        wc = len(body.split())
        answers.append({
            "question_number": num,
            "answer_parts": structured,
            "metadata": {
                "word_count": wc,
                "char_count": len(body)
            }
        })
    return answers

def parse_file(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw = data.get("extracted_text", "")

    raw = remove_headers(raw)
    raw = full_clean(raw)
    raw = unify_answer_numbers(raw)

    answers = extract_answers(raw)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=4, ensure_ascii=False)

    print(f"✔ Parsed {len(answers)} answers → {out_path}")

def main():
    print("\n1 = Student\n2 = Reference\n3 = Exit")
    ch = input("Choice: ").strip()

    if ch == "1":
        if os.path.exists("OCR_student_answer.json"):
            parse_file("OCR_student_answer.json", "regrex_student_answer.json")
    elif ch == "2":
        if os.path.exists("OCR_reference_answer.json"):
            parse_file("OCR_reference_answer.json", "regrex_reference_answer.json")
    else:
        return

if __name__ == "__main__":
    main()
