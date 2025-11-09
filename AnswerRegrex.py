import re
import json

def clean_text(t):
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def parse_multilevel_answer(answer_text):
    subparts = {}

    # Split by Roman numerals
    roman_splits = re.split(r"(?=\([ivx]+\))", answer_text, flags=re.IGNORECASE)
    if len(roman_splits) > 1:
        for part in roman_splits:
            part = part.strip()
            if not part:
                continue
            match = re.match(r"\(([ivx]+)\)\s*(.*)", part, flags=re.IGNORECASE | re.DOTALL)
            if match:
                roman_label = f"({match.group(1)})"
                content = match.group(2).strip()

                # Split inside by (a), (b)
                letter_splits = re.split(r"(?=\([a-z]\))", content)
                if len(letter_splits) > 1:
                    letter_dict = {}
                    for sub in letter_splits:
                        sub = sub.strip()
                        if not sub:
                            continue
                        m2 = re.match(r"\(([a-z])\)\s*(.*)", sub, re.DOTALL)
                        if m2:
                            letter_dict[f"({m2.group(1)})"] = clean_text(m2.group(2))
                    subparts[roman_label] = letter_dict
                else:
                    subparts[roman_label] = clean_text(content)
        return subparts

    # Split by letters if no Roman numerals
    letter_splits = re.split(r"(?=\([a-z]\))", answer_text)
    if len(letter_splits) > 1:
        for sub in letter_splits:
            sub = sub.strip()
            if not sub:
                continue
            m2 = re.match(r"\(([a-z])\)\s*(.*)", sub, re.DOTALL)
            if m2:
                subparts[f"({m2.group(1)})"] = clean_text(m2.group(2))
        return subparts

    return clean_text(answer_text)

def parse_ocr_to_structured_answers(ocr_json_path, output_path):
    with open(ocr_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    text = data["extracted_text"]
    text = re.sub(r"[^\S\r\n]+", " ", text)
    text = re.sub(r"\r", "", text).strip()

    # Detect section labels like SECTION A or **SECTION-B**
    section_pattern = re.compile(r"(?:\*\*)?\s*SECTION[-\s]*(A|B|C)(?:\s*\*\*)?", re.IGNORECASE)
    sections = list(section_pattern.finditer(text))

    # Detect answers like **Ans1)**, Q1, Answer 1, etc.
    answer_pattern = re.compile(
        r"(?:\*\*)?\s*(Ans(?:wer)?|Q(?:ues)?)(\s*\d+)\s*\)*\s*(?:\*\*)?\s*(.*?)(?=\n\s*(?:\*\*)?\s*(?:Ans(?:wer)?|Q(?:ues)?)(?:\s*\d+)|\Z)",
        flags=re.IGNORECASE | re.DOTALL
    )

    answers = []

    def extract_answers(section_text, section_label):
        extracted = []
        for match in answer_pattern.finditer(section_text):
            q_num = re.sub(r"\D", "", match.group(2)) or "?"
            q_id = f"{section_label}{q_num}"
            ans_text = clean_text(match.group(3))
            structured_parts = parse_multilevel_answer(ans_text)
            metadata = {
                "word_count": len(ans_text.split()),
                "line_count": len(ans_text.splitlines()),
                "char_count": len(ans_text)
            }
            extracted.append({
                "question_number": q_num,
                "question_id": q_id,
                "section": section_label,
                "answer_parts": structured_parts,
                "metadata": metadata
            })
        return extracted

    if sections:
        section_positions = [(m.start(), m.group(1).upper()) for m in sections]
        section_positions.append((len(text), None))  # Sentinel
        for i in range(len(section_positions) - 1):
            start, sec_label = section_positions[i]
            end = section_positions[i + 1][0]
            section_text = text[start:end]
            answers.extend(extract_answers(section_text, sec_label))
    else:
        answers.extend(extract_answers(text, "A"))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=4, ensure_ascii=False)

    print(f"✅ Parsed {len(answers)} answers saved to {output_path}")


# --- Menu-based usage ---
if __name__ == "__main__":
    to_embed = input(
        "choose \n 1.student_answer \n 2.reference_answer \n 3.book_reference \n 4.both 1 and 2 \n 5.both 1 and 3\n"
    )

    if to_embed == "1":
        parse_ocr_to_structured_answers("OCR_student_answer.json", "regrex_student_answer.json")

    elif to_embed == "2":
        parse_ocr_to_structured_answers("OCR_reference_answer.json", "regrex_reference_answer.json")

    elif to_embed == "3":
        parse_ocr_to_structured_answers("OCR_Book.json", "regrex_Book.json")

    elif to_embed == "4":
        parse_ocr_to_structured_answers("OCR_student_answer.json", "regrex_student_answer.json")
        parse_ocr_to_structured_answers("OCR_reference_answer.json", "regrex_reference_answer.json")

    elif to_embed == "5":
        parse_ocr_to_structured_answers("OCR_student_answer.json", "regrex_student_answer.json")
        parse_ocr_to_structured_answers("OCR_Book.json", "regrex_Book.json")
