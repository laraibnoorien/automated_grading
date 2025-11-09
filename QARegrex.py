import re
import json

def parse_ocr_text_to_qa_unified(input_file, output_file):
    # Load OCR JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle either {"extracted_text": "..."} or [{"text": "..."}]
    if isinstance(data, dict) and "extracted_text" in data:
        text = data["extracted_text"]
    elif isinstance(data, list):
        text = " ".join(item.get("text", "") for item in data)
    else:
        raise ValueError("Invalid OCR JSON format.")

    # Clean and normalize
    text = re.sub(r'---.*?---', '', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # remove markdown bold
    text = re.sub(r'Here.*?image:?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n+', '\n', text.strip())

    # ✅ Split only when Q. or a number appears at start of line
    questions = re.split(r'(?=^(?:Q[\.\-]?\d+[\).]?|\d+\.[\)]?)\s)', text, flags=re.MULTILINE)
    qa_data = []

    for q_block in questions:
        q_block = q_block.strip()
        if not q_block:
            continue

        # Extract question number and text
        match = re.match(r'^(?:Q[\.\-]?|\b)(\d+)[\).]?\s*(.*)', q_block, re.DOTALL)
        if not match:
            continue

        q_num = match.group(1).strip()
        content = match.group(2).strip()

        # Split by Roman numerals like (i), (ii)
        roman_parts = re.split(r'(?=\([ivx]+\))', content, flags=re.IGNORECASE)
        parts_dict = {}

        for part in roman_parts:
            part = part.strip()
            if not part:
                continue

            roman_match = re.match(r'(\([ivx]+\))\s*(.*)', part, re.DOTALL | re.IGNORECASE)
            if roman_match:
                roman_key = roman_match.group(1)
                roman_content = roman_match.group(2).strip()

                # Inside Roman, split subparts (a), (b), etc.
                letter_parts = re.split(r'(?=\([a-z]\))', roman_content)
                if len(letter_parts) > 1:
                    sub_dict = {}
                    for sub in letter_parts:
                        sub = sub.strip()
                        if not sub:
                            continue
                        letter_match = re.match(r'(\([a-z]\))\s*(.*)', sub, re.DOTALL)
                        if letter_match:
                            sub_dict[letter_match.group(1)] = letter_match.group(2).strip()
                    parts_dict[roman_key] = sub_dict
                else:
                    parts_dict[roman_key] = roman_content
            else:
                parts_dict["main"] = part

        qa_data.append({
            "question_number": q_num,
            "parts": parts_dict
        })

    # Save the result
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, indent=4, ensure_ascii=False)

    print(f"✅ Parsed {len(qa_data)} main questions → saved to {output_file}")



if __name__ == "__main__":
    to_embed = input("choose \n 1.student_answer \n 2.reference_answer \n 3.book_reference \n 4.both 1 and 2 \n 5.both 1 and 3\n")
    if to_embed == "1":
        parse_ocr_text_to_qa_unified("OCR_student_answer.json", "regrex_student_answer.json")
    elif to_embed == "2":
        parse_ocr_text_to_qa_unified("OCR_reference_answer.json", "regrex_reference_answer.json")
    elif to_embed == "3":
        parse_ocr_text_to_qa_unified("OCR_Book.json", "rergex_Book.json")
    elif to_embed == "4":
        parse_ocr_text_to_qa_unified("OCR_student_answer.json", "regrex_student_answer.json")
        parse_ocr_text_to_qa_unified("OCR_reference_answer.json", "regrex_reference_answer.json")
    elif to_embed == "5":
        parse_ocr_text_to_qa_unified("OCR_student_answer.json", "regrex_student_answer.json")
        parse_ocr_text_to_qa_unified("OCR_Book.json", "regrex_Book.json")
