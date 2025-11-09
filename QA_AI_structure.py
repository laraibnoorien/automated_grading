from transformers import pipeline
import json, re

# use flan-t5-small (lightweight)
model = pipeline("text2text-generation", model="google/flan-t5-small")

def clean_json_output(raw_text):
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return {"raw_output": raw_text.strip()}
    return {"raw_output": raw_text.strip()}


def parse_ocr_answers(input_path, output_path):
    print(f"[+] Parsing {input_path}...")

    with open(input_path, "r") as f:
        ocr_text = f.read()

    # smaller chunks = fewer token overflows
    words = ocr_text.split()
    chunks = [' '.join(words[i:i+250]) for i in range(0, len(words), 250)]

    all_sections = {}

    for i, chunk in enumerate(chunks):
        print(f" → processing chunk {i+1}/{len(chunks)}")
        prompt = f"""
You are a text parser that converts OCR exam answers into clean JSON.

Rules:
- Detect sections (Section A/B/C, I/II/III, Part 1/2 etc.)
- Detect question numbers (1., i), Q1, etc.)
- Each question → full answer text.
- If no section, use "General".
Output only valid JSON:
{{
  "Section Name": {{
    "Question Number": "Answer"
  }}
}}

Text:
{chunk}
"""

        try:
            result = model(prompt, max_new_tokens=256)[0]["generated_text"]
        except Exception as e:
            print(f"   ⚠️ model error on chunk {i+1}: {e}")
            continue

        parsed = clean_json_output(result)

        # merge safely
        for section, qas in parsed.items():
            if isinstance(qas, dict):
                all_sections.setdefault(section, {}).update(qas)
            else:
                # skip malformed section
                all_sections.setdefault("Unstructured", {})[f"chunk_{i+1}"] = qas

    with open(output_path, "w") as f:
        json.dump(all_sections, f, indent=2)

    print(f"[+] Saved structured output to {output_path}")


if __name__ == "__main__":
    file_name = input("\n 1. student_answer \n 2.reference_answer \n 3.both\nEnter choice: ")
    if file_name == "1":
        parse_ocr_answers("OCR_student_answer.json", "AI_student_answer.json")
    elif file_name == "2":
        parse_ocr_answers("OCR_reference_answer.json", "AI_reference_answer.json")
    elif file_name == "3":
        parse_ocr_answers("OCR_student_answer.json", "AI_student_answer.json")
        parse_ocr_answers("OCR_reference_answer.json", "AI_reference_answer.json")
