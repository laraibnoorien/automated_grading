import re
import json
from nltk.tokenize import sent_tokenize
from B_normalize import full_clean

def clean_line(text):
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)   # remove images
    text = re.sub(r'\*{1,2}', '', text)
    text = text.strip()
    return text

def chunk_text(text, max_words=100, overlap=20):
    sentences = sent_tokenize(text)
    chunks, current = [], []
    count = 0
    for sent in sentences:
        words = sent.split()
        ln = len(words)
        if count + ln > max_words and current:
            chunks.append(" ".join(current))
            overlap_words = " ".join(current).split()[-overlap:]
            current = overlap_words[:] if overlap_words else []
            count = len(current)
        current.append(sent)
        count += ln
    if current:
        chunks.append(" ".join(current))
    return chunks

def parse_book_ocr(input_file, output_file="regrex_book.json", chunk_size=100):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "extracted_text" in data:
        raw_text = data["extracted_text"]
    elif isinstance(data, list):
        raw_text = " ".join(item.get("text", "") for item in data)
    else:
        raise ValueError("Invalid OCR JSON format")

    text = full_clean(raw_text)

    # use sentence-like split first, fall back to newline when sentences are too long
    paragraphs = re.split(r'\n{1,}', text)
    if len(paragraphs) == 1:
        paragraphs = re.split(r'(?<=[\.!?])\s+', text)

    structured = []
    intro_buffer = []
    current_chapter = None
    current_topic = None

    for para in paragraphs:
        line = clean_line(para).strip()
        if not line:
            continue

        chapter_match = re.match(r'chapter\s*\d+[:\-\s]*?(.*)', line, re.IGNORECASE)
        if chapter_match:
            if intro_buffer:
                structured.append({
                    "chapter_title": "Intro / Preface",
                    "topics": [{"topic_title": "General Intro", "content": " ".join(intro_buffer).strip()}]
                })
                intro_buffer = []
            chapter_title = "chapter " + re.sub(r'chapter\s*', '', chapter_match.group(0), flags=re.IGNORECASE).strip()
            current_chapter = {"chapter_title": chapter_title, "topics": []}
            structured.append(current_chapter)
            current_topic = None
            continue

        heading_match = re.match(r'(^#{1,3}\s*(.*))|(^topic[:\-\s]*(.*))|(^section[:\-\s]*(.*))', line, re.IGNORECASE)
        if heading_match:
            # pick whichever captured group has content
            title = next((g for g in heading_match.groups() if g and not g.startswith('#')), None)
            if not title:
                title = re.sub(r'^#{1,3}\s*', '', line).strip()
            topic_title = title.strip()
            current_topic = {"topic_title": topic_title, "content": ""}
            if current_chapter is not None:
                current_chapter["topics"].append(current_topic)
            else:
                # if no chapter yet, put into intro as a pseudo-chapter
                if not structured or structured[-1].get("chapter_title") != "Intro / Preface":
                    structured.append({
                        "chapter_title": "Intro / Preface",
                        "topics": []
                    })
                structured[-1]["topics"].append(current_topic)
            continue

        if current_chapter:
            if current_topic is None:
                current_topic = {"topic_title": "", "content": ""}
                current_chapter["topics"].append(current_topic)
            current_topic["content"] += " " + line
        else:
            intro_buffer.append(line)

    if intro_buffer:
        structured.append({
            "chapter_title": "Intro / Preface",
            "topics": [{"topic_title": "General Intro", "content": " ".join(intro_buffer).strip()}]
        })

    for chapter in structured:
        for topic in chapter.get("topics", []):
            content = topic.get("content", "").strip()
            topic["chunks"] = chunk_text(content, chunk_size, overlap=20) if content else []
            topic["content"] = content

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=4, ensure_ascii=False)

    print(f"✔ Parsed {len(structured)} chapters saved to {output_file}")

if __name__ == "__main__":
    parse_book_ocr("OCR_Book.json")
