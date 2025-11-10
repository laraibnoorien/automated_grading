import re
import json
from nltk.tokenize import sent_tokenize

def clean_text(text):
    """Remove page markers, image links, extra spaces, and Markdown formatting."""
    text = re.sub(r'--- Page \d+ ---', '', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # remove images
    text = re.sub(r'\*{1,2}', '', text)  # remove bold/italic
    text = re.sub(r'\s+', ' ', text)  # normalize spaces
    return text.strip()


def chunk_text(text, max_words=100, overlap=20):
    """Split text into overlapping chunks of sentences to preserve context."""
    sentences = sent_tokenize(text)
    chunks, current = [], []
    count = 0

    for sent in sentences:
        words = sent.split()
        sent_len = len(words)

        if count + sent_len > max_words:
            chunks.append(" ".join(current))
            # Create overlap from last 'overlap' words of the previous chunk
            overlap_words = " ".join(current).split()[-overlap:]
            current = overlap_words[:] if overlap_words else []
            count = len(current)

        current.append(sent)
        count += sent_len

    if current:
        chunks.append(" ".join(current))

    return chunks



def parse_book_ocr(input_file, output_file="regrex_book.json", chunk_size=100):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "extracted_text" in data:
        text = data["extracted_text"]
    elif isinstance(data, list):
        text = " ".join(item.get("text", "") for item in data)
    else:
        raise ValueError("Invalid OCR JSON format")

    lines = text.splitlines()
    structured = []
    current_chapter = None
    current_topic = None

    # NEW: buffer for intro text before any chapter/topic
    intro_buffer = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        chapter_match = re.match(r'#\s*CHAPTER\s*\d+[:\-]?\s*(.*)', line, re.IGNORECASE)
        if chapter_match:
            if intro_buffer:  # store intro text as Chapter 0
                structured.append({
                    "chapter_title": "Intro / Preface",
                    "topics": [{"topic_title": "General Intro", "content": " ".join(intro_buffer)}]
                })
                intro_buffer = []
            chapter_title = clean_text(chapter_match.group(0))
            current_chapter = {"chapter_title": chapter_title, "topics": []}
            structured.append(current_chapter)
            current_topic = None
            continue

        topic_match = re.match(r'##\s*(.*)', line)
        if topic_match:
            topic_title = clean_text(topic_match.group(1))
            current_topic = {"topic_title": topic_title, "content": ""}
            if current_chapter is not None:
                current_chapter["topics"].append(current_topic)
            continue

        # Add content
        if current_chapter is not None:
            if current_topic is None:
                current_chapter["topics"].append({"topic_title": "", "content": ""})
                current_topic = current_chapter["topics"][-1]
            current_topic["content"] += " " + clean_text(line)
        else:
            intro_buffer.append(clean_text(line))

    # Handle remaining intro text
    if intro_buffer:
        structured.append({
            "chapter_title": "Intro / Preface",
            "topics": [{"topic_title": "General Intro", "content": " ".join(intro_buffer)}]
        })

    # Split content into chunks
    # Split content into overlapping chunks
    for chapter in structured:
        for topic in chapter["topics"]:
            content = topic["content"].strip()
            if content:
                topic["chunks"] = chunk_text(content, chunk_size, overlap=20)
            else:
                topic["chunks"] = []
            topic["content"] = content


    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=4, ensure_ascii=False)

    print(f"✅ Parsed {len(structured)} chapters (including intro) saved to {output_file}")



if __name__ == "__main__":
    parse_book_ocr("OCR_Book.json")
