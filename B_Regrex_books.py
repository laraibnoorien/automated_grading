import re
import json
from nltk.tokenize import sent_tokenize
from B_normalize_book import normalize_book_text

# ======================================================
# CHUNKER — optimal for FAISS + NLI + Cross-Encoder
# ======================================================
def chunk_text(text, max_words=45, overlap=10):
    text = text.strip()
    if not text:
        return []

    sentences = sent_tokenize(text)
    chunks, current = [], []
    count = 0

    for sent in sentences:
        words = sent.split()
        ln = len(words)

        if count + ln > max_words and current:
            chunks.append(" ".join(current))

            tail = " ".join(current).split()[-overlap:]
            current = tail[:] if tail else []
            count = len(current)

        current.append(sent)
        count += ln

    if current:
        chunks.append(" ".join(current))

    return chunks


# ======================================================
# SMART HEADING PATTERNS (covers 99% NCERT/CBSE books)
# ======================================================
HEADING_PATTERNS = [
    r"^chapter\s+\d+",
    r"^\d+\s*[\.:]\s+[A-Z].+",
    r"^\d+\.\d+\s+[A-Z].+",
    r"^\d+\.\d+\.\d+\s+[A-Z].+",
    r"^matter in .*",  
    r"^activity\s*\d+",
    r"^exercises?$",
    r"^exercise\s*\d+",
    r"^summary$",
    r"^group activity$",
    r"^introduction$",
]

def looks_like_heading(line):
    t = line.strip()
    for p in HEADING_PATTERNS:
        if re.match(p, t, flags=re.IGNORECASE):
            return True
    return False


# ======================================================
# MAIN BOOK PARSER
# ======================================================
def parse_book_ocr_text(text):

    lines = text.split("\n")
    sections = []
    current_title = None
    current_content = []

    for line in lines:
        clean = line.strip()
        if not clean:
            continue

        # detect headings
        if looks_like_heading(clean):
            if current_title and current_content:
                content = "\n".join(current_content).strip()
                sections.append((current_title, content))

            current_title = clean
            current_content = []
        else:
            current_content.append(clean)

    # save final section
    if current_title and current_content:
        content = "\n".join(current_content).strip()
        sections.append((current_title, content))

    # Fallback if nothing matched
    if not sections:
        sections = [("Book Content", text)]

    # Build topics list
    topics = []
    for title, content in sections:
        topics.append({
            "topic_title": title.strip(),
            "content": content,
            "chunks": chunk_text(content)
        })

    return topics


# ======================================================
# WRAPPER
# ======================================================
def parse_book_ocr(input_file, output_file="regrex_book.json"):
    with open(input_file, "r", encoding="utf-8") as f:
        raw = json.load(f).get("extracted_text", "")

    if not raw:
        raise ValueError("Missing 'extracted_text' field")

    # CLEAN OCR TEXT
    text = normalize_book_text(raw)

    topics = parse_book_ocr_text(text)

    output = [{
        "chapter_title": "OCR Book",
        "topics": topics
    }]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"✔ Saved {output_file}")


if __name__ == "__main__":
    parse_book_ocr("OCR_Book.json")
