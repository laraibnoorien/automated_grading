# B_normalize_book.py

"""
Safe cleaner for textbook OCR — preserves scientific notation & symbols.
NEVER use answer-normalization logic here.
"""

import re

def normalize_book_text(text: str) -> str:

    # Remove markdown images
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove OCR artifacts
    text = re.sub(r"image_url", "", text, flags=re.IGNORECASE)

    # Page markers
    text = re.sub(r"---\s*page\s*\d+\s*---", "", text, flags=re.IGNORECASE)
    text = re.sub(r"page\s*\d+\b", "", text, flags=re.IGNORECASE)

    # Copyright / reprint text
    text = re.sub(r"reprint\s*\d{4}.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"©.*", "", text)

    # Remove "not to be reproduced"
    text = re.sub(r"not to be reproduced", "", text, flags=re.IGNORECASE)

    # Remove PDF headers/footers
    text = re.sub(r"(header|footer):.*", "", text, flags=re.IGNORECASE)

    # Remove long separators
    text = re.sub(r"[-_]{3,}", " ", text)

    # Cleanup excessive blank lines
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    return text.strip()
