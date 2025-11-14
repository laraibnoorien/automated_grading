# common_normalize.py
import re
import unicodedata
from datetime import datetime

NORMALIZE_VERSION = 1  # increment when you change normalization rules

# base normalizer used by all paths
def _base_normalize(text: str) -> str:
    if not isinstance(text, str):
        text = str(text or "")
    # unicode normalization
    text = unicodedata.normalize("NFKC", text)
    # unify whitespace
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")
    text = text.replace("\u00A0", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

# normalize text intended for books (preserve math/units where possible)
def normalize_for_book(text: str) -> str:
    t = _base_normalize(text)
    # remove common OCR noise but keep non-ascii (for symbols)
    t = re.sub(r"!\[.*?\]\(.*?\)", "", t)         # md images
    t = re.sub(r"http\S+", "", t)                 # urls
    t = re.sub(r"image_url", "", t, flags=re.IGNORECASE)
    t = re.sub(r"---\s*page\s*\d+\s*---", "", t, flags=re.IGNORECASE)
    t = re.sub(r"page\s*\d+\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"reprint\s*\d{4}.*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"©.*", "", t)
    t = re.sub(r"not to be reproduced", "", t, flags=re.IGNORECASE)
    t = re.sub(r"(header|footer):.*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"[-_]{3,}", " ", t)
    # leave mathematical symbols and greek letters intact (don't strip non-ascii)
    t = t.strip()
    return t

# normalize text intended for student answers (more aggressive)
def normalize_for_answer(text: str) -> str:
    t = _base_normalize(text)
    # lowercase and drop weird non-ascii tokens likely produced by OCR in answers
    t = t.lower()
    t = re.sub(r"[^\x00-\x7f]+", " ", t)  # remove non-ascii in answers to reduce noise
    # remove page headers/footers and personal metadata
    t = re.sub(r"(name|roll|university|program|batch|course code|course title|semester|date of examination|invigilator).*", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"here is the (extracted|text).*", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"[\-\_\=]{3,}", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# normalize query text (what you embed for retrieval)
def normalize_for_query(text: str) -> str:
    t = _base_normalize(text)
    t = t.lower()
    t = re.sub(r"[\-\_\=]{3,}", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# wrapper to return normalized and metadata
def normalize_text(text: str, mode: str = "answer"):
    """
    mode: 'book' | 'answer' | 'query'
    returns dict:
      {
        "normalized": str,
        "normalize_version": int,
        "ts": iso timestamp
      }
    """
    if mode == "book":
        norm = normalize_for_book(text)
    elif mode == "query":
        norm = normalize_for_query(text)
    else:
        norm = normalize_for_answer(text)

    return {
        "normalized": norm,
        "normalize_version": NORMALIZE_VERSION,
        "ts": datetime.utcnow().isoformat() + "Z"
    }
