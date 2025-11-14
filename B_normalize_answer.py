import re
import unicodedata

def normalize_text(text: str) -> str:
    # unicode normalize and basic fixes
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")
    text = text.replace("\u00A0", " ")
    text = text.strip()
    text = text.lower()
    text = re.sub(r"[^\x00-\x7f]+", " ", text)  # drop non-ascii junk
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def regex_cleanup(text: str) -> str:
    # remove common page markers, headers/footers, separators, and stray punctuation
    text = re.sub(r"---\s*page\s*\d+\s*---", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"(name|roll|university|program|batch|course code|course title|semester|date of examination|invigilator).*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"page\s*\d+\s*/\s*\d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"(^|\n)\s*page[:\s]*\d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[\-\_\=]{3,}", " ", text)  # long separators
    text = re.sub(r"\s*\d+\s*[:\)]\s*$", " ", text, flags=re.MULTILINE)  # stray line-end numbers
    text = re.sub(r"(^\s*header:.*?$)|(^\s*footer:.*?$)", " ", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"here is the (extracted|text).*", " ", text, flags=re.IGNORECASE)
    return text.strip()

def normalize_numbers_and_labels(text: str) -> str:
    # normalize common question/answer labels for easier regex
    text = re.sub(r"\bq[\s\.\-]*\d+\b", lambda m: m.group(0).replace(" ", ""), text, flags=re.IGNORECASE)
    text = re.sub(r"\b(ans|answer|a ns|a\.ns|ans\.)\b", "answer", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(quest|que|ques|q)\b\.?", "q", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*([ivx]+)\s*\)", lambda m: f"({m.group(1)})", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*([a-z])\s*\)", lambda m: f"({m.group(1)})", text, flags=re.IGNORECASE)
    return text

def full_clean(text: str) -> str:
    t = normalize_text(text)
    t = regex_cleanup(t)
    t = normalize_numbers_and_labels(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
