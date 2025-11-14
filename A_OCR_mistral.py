import os
import base64
import requests
from PIL import Image
from pdf2image import convert_from_path
from dotenv import load_dotenv
load_dotenv()
import json
import re

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"
MODEL = "pixtral-large-latest"

# CLEAN OCR TEXT FOR DOWNSTREAM PIPELINE
def clean_raw_ocr(text):
    """Remove markdown images, page numbers, reprint notes, etc."""
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)          # remove markdown images
    text = re.sub(r'Reprint.*\d{4}.*', '', text)         # remove reprint lines
    text = re.sub(r'Page\s*\d+', '', text, flags=re.I)   # remove page numbers
    text = re.sub(r'--- Page .*?---', '', text)          # remove page markers
    text = re.sub(r'image_url', '', text)                # remove weird OCR terms
    text = re.sub(r'©.*', '', text)                      # remove copyright notes
    text = re.sub(r'\*not to be reproduced\*', '', text, flags=re.I)

    # Remove excessive blank lines
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

    return text.strip()


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_text_from_image(image_path):
    """OCR using Mistral Vision."""
    import ssl
    from requests.adapters import HTTPAdapter
    from urllib3.poolmanager import PoolManager

    class SSLAdapter(HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            ctx = ssl.create_default_context()
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            kwargs['ssl_context'] = ctx
            return super().init_poolmanager(*args, **kwargs)

    img_b64 = encode_image(image_path)
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all text only from this image. No images, no markdown."},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"}
                ]
            }
        ]
    }

    session = requests.Session()
    session.mount("https://", SSLAdapter())

    try:
        response = session.post(MISTRAL_ENDPOINT, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
    except requests.exceptions.SSLError:
        response = requests.post(MISTRAL_ENDPOINT, headers=headers, json=payload, verify=False, timeout=30)
        response.raise_for_status()

    raw = response.json()["choices"][0]["message"]["content"]
    return clean_raw_ocr(raw)


def extract_text_from_pdf(pdf_path):
    pages = convert_from_path(pdf_path)
    all_text = ""

    for i, page in enumerate(pages):
        img_path = f"page_{i+1}.png"
        page.save(img_path, "PNG")

        print(f"[+] Processing page {i+1}...")
        text = extract_text_from_image(img_path)

        # Separate pages by two blank lines
        all_text += f"\n\n{text}\n\n"
        os.remove(img_path)

    return clean_raw_ocr(all_text)


if __name__ == "__main__":
    save_file = input("choose \n0. upload question paper\n1. check student answer sheet\n2. upload reference paper\n3. upload book\n")
    file_path = input("Enter image or PDF path: ").strip('"').strip("'")

    if save_file == "0":
        OUTPUT_FILE = "OCR_questions.json"
    elif save_file == "1":
        OUTPUT_FILE = "OCR_student_answer.json"
    elif save_file == "2":
        OUTPUT_FILE = "OCR_reference_answer.json"
    elif save_file == "3":
        OUTPUT_FILE = "OCR_Book.json"
    else:
        raise ValueError("Invalid choice")

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    if file_path.lower().endswith(".pdf"):
        result = extract_text_from_pdf(file_path)
    else:
        result = extract_text_from_image(file_path)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"extracted_text": result}, f, indent=4, ensure_ascii=False)

    print(f"\nOCR text saved to {OUTPUT_FILE}\n")
    print(result)
