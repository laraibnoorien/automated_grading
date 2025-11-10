import os
import base64
import requests
from PIL import Image
from pdf2image import convert_from_path
from dotenv import load_dotenv
load_dotenv()
import json


MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"
MODEL = "pixtral-large-latest"  

def encode_image(image_path):
    """Encode image as base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extract_text_from_image(image_path):
    """Send an image to Mistral OCR model and return extracted text."""
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
                    {"type": "text", "text": "Extract all text accurately from this image."},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"}
                ]
            }
        ]
    }

    # Create TLS-secure session
    session = requests.Session()
    session.mount("https://", SSLAdapter())

    try:
        response = session.post(MISTRAL_ENDPOINT, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
    except requests.exceptions.SSLError as e:
        print("❌ SSL handshake failed:", e)
        print("⚙️ Retrying with SSL verification disabled...")
        response = requests.post(MISTRAL_ENDPOINT, headers=headers, json=payload, verify=False, timeout=30)
        response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


def extract_text_from_pdf(pdf_path):
    """Convert PDF pages to images and run OCR on each page."""
    pages = convert_from_path(pdf_path)
    all_text = ""
    for i, page in enumerate(pages):
        img_path = f"page_{i+1}.png"
        page.save(img_path, "PNG")
        print(f"[+] Processing page {i+1}...")
        text = extract_text_from_image(img_path)
        all_text += f"\n\n--- Page {i+1} ---\n{text}"
        os.remove(img_path)
    return all_text

"""if __name__ == "__main__":
    file_path = input("Enter image or PDF path: ").strip().strip('"').strip("'")
    if file_path.lower().endswith(".pdf"):
        result = extract_text_from_pdf(file_path)
    else:
        result = extract_text_from_image(file_path)
    
    print("\n========= EXTRACTED TEXT =========\n")
    print(result)
"""
#OUTPUT_FILE = "ocr_output.json"

if __name__ == "__main__":
    save_file= input("choose \n 0. upload question paper \n 1. check student answer sheet \n 2. upload reference paper\n 3. upload book  ")
    file_path = input("Enter image or PDF path: ").strip().strip('"').strip("'")
    if save_file=="0":
        OUTPUT_FILE = "OCR_questions.json"
    elif save_file=="1":
        OUTPUT_FILE = "OCR_student_answer.json"
    elif save_file=="2":
        OUTPUT_FILE = "OCR_reference_answer.json"
    elif save_file=="3":
        OUTPUT_FILE = "OCR_Book.json"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.lower().endswith(".pdf"):
        result = extract_text_from_pdf(file_path)
    else:
        result = extract_text_from_image(file_path)

    # Save to JSON for downstream evaluation
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"extracted_text": result}, f, indent=4, ensure_ascii=False)
    
    print(f"\nOCR text saved to {OUTPUT_FILE}\n")
    #print("\n========= EXTRACTED TEXT =========\n")
    print(result)