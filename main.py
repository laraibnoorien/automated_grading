#!/usr/bin/env python3
"""
orchestrator_full_pipeline.py

End-to-end runner for your auto-grader.

Usage:
    python orchestrator_full_pipeline.py <student_input>

student_input may be:
 - a PDF or image containing student answer(s)
 - a .txt file containing student answer text
 - a pre-parsed JSON: regrex_student_answer.json (list of answers)

Outputs:
 - final_marks.json
 - marks_report.pdf

Notes:
 - This script will use existing FAISS index if present.
 - If FAISS index is missing, it will attempt to build it from regrex_book.json (or OCR_Book.json).
 - OCR uses pytesseract/pdf2image if available; otherwise you must run your OCR script separately first.
"""

import sys
import os
import json
from pathlib import Path
import shutil
import tempfile
from typing import Optional

# File names used in your pipeline
OCR_STUDENT_JSON = "OCR_student_answer.json"
REGEX_STUDENT_JSON = "regrex_student_answer.json"
OCR_BOOK_JSON = "OCR_Book.json"
REGEX_BOOK_JSON = "regrex_book.json"
FAISS_INDEX = "faiss_index.index"
FAISS_META = "faiss_metadata.json"

# Outputs
NLI_OUT = "nli_output.json"
CROSS_OUT = "cross_rerank_output.json"
FINAL_MARKS = "final_marks.json"
PDF_REPORT = "marks_report.pdf"

# Try to import pipeline modules (these should be in same folder)
def import_or_none(name):
    try:
        return __import__(name)
    except Exception:
        return None

regrex_answer = import_or_none("regrex_answer")
regrex_book = import_or_none("regrex_book")
local_embed_faiss = import_or_none("local_embed_faiss")
local_faiss_search = import_or_none("local_faiss_search") or import_or_none("faiss_search_student_batch") or import_or_none("local_faiss_search")
nli_filter = import_or_none("nli_filter")
cross_rerank = import_or_none("cross_rerank")
marks_generator = import_or_none("marks_generator")

# PDF generation
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# OCR helpers (try to use pdf2image+pytesseract fallback)
OCR_AVAILABLE = False
try:
    from PIL import Image
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

def log(s):
    print(s)

# -------------------------
# Step 0: Input handling
# -------------------------
def ocr_file_to_json(input_path: str, out_json: str = OCR_STUDENT_JSON) -> bool:
    """
    Try to OCR image/pdf -> write {"extracted_text": text} to out_json
    Returns True if written, False otherwise.
    """
    p = Path(input_path)
    if not p.exists():
        log(f"❌ Input file not found: {input_path}")
        return False

    # If input is already a JSON regrex_student file, copy
    if p.suffix.lower() == ".json":
        # assume user provided already parsed regrex_student_answer.json
        shutil.copy(input_path, REGEX_STUDENT_JSON)
        log(f"Copied pre-parsed JSON to {REGEX_STUDENT_JSON}")
        return True

    # If it's plain text, wrap into OCR json
    if p.suffix.lower() in [".txt"]:
        text = p.read_text(encoding="utf-8")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"extracted_text": text}, f, indent=2, ensure_ascii=False)
        log(f"Saved text to {out_json}")
        return True

    # If it's image/pdf, attempt OCR with pdf2image + pytesseract
    if not OCR_AVAILABLE:
        log("⚠️ OCR libs (pytesseract/pdf2image) not available locally.")
        log("Please run your OCR script to produce OCR_student_answer.json, or install pytesseract & pdf2image.")
        return False

    extracted = ""
    if p.suffix.lower() == ".pdf":
        log("Converting PDF to images for OCR (pdf2image)...")
        try:
            pages = convert_from_path(str(p))
            for i, page in enumerate(pages):
                text = pytesseract.image_to_string(page)
                extracted += f"\n\n--- Page {i+1} ---\n" + text
        except Exception as e:
            log("❌ PDF -> image OCR failed:", e)
            return False
    else:
        # single image
        try:
            img = Image.open(str(p))
            text = pytesseract.image_to_string(img)
            extracted = text
        except Exception as e:
            log("❌ Image OCR failed:", e)
            return False

    if not extracted.strip():
        log("⚠️ OCR produced empty text.")
        return False

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"extracted_text": extracted}, f, indent=2, ensure_ascii=False)
    log(f"[+] OCR output saved to {out_json}")
    return True

# -------------------------
# Step 1: Produce regrex_student_answer.json using regrex_answer.parse_file
# -------------------------
def ensure_parsed_student(from_input: str) -> bool:
    """
    If user provided a regrex_student_answer.json already, just use it.
    Else if they provided OCR json output (OCR_STUDENT_JSON) run regrex_answer.parse_file to produce regrex_student_answer.json.
    If they provided a pdf/image -> run OCR then parse.
    """
    # If regrex file exists, done
    if Path(REGEX_STUDENT_JSON).exists():
        log(f"[OK] Found existing {REGEX_STUDENT_JSON}")
        return True

    inp = Path(from_input)
    # if they passed a regrex format json directly
    if inp.suffix.lower() == ".json":
        # assume it's OCR JSON or regrex student json; try detect 'extracted_text'
        try:
            data = json.load(open(str(inp), "r"))
            if isinstance(data, list):
                # maybe already regrex_student_answer format — copy to expected path
                shutil.copy(str(inp), REGEX_STUDENT_JSON)
                log(f"[OK] Copied provided JSON to {REGEX_STUDENT_JSON}")
                return True
            if "extracted_text" in data:
                shutil.copy(str(inp), OCR_STUDENT_JSON)
                # Use regrex_answer.parse_file
            else:
                # unknown JSON format — copy and try parse
                shutil.copy(str(inp), OCR_STUDENT_JSON)
        except Exception as e:
            log("⚠️ Could not read provided JSON:", e)
            return False

    # If user provided pdf/image/txt -> attempt OCR
    if inp.suffix.lower() in [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".txt"]:
        ok = ocr_file_to_json(str(inp))
        if not ok:
            return False
    else:
        log("⚠️ Unsupported input type. Provide PDF/image/txt or pre-parsed JSON.")
        return False

    # now run regrex_answer.parse_file
    if regrex_answer is None:
        log("❌ regrex_answer.py not found in project. Please ensure regrex_answer.py is available.")
        return False

    try:
        # regrex_answer.parse_file(in_path, out_path)
        regrex_answer.parse_file(OCR_STUDENT_JSON, REGEX_STUDENT_JSON)
        log(f"[+] Parsed student answers → {REGEX_STUDENT_JSON}")
        return True
    except Exception as e:
        log("❌ regrex_answer.parse_file failed:", e)
        return False

# -------------------------
# Step 2: Ensure regrex_book.json exists (or parse OCR_Book.json)
# -------------------------
def ensure_book_parsed():
    if Path(REGEX_BOOK_JSON).exists():
        log(f"[OK] Found {REGEX_BOOK_JSON}")
        return True

    if regrex_book is None:
        log("⚠️ regrex_book.py not found. If you already have regrex_book.json, place it in project root.")
        return False

    if not Path(OCR_BOOK_JSON).exists():
        log("⚠️ OCR_Book.json not found. Provide OCR_Book.json for automatic book parsing, or produce regrex_book.json manually.")
        return False

    try:
        regrex_book.parse_book_ocr(OCR_BOOK_JSON, REGEX_BOOK_JSON)
        log(f"[+] Parsed OCR_Book.json -> {REGEX_BOOK_JSON}")
        return True
    except Exception as e:
        log("❌ regrex_book.parse_book_ocr failed:", e)
        return False

# -------------------------
# Step 3: Ensure FAISS index exists (use existing if available)
# -------------------------
def ensure_faiss_index(use_existing=True) -> bool:
    # If index exists and metadata exists, accept it
    if Path(FAISS_INDEX).exists() and Path(FAISS_META).exists():
        log("[OK] Found existing FAISS index + metadata.")
        return True

    if use_existing:
        log("[INFO] FAISS not found. Will attempt to create FAISS index from regrex_book.json (book only).")

    # Need local_embed_faiss to build index programmatically
    if local_embed_faiss is None:
        log("❌ local_embed_faiss.py not found. Cannot build FAISS programmatically.")
        return False

    if not Path(REGEX_BOOK_JSON).exists():
        log("❌ regrex_book.json not found. Cannot build FAISS index.")
        return False

    # load book chunks and add to faiss
    try:
        chunks = local_embed_faiss.load_book_chunks(REGEX_BOOK_JSON)
        if not chunks:
            log("⚠️ No book chunks found to index.")
            return False
        local_embed_faiss.add_to_faiss(chunks, role="passage", file=REGEX_BOOK_JSON)
        log("[+] Built FAISS index from book chunks.")
        return True
    except Exception as e:
        log("❌ Failed to build FAISS index:", e)
        return False

# -------------------------
# Step 4: Run FAISS search for all student answers
# -------------------------
def run_faiss_search_batch():
    if local_faiss_search is None:
        log("❌ local_faiss_search or batch search module not available.")
        return False

    # local_faiss_search.run_batch will print results — we want to produce intermediate files for next stages
    # But we rely on nli_filter.py which runs FAISS internally per student; so just ensure regrex_student exists
    log("[*] Running FAISS -> NLI -> CrossEncoder -> Marks pipeline via individual modules.")
    return True

# -------------------------
# Step 5..7: NLI, Cross-rerank, Marks
# -------------------------
def run_nli_cross_marks():
    # NLI
    if nli_filter is None:
        log("❌ nli_filter.py not present.")
        return False
    try:
        nli_filter.run_nli_for_all(top_k=5)
    except Exception as e:
        log("❌ nli_filter.run_nli_for_all failed:", e)
        return False

    # Cross-encoder
    if cross_rerank is None:
        log("❌ cross_rerank.py not present.")
        return False
    try:
        cross_rerank.run_cross_encoder()
    except Exception as e:
        log("❌ cross_rerank.run_cross_encoder failed:", e)
        return False

    # Marks
    if marks_generator is None:
        log("❌ marks_generator.py not present.")
        return False
    try:
        marks_generator.generate_marks()
    except Exception as e:
        log("❌ marks_generator.generate_marks failed:", e)
        return False

    return True

# -------------------------
# Step 8: Create PDF marksheet
# -------------------------
def create_pdf_from_final_marks(pdf_path: str = PDF_REPORT, json_path: str = FINAL_MARKS):
    if not Path(json_path).exists():
        log(f"❌ {json_path} not found. Cannot create PDF.")
        return False

    final = json.load(open(json_path, "r", encoding="utf-8"))

    if not REPORTLAB_AVAILABLE:
        # fallback: save a human-readable text file next to JSON and inform user
        txt_path = Path("marks_report.txt")
        with txt_path.open("w", encoding="utf-8") as f:
            for item in final:
                f.write(f"Question: {item.get('question_number')}\n")
                f.write(f"Student Answer: {item.get('student_answer')}\n")
                f.write(f"Best Passage: {item.get('best_passage_text','')}\n")
                f.write(f"Semantic: {item.get('semantic_score')} | Keyword: {item.get('keyword_score')} | Final: {item.get('final_score')}\n")
                f.write("-"*60 + "\n")
        log(f"[!] reportlab not installed — saved plain-text report to {txt_path}")
        return True

    # Create PDF
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    margin = 20 * mm
    x = margin
    y = height - margin

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, "Marks Report")
    y -= 12 * mm

    c.setFont("Helvetica", 10)

    for item in final:
        if y < margin + 40:
            c.showPage()
            y = height - margin
            c.setFont("Helvetica", 10)

        qnum = item.get("question_number", "")
        student_answer = item.get("student_answer", "")
        best = item.get("best_passage_text", "")
        semantic = item.get("semantic_score", "")
        keyword = item.get("keyword_score", "")
        final_score = item.get("final_score", "")

        c.setFont("Helvetica-Bold", 11)
        c.drawString(x, y, f"Question: {qnum} — Score: {final_score}/100")
        y -= 6 * mm
        c.setFont("Helvetica", 9)
        # Student answer (trim and wrap)
        lines = split_text_lines(student_answer, max_chars=100)
        c.drawString(x, y, "Student answer:")
        y -= 4 * mm
        for ln in lines:
            c.drawString(x + 6 * mm, y, ln)
            y -= 4 * mm

        y -= 2 * mm
        c.drawString(x, y, f"Best matched passage (excerpt):")
        y -= 4 * mm
        for ln in split_text_lines(best, max_chars=100):
            c.drawString(x + 6 * mm, y, ln)
            y -= 4 * mm

        y -= 2 * mm
        c.drawString(x, y, f"Semantic: {semantic} | Keyword: {keyword}")
        y -= 8 * mm

    c.save()
    log(f"[+] PDF report saved to {pdf_path}")
    return True

def split_text_lines(text: str, max_chars=90):
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        if len(cur) + 1 + len(w) > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines

# -------------------------
# Orchestrator main
# -------------------------
def orchestrate(student_input_path: str):
    log("=== START: Full pipeline orchestrator ===")

    # Step 1: parse student
    ok = ensure_parsed_student(student_input_path)
    if not ok:
        log("❌ Aborting: cannot parse student answers.")
        return

    # Step 2: ensure book parsed (if not present we try)
    book_ok = ensure_book_parsed()
    if not book_ok:
        log("⚠️ regrex_book.json not present; continuing (FAISS build will fail if index is missing).")

    # Step 3: ensure FAISS index exists (use existing if available)
    faiss_ok = ensure_faiss_index(use_existing=True)
    if not faiss_ok:
        log("❌ Aborting: FAISS index not available and could not be built.")
        return

    # Step 4..7: run NLI + cross rerank + marks
    pipeline_ok = run_nli_cross_marks()
    if not pipeline_ok:
        log("❌ Aborting: pipeline (NLI/Cross/Marks) failed.")
        return

    # Step 8: create PDF
    pdf_ok = create_pdf_from_final_marks()
    if not pdf_ok:
        log("⚠️ PDF generation failed (or reportlab missing). Check final_marks.json")
    else:
        log(f"[✔] Orchestration complete. PDF at {PDF_REPORT}")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python orchestrator_full_pipeline.py <student_input (pdf|png|jpg|txt|regrex_student_answer.json)>")
        sys.exit(1)

    path = sys.argv[1]
    orchestrate(path)
