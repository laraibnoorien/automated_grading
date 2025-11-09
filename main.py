import os
import json
from OCR_mistral import extract_text_from_pdf, extract_text_from_image
from QARegrex import parse_ocr_text_to_qa_unified
from Sent_embedding import embed_file
from evaluation import hybrid_check


def ocr_stage():
    print("\n=== OCR STAGE ===")
    save_file = input(
        "Choose:\n"
        " 1. Check student answer sheet\n"
        " 2. Upload reference paper\n"
        " 3. Upload book\n"
        " 4. Both 1 and 2\n"
        " 5. Both 1 and 3\n> "
    ).strip()

    mapping = {
        "1": ["OCR_student_answer.json"],
        "2": ["OCR_reference_answer.json"],
        "3": ["OCR_Book.json"],
        "4": ["OCR_student_answer.json", "OCR_reference_answer.json"],
        "5": ["OCR_student_answer.json", "OCR_Book.json"],
    }

    selected_files = mapping.get(save_file)
    if not selected_files:
        print("[-] Invalid choice. Exiting.")
        exit(1)

    for OUTPUT_FILE in selected_files:
        file_path = input(f"Enter image or PDF path for {OUTPUT_FILE.replace('OCR_', '').replace('.json', '')}: ").strip().strip('"').strip("'")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.lower().endswith(".pdf"):
            result = extract_text_from_pdf(file_path)
        else:
            result = extract_text_from_image(file_path)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump({"extracted_text": result}, f, indent=4, ensure_ascii=False)

        print(f"[+] OCR text saved to {OUTPUT_FILE}\n")

    return selected_files



def chunking_stage():
    print("\n=== CHUNKING / REGEX STAGE ===")
    to_embed = input(
        "Choose: \n 1. student_answer \n 2. reference_answer \n 3. book_reference \n 4. both 1 and 2 \n 5. both 1 and 3 \n> "
    ).strip()

    options = {
        "1": [("OCR_student_answer.json", "regrex_student_answer.json")],
        "2": [("OCR_reference_answer.json", "regrex_reference_answer.json")],
        "3": [("OCR_Book.json", "regrex_Book.json")],
        "4": [
            ("OCR_student_answer.json", "regrex_student_answer.json"),
            ("OCR_reference_answer.json", "regrex_reference_answer.json")
        ],
        "5": [
            ("OCR_student_answer.json", "regrex_student_answer.json"),
            ("OCR_Book.json", "regrex_Book.json")
        ]
    }

    selected = options.get(to_embed)
    if not selected:
        print("[-] Invalid choice. Exiting.")
        exit(1)

    for src, dst in selected:
        if not os.path.exists(src):
            print(f"[-] Missing source file: {src}")
            continue
        parse_ocr_text_to_qa_unified(src, dst)
        print(f"[+] Parsed {src} → {dst}")

    return [dst for _, dst in selected]


def embedding_stage():
    print("\n=== EMBEDDING STAGE ===")
    SUBJECT = input("Enter subject: ").strip()
    BOARD = input("Enter board: ").strip()
    CLASS = input("Enter class: ").strip()

    FILES = {
        "regrex_student_answer": "regrex_student_answer.json",
        "regrex_reference_answer": "regrex_reference_answer.json",
        "regrex_book": "regrex_Book.json",
    }

    print("\nSelect what to embed:")
    print("1 → regrex_student_answer")
    print("2 → regrex_reference_answer")
    print("3 → regrex_book")
    print("4 → both 1 and 2")
    print("5 → both 1 and 3")

    choice = input("\nEnter your choice (1-5): ").strip()
    mapping = {
        "1": ["regrex_student_answer"],
        "2": ["regrex_reference_answer"],
        "3": ["regrex_book"],
        "4": ["regrex_student_answer", "regrex_reference_answer"],
        "5": ["regrex_student_answer", "regrex_book"],
    }

    selected = mapping.get(choice)
    if not selected:
        print("[-] Invalid choice. Exiting.")
        exit(1)

    for key in selected:
        path = FILES.get(key)
        if os.path.exists(path):
            embed_file(path, SUBJECT, BOARD, CLASS, key)
        else:
            print(f"[-] Missing file: {path}")


def evaluation_stage():
    print("\n=== HYBRID EVALUATION STAGE ===")
    print("[*] Performing semantic + NLI grading...")
    hybrid_check()
    print("[+] Grading completed successfully!\n")


if __name__ == "__main__":
    print("\n========= AUTOMATED GRADING SYSTEM =========")

    while True:
        print("\nMain Menu:")
        print("1 → OCR & Extraction")
        print("2 → Chunking / Regex Parsing")
        print("3 → Embedding")
        print("4 → Hybrid Evaluation (Semantic + NLI)")
        print("5 → Run Full Pipeline")
        print("0 → Exit")

        choice = input("\nSelect your option: ").strip()

        if choice == "1":
            ocr_stage()
        elif choice == "2":
            chunking_stage()
        elif choice == "3":
            embedding_stage()
        elif choice == "4":
            evaluation_stage()
        elif choice == "5":
            ocr_stage()
            chunking_stage()
            embedding_stage()
            evaluation_stage()
        elif choice == "0":
            print("Exiting...")
            break
        else:
            print("[-] Invalid choice, try again.")
