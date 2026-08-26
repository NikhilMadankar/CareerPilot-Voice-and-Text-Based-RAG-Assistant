import os
import fitz  # PyMuPDF


def pdf_to_text(pdf_path, txt_path):
    doc = fitz.open(pdf_path)
    text = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"✅ Converted '{pdf_path}' to '{txt_path}' successfully.")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    pdf_path = os.path.join(base_dir, "different careers by ncert .pdf")
    txt_path = os.path.join(base_dir, "career_guide.txt")

    if os.path.exists(pdf_path):
        pdf_to_text(pdf_path, txt_path)
    else:
        print(f"⚠️ PDF file not found at: {pdf_path}")
