def load_pdf_text(path):
    try:
        from pypdf import PdfReader

        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception:
        pass

    try:
        import fitz

        text = ""
        doc = fitz.open(path)
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text
    except Exception:
        return ""
