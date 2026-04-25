import pytesseract
from PIL import Image


def load_image_text(image_path: str) -> str:
    try:
        image = Image.open(image_path).convert("RGB")
        text = pytesseract.image_to_string(image)
        return " ".join((text or "").split())
    except Exception:
        return ""
