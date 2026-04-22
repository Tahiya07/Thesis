#src/loaders/multimodal_loader.py

from PIL import Image
import pytesseract

def load_image_text(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text