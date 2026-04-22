#src/loaders/image_loader.py

from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def load_image_caption(image_path):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)

    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption