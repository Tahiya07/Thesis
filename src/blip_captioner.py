#src/blip_captioner.py

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

class BLIPCaptioner:
    def __init__(self):
        print("🖼️ Loading BLIP image captioning model...")

        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

        self.device = "cpu"
        self.model.to(self.device)

    def caption(self, image_path: str) -> str:
        try:
            image = Image.open(image_path).convert("RGB")

            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            out = self.model.generate(**inputs, max_new_tokens=50)

            caption = self.processor.decode(out[0], skip_special_tokens=True)

            return caption

        except Exception as e:
            print(f"[IMAGE ERROR] {e}")
            return "Image could not be processed."