# src/blip_captioner.py

from PIL import Image


class BLIPCaptioner:
    """
    Fully lazy-loaded BLIP model.
    No model is loaded until caption() is called.
    """

    def __init__(self):
        self.processor = None
        self.model = None
        self.device = "cpu"

    def _load(self):
        """
        Lazy initialization of BLIP model.
        """
        if self.model is None:
            print("🖼️ Loading BLIP model (lazy)...")

            from transformers import BlipProcessor, BlipForConditionalGeneration

            self.processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )

            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )

            self.model.to(self.device)

    def caption(self, image_path: str) -> str:
        """
        Generate caption for image.
        """

        try:
            self._load()

            image = Image.open(image_path).convert("RGB")

            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            output = self.model.generate(**inputs, max_new_tokens=50)

            return self.processor.decode(output[0], skip_special_tokens=True)

        except Exception as e:
            print(f"[BLIP ERROR] {e}")
            return "Image could not be processed."