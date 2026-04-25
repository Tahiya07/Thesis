from PIL import Image


_processor = None
_model = None
_device = "cpu"


def _load_blip():
    global _processor, _model
    if _processor is not None and _model is not None:
        return _processor, _model

    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        _processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            local_files_only=True
        )
        _model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            local_files_only=True
        )
        _model.to(_device)
        return _processor, _model
    except Exception:
        _processor = None
        _model = None
        return None, None


def load_image_caption(image_path):
    processor, model = _load_blip()
    if processor is None or model is None:
        return "Image content unavailable from local caption model."

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(_device)
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
