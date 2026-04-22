from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL = "Qwen/Qwen2-0.5B"


def load_bloom_model(path="finetune/lora_model"):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32
    )

    model = PeftModel.from_pretrained(base, path)
    model.eval()

    return model, tokenizer


def classify_bloom(model, tokenizer, question):
    prompt = f"""
Classify Bloom level.

Question: {question}

Answer ONLY one word:
Remember, Understand, Apply, Analyze, Evaluate, Create
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    output = model.generate(
        **inputs,
        max_new_tokens=5,
        temperature=0
    )

    result = tokenizer.decode(output[0], skip_special_tokens=True)

    # CLEAN OUTPUT
    for label in ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]:
        if label in result:
            return label

    return "Understand"  # fallback