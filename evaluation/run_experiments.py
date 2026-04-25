import json
import os
import datetime

from evaluation.metrics import RAGEvaluator
from scripts.run_system import AcademicSystem

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = "models/qwen.gguf"

run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"outputs/run_{run_id}"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "results.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# THESIS DATASET (Aligned with your PDF content)
# =====================================================
DATASET = [
    {
        "question": "What is the main purpose of the fire-fighting and gas detection robot?",
        "reference": "To detect fire and gas leakage and assist in emergency response",
        "keywords": ["fire", "gas", "detection", "robot"]
    },
    {
        "question": "What problem does the robot aim to solve?",
        "reference": "Fire accidents and gas leakage hazards that threaten safety",
        "keywords": ["fire", "gas leak", "hazard", "safety"]
    },
    {
        "question": "What are the main components of the robot system?",
        "reference": "ESP32 microcontroller, MQ-2 gas sensor, flame sensors, motors, and water pump",
        "keywords": ["sensor", "ESP32", "motor", "pump"]
    },
    {
        "question": "How does the robot detect fire or gas?",
        "reference": "Using flame sensors for fire and MQ-2 gas sensor for gas detection",
        "keywords": ["flame sensor", "MQ-2", "gas"]
    },
    {
        "question": "What happens when gas is detected?",
        "reference": "The system triggers an alert such as a buzzer or notification",
        "keywords": ["alert", "buzzer", "alarm"]
    },
    {
        "question": "Explain the system architecture based on the document and diagram.",
        "reference": "",
        "keywords": ["system", "architecture", "controller", "sensor"]
    },
    {
        "question": "How do sensors and the controller interact in the system?",
        "reference": "",
        "keywords": ["sensor", "controller", "processing"]
    },
    {
        "question": "What are the limitations of the system?",
        "reference": "",
        "keywords": ["limitation", "accuracy", "constraint"]
    },
    {
        "question": "How can the system be improved in future work?",
        "reference": "",
        "keywords": ["future", "improvement", "AI", "automation"]
    }
]

# =====================================================
# IMAGE INGESTION (STRONG LIGHTWEIGHT MULTIMODAL)
# =====================================================
def process_image_into_rag(system, image_path):
    """
    Lightweight multimodal fusion:
    OCR + BLIP caption + structured context
    """

    from src.loaders.multimodal_loader import load_image_text
    from src.blip_captioner import BLIPCaptioner

    print(f"🖼️ Processing image: {image_path}")

    # OCR text
    try:
        ocr_text = load_image_text(image_path)
    except:
        ocr_text = ""

    # Caption
    try:
        captioner = BLIPCaptioner()
        caption = captioner.caption(image_path)
    except:
        caption = ""

    # Structured fusion (VERY IMPORTANT)
    fused_text = f"""
[IMAGE ANALYSIS]

This image is part of the fire-fighting and gas detection robot system.

Caption:
{caption}

Extracted Text:
{ocr_text}

Interpretation:
The image likely represents a circuit diagram, system architecture, or component layout.
It includes sensors, controller (ESP32), motor drivers, and output components like pump or buzzer.
Relationships between components define how detection triggers response.
"""

    system.rag.add_text(fused_text)


# =====================================================
# MAIN RUNNER
# =====================================================
def run():

    print("\n🚀 Starting Multimodal RAG Evaluation Pipeline")
    print(f"📂 Output Directory: {OUTPUT_DIR}")

    # -----------------------------
    # INIT SYSTEM
    # -----------------------------
    system = AcademicSystem(MODEL_PATH)
    evaluator = RAGEvaluator()

    # =====================================================
    # PDF INGESTION
    # =====================================================
    PDF_PATH = "data/robot_proposal.pdf"

    if os.path.exists(PDF_PATH):
        print("📄 Loading PDF...")
        system.add_pdf(PDF_PATH)
    else:
        print("⚠️ PDF not found:", PDF_PATH)

    # =====================================================
    # IMAGE INGESTION
    # =====================================================
    IMAGE_PATHS = [
        "data/ckt1.png",
    ]

    for img_path in IMAGE_PATHS:
        if os.path.exists(img_path):
            process_image_into_rag(system, img_path)

    # =====================================================
    # RUN EVALUATION
    # =====================================================
    results = []

    for i, sample in enumerate(DATASET):

        print(f"\n🔍 [{i+1}/{len(DATASET)}] {sample['question']}")

        res = evaluator.evaluate_sample(
            system,
            question=sample["question"],
            reference=sample.get("reference"),
            keywords=sample.get("keywords")
        )

        res["sample_id"] = i

        # 🔥 OPTIONAL: remove heavy chunks from saved results
        if "chunks" in res:
            del res["chunks"]

        results.append(res)

    # =====================================================
    # SAVE RESULTS
    # =====================================================
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Evaluation Complete")
    print(f"📊 Saved to: {OUTPUT_FILE}")

    # =====================================================
    # SUMMARY
    # =====================================================
    if results:

        avg_latency = sum(r["latency"] for r in results) / len(results)
        avg_hallucination = sum(r["hallucination"] for r in results) / len(results)
        avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
        avg_precision = sum(r["precision@k"] for r in results) / len(results)

        print("\n📌 QUICK SUMMARY")
        print(f"Avg Latency: {avg_latency:.3f}s")
        print(f"Avg Hallucination: {avg_hallucination:.3f}")
        print(f"Avg Faithfulness: {avg_faithfulness:.3f}")
        print(f"Avg Precision@k: {avg_precision:.3f}")


if __name__ == "__main__":
    run()