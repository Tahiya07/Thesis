from src.rag_engine import RAGEngine
from src.llm import load_rag_model
from src.bloom_model import load_bloom_model, classify_bloom

# LLM (generation)
llm = load_rag_model("models/qwen.gguf")

# Bloom classifier
bloom_model, tokenizer = load_bloom_model()

# RAG
rag = RAGEngine()
rag.add_text("Fire detection system uses sensors to detect flame and gas leakage.")
rag.add_text("ESP32 controls motors and sensors in robotic systems.")
rag.build_index()

# TEST
question = "What is the role of ESP32 in the system?"

bloom_level = classify_bloom(bloom_model, tokenizer, question)

result = rag.ask(llm, bloom_model, question)

print("\n--- BLOOM LEVEL ---")
print(bloom_level)

print("\n--- RESPONSE ---")
print(result["response"])