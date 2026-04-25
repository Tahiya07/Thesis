from scripts.run_system import AcademicSystem


system = AcademicSystem("models/qwen.gguf")
system.rag.add_text("Fire detection system uses sensors to detect flame and gas leakage.")
system.rag.add_text("ESP32 controls motors and sensors in robotic systems.")

question = "What is the role of ESP32 in the system?"
result = system.ask(question)

print("\n--- RESPONSE ---")
print(result["answer"])

print("\n--- CONFIDENCE ---")
print(result["confidence"])

print("\n--- TRACE ---")
for chunk in result["chunks"]:
    print(chunk["score"], chunk["text"][:120])
