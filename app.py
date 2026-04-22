from src.llm import load_rag_model
from src.rag_engine import RAGEngine
from src.summarizer import summarize
from src.bloom import classify_bloom


# =========================
# INIT SYSTEM
# =========================

llm = load_rag_model("models/qwen.gguf")
rag = RAGEngine()


# =========================
# CLI LOOP
# =========================

while True:
    print("\n===== Academic Assistant =====")
    print("1. Load PDF")
    print("2. Load Website")
    print("3. Ask Question (RAG)")
    print("4. Summarize Text")
    print("5. Bloom Classification")
    print("6. Exit")

    choice = input("Choose: ")

    # -------------------------
    # PDF ingestion
    # -------------------------
    if choice == "1":
        path = input("Enter PDF path: ")
        rag.build_from_pdf(path)
        print("✅ PDF indexed successfully!")

    # -------------------------
    # Website ingestion
    # -------------------------
    elif choice == "2":
        url = input("Enter URL: ")
        rag.build_from_url(url)
        print("✅ Website indexed successfully!")

    # -------------------------
    # RAG QA
    # -------------------------
    elif choice == "3":
        q = input("Question: ")

        result = rag.ask(llm, q, bloom_classifier=classify_bloom)

        print("\n===== ANSWER =====\n")
        print(result["answer"])
        print("\nBloom Level:", result["bloom_level"])
        print("Used RAG:", result["used_rag"])

    # -------------------------
    # Summarization (manual test tool)
    # -------------------------
    elif choice == "4":
        t = input("Enter text: ")
        print("\nSummary:")
        print(summarize(llm, t))

    # -------------------------
    # Bloom classification only
    # -------------------------
    elif choice == "5":
        q = input("Enter question: ")
        print("\nBloom Level:")
        print(classify_bloom(q))

    # -------------------------
    # Exit
    # -------------------------
    elif choice == "6":
        print("Exiting...")
        break

    else:
        print("Invalid choice. Try again.")