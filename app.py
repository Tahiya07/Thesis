from scripts.run_system import AcademicSystem
from src.summarizer import summarize


system = AcademicSystem("models/qwen.gguf")


while True:
    print("\n===== Academic Assistant =====")
    print("1. Load PDF")
    print("2. Load Website")
    print("3. Load Image")
    print("4. Ask Question")
    print("5. Summarize Text")
    print("6. Exit")

    choice = input("Choose: ").strip()

    if choice == "1":
        path = input("Enter PDF path: ").strip()
        system.add_pdf(path)
        print("Indexed PDF into the knowledge base.")

    elif choice == "2":
        url = input("Enter URL: ").strip()
        system.add_url(url)
        print("Indexed webpage into the knowledge base.")

    elif choice == "3":
        path = input("Enter image path: ").strip()
        system.add_image(path)
        print("Indexed image content into the knowledge base.")

    elif choice == "4":
        question = input("Question: ").strip()
        result = system.ask(question)
        print("\n===== ANSWER =====\n")
        print(result["answer"])
        print(f"\nContext Used: {result['context_used']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Uncertain: {result['uncertain']}")

    elif choice == "5":
        text = input("Enter text: ").strip()
        llm = system.get_llm()
        print("\nSummary:")
        print(summarize(llm, text))

    elif choice == "6":
        print("Exiting...")
        break

    else:
        print("Invalid choice. Try again.")
