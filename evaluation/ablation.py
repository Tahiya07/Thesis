from scripts.run_system import AcademicSystem
from src.metrics import RAGEvaluator

def run_ablation():

    evaluator = RAGEvaluator()

    configs = {
        "full_rag": {"use_rag": True},
        "no_rag": {"use_rag": False}
    }

    question = "Who is the head of the department?"

    for name, cfg in configs.items():

        system = AcademicSystem("models/qwen.gguf")

        print(f"\n🔬 Running: {name}")

        result = evaluator.evaluate_sample(
            system,
            question=question
        )

        for k, v in result.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")