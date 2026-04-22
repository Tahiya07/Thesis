import tiktoken

# fallback tokenizer (lightweight approximation for GGUF models)
# we approximate: 1 token ≈ 0.75 words

class TokenManager:
    def __init__(self, model_context_size=2304):
        self.max_tokens = model_context_size
        self.safety_margin = 0.8  # keep 20% buffer

    def estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) / 0.75)

    def truncate_to_fit(self, text: str, max_tokens: int) -> str:
        words = text.split()
        max_words = int(max_tokens * 0.75)
        return " ".join(words[:max_words])

    def build_safe_context(self, chunks, question: str):
        """
        Dynamically selects how many chunks can fit safely.
        """

        question_tokens = self.estimate_tokens(question)

        available_tokens = int(
            self.max_tokens * self.safety_margin - question_tokens - 200
        )

        context = []
        used_tokens = 0

        for chunk in chunks:
            chunk_tokens = self.estimate_tokens(chunk)

            if used_tokens + chunk_tokens > available_tokens:
                break

            context.append(chunk)
            used_tokens += chunk_tokens

        final_context = "\n".join(context)

        # final safety trim
        return self.truncate_to_fit(final_context, available_tokens)