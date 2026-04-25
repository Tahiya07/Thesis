import numpy as np

class BloomLDL:

    levels = [
        "Remember",
        "Understand",
        "Apply",
        "Analyze",
        "Evaluate",
        "Create"
    ]

    def predict_distribution(self, text):

        text = text.lower()

        dist = np.ones(len(self.levels)) * 0.1

        if "define" in text or "what is" in text:
            dist[0] += 0.6

        if "explain" in text:
            dist[1] += 0.5

        if "solve" in text or "calculate" in text:
            dist[2] += 0.5

        if "analyze" in text:
            dist[3] += 0.5

        if "compare" in text:
            dist[4] += 0.5

        dist = dist / dist.sum()

        return {
            "distribution": dist,
            "level": self.levels[np.argmax(dist)]
        }