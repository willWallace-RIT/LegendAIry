import pickle
import numpy as np

class PersonaEngine:
    def __init__(self, embedding_path):
        with open(embedding_path, "rb") as f:
            self.embeddings = pickle.load(f)

    def blend_personas(self, char1, char2, alpha=0.5):
        emb1 = self.embeddings[char1]
        emb2 = self.embeddings[char2]

        return alpha * emb1 + (1 - alpha) * emb2

    def describe_style(self, vector):
        # crude interpretation layer
        return f"Confident, manipulative, witty, but emotionally aware"
