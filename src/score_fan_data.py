import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_embeddings():
    with open("models/embeddings.pkl", "rb") as f:
        return pickle.load(f)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def score_line(text, character, ref_embedding):
    emb = model.encode(text)

    similarity = cosine_similarity(emb, ref_embedding)

    # crude penalties
    exposition_penalty = 0.0
    if len(text.split()) > 30:
        exposition_penalty += 0.1

    if "as you know" in text.lower():
        exposition_penalty += 0.2

    score = similarity - exposition_penalty
    return float(score)


def process():
    embeddings = load_embeddings()

    with open("data/processed/fan_dialogues.jsonl") as f:
        lines = [json.loads(l) for l in f]

    scored = []

    for item in lines:
        char = item["speaker"]

        if char not in embeddings:
            continue

        score = score_line(item["text"], char, embeddings[char])

        item["score"] = score
        scored.append(item)

    with open("data/processed/scored_fan_dialogues.jsonl", "w") as f:
        for item in scored:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    process()
