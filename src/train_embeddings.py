import json
from sentence_transformers import SentenceTransformer
import pickle

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_data(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def group_by_character(data):
    char_lines = {}
    for item in data:
        char = item["speaker"]
        if char not in char_lines:
            char_lines[char] = []
        char_lines[char].append(item["text"])
    return char_lines

def compute_embeddings(char_lines):
    embeddings = {}
    for char, lines in char_lines.items():
        emb = model.encode(lines)
        embeddings[char] = emb.mean(axis=0)
    return embeddings

if __name__ == "__main__":
    data = load_data("data/processed/dialogues.jsonl")
    grouped = group_by_character(data)
    embeddings = compute_embeddings(grouped)

    with open("models/embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
