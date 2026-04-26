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
def compute_embeddings_weighted(data):
    char_lines = {}
    char_weights = {}

    for item in data:
        char = item["speaker"]

        if char not in char_lines:
            char_lines[char] = []
            char_weights[char] = []

        char_lines[char].append(item["text"])
        char_weights[char].append(item.get("weight", 1.0))

    embeddings = {}

    for char in char_lines:
        embs = model.encode(char_lines[char])
        weights = np.array(char_weights[char])

        weighted_avg = np.average(embs, axis=0, weights=weights)
        embeddings[char] = weighted_avg

    return embeddings
if __name__ == "__main__":
    data = load_data("data/processed/dialogues.jsonl")
    grouped = group_by_character(data)
    embeddings = compute_embeddings(grouped)

    with open("models/embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
