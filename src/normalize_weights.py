import json
import numpy as np

def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]

def save(data, path):
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

def normalize(data):
    scores = np.array([d["score"] for d in data])

    min_s = scores.min()
    max_s = scores.max()

    for d in data:
        d["weight"] = (d["score"] - min_s) / (max_s - min_s + 1e-6)

    return data


if __name__ == "__main__":
    data = load("data/processed/scored_fan_dialogues.jsonl")
    data = normalize(data)
    save(data, "data/processed/weighted_fan_dialogues.jsonl")
