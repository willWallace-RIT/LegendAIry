import json

def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]

def save(data, path):
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

canon = load("data/processed/dialogues.jsonl")
fan = load("data/processed/weighted_fan_dialogues.jsonl")

for c in canon:
    c["weight"] = 1.0
    c["source"] = "canon"

merged = canon + fan

save(merged, "data/processed/final_dataset.jsonl")
