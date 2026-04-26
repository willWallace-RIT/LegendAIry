import json
import re

def parse_script(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    lines = text.split("\n")
    data = []

    for line in lines:
        match = re.match(r"(\w+): (.+)", line)
        if match:
            speaker = match.group(1)
            dialogue = match.group(2)

            data.append({
                "speaker": speaker,
                "text": dialogue
            })

    return data


def save_jsonl(data, output_path):
    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    data = parse_script("data/raw/himym_scripts.txt")
    save_jsonl(data, "data/processed/dialogues.jsonl")
