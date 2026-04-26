import json
import re
import os

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
                "text": dialogue,
                "source": "fan"
            })

    return data


def process_folder(folder_path):
    all_data = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            all_data.extend(parse_script(os.path.join(folder_path, file)))
    return all_data


def save_jsonl(data, path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    data = process_folder("data/raw/fan_scripts/")
    save_jsonl(data, "data/processed/fan_dialogues.jsonl")
