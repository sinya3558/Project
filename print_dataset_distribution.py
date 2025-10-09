import os
import json
from tqdm import tqdm

ann_dir = "/mnt/c/Users/sinya/OneDrive/Documents/GitHub/Project/data/annotations"

success = 0
fail = 0

json_files = []
for root, _, files in os.walk(ann_dir):
    for f in files:
        if f.lower().endswith(".json"):
            json_files.append(os.path.join(root, f))

print(f"Num of JSON files: {len(json_files)}")

for json_path in tqdm(json_files, desc="Processing JSONs"):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
        if ann.get("grip_succeed", 0) == 1:
            success += 1
        else:
            fail += 1
    except Exception as e:
        print(f"{json_path} failed to read files : {e}")
        continue

print("Success samples:", success)
print("Fail samples:", fail)
