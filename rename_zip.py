import os

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dirs = [
    os.path.join(base_dir, "Validation/imgs"),
    os.path.join(base_dir, "Validation/annotations")
]

for root_dir in root_dirs:
    for name in os.listdir(root_dir):
        if name.endswith(".zip"):
            old_path = os.path.join(root_dir, name)
            new_name = name.replace(".zip", "")
            new_path = os.path.join(root_dir, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")
