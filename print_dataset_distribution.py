import os
import json
from tqdm import tqdm

def collect_json_paths(root_dir):
    json_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(".json"):
                json_files.append(os.path.join(dirpath, f))
    return sorted(json_files)

# JSON ↔ RGBD 
def make_rgbd_tuples(json_files, imgs_root, anns_root):
    """
    annotations 경로 기준으로 imgs 경로를 맞춰 (rgb, depth, json) 튜플 생성
    예시:
        annotations/.../TL_상품2D데이터_흡착그리퍼_후추/식품_CR05_후추_00001_01_20231026.json
        → imgs/.../TL_상품2D데이터_흡착그리퍼_후추/식품_CR05_후추_00001_01_20231026.jpg
        imgs/.../TL_상품2D데이터_흡착그리퍼_후추/식품_CR05_후추_00001_01_20231026_depth.png
    """
    tuples = []
    for json_path in json_files:
        # annotations 기준 상대 경로
        rel_path = os.path.relpath(json_path, anns_root)
        rel_dir = os.path.dirname(rel_path)
        base = os.path.splitext(os.path.basename(json_path))[0]

        # prefix 변환 (TL↔TS, VL↔VS)
        if rel_dir.startswith("TL_"):
            rel_dir_corrected = rel_dir.replace("TL_", "TS_", 1)
        elif rel_dir.startswith("VL_"):
            rel_dir_corrected = rel_dir.replace("VL_", "VS_", 1)
        else:
            rel_dir_corrected = rel_dir  # 변환 필요 없으면 그대로

        # imgs 폴더 경로 조합
        rgb_path = os.path.join(imgs_root, rel_dir_corrected, base + ".jpg")
        depth_path = os.path.join(imgs_root, rel_dir_corrected, base + "_depth.png")

        # 확장자 fallback
        if not os.path.exists(rgb_path):
            alt_exts = [".jpeg", ".JPEG", ".JPG", ".png"]
            for ext in alt_exts:
                alt_candidate = os.path.join(imgs_root, rel_dir_corrected, base + ext)
                if os.path.exists(alt_candidate):
                    rgb_path = alt_candidate
                    break

        if os.path.exists(rgb_path):
            tuples.append((rgb_path, depth_path, json_path))
        else:
            print(f"[WARN] Missing RGB for: {json_path}")

    return tuples



# 데이터 분포도 뽑아내기
def count_success_fail(file_list, name="Dataset"):
    success_count, fail_count = 0, 0
    for _, _, json_path in tqdm(file_list, desc=f"Counting {name}"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            result = data.get("grip_succeed", None)

            # 1 = SUCCESS <-> 0 = FAIL
            if result in [1, "1", True, "True"]:
                success_count += 1
            else:
                fail_count += 1

        except Exception:
            fail_count += 1

    total = success_count + fail_count
    print(f"\n [{name}] Samples: {total}")
    print(f" Success: {success_count}")
    print(f" Fail: {fail_count}")
    print(f"Success Ratio: {success_count / (total + 1e-6):.3f}")
    return success_count, fail_count


train_imgs_root = "data/Training/imgs"
train_anns_root = "data/Training/annotations"
val_imgs_root = "data/Validation/imgs"
val_anns_root = "data/Validation/annotations"

train_jsons = collect_json_paths(train_anns_root)
val_jsons = collect_json_paths(val_anns_root)

train_files = make_rgbd_tuples(train_jsons, train_imgs_root, train_anns_root)
val_files = make_rgbd_tuples(val_jsons, val_imgs_root, val_anns_root)

print(f"\n Training JSON files: {len(train_jsons)} → RGBD matched: {len(train_files)}")
print(f" Validation JSON files: {len(val_jsons)} → RGBD matched: {len(val_files)}")

# 분포 확인
train_success, train_fail = count_success_fail(train_files, "Training")
val_success, val_fail = count_success_fail(val_files, "Validation")

print("\n===== 전체 합계 =====")
print(f"전체 Success: {train_success + val_success}")
print(f"전체 Fail: {train_fail + val_fail}")
print(f"전체 Success Ratio: {(train_success + val_success) / ((train_success + val_success + train_fail + val_fail) + 1e-6):.3f}")
