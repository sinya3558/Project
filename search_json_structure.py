import os, json

root = "data/Training/annotations"
found = False

for dp, _, fs in os.walk(root):
    for f in fs:
        if f.endswith(".json"):
            path = os.path.join(dp, f)
            print(f"\n {path}")  # 파일 경로 먼저 출력시키고
            with open(path, "r") as j:
                try:
                    data = json.load(j)
                    print("최상단 키 목록:", list(data.keys()))
                    if "grip_succeed" not in data:
                        # 내부 키 탐색
                        for k, v in data.items():
                            if isinstance(v, dict):
                                print(f"  - {k} 내부 키:", list(v.keys()))
                    found = True
                except Exception as e:
                    print(f"JSON 가져오기 실패: {e}")
            break
    if found:
        break
