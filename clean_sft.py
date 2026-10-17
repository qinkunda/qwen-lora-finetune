# clean_sft.py
import json, random, os
BASE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(BASE, "sft_raw.json"), encoding="utf-8"))
seen, clean = set(), []
drop = {"dup": 0, "len": 0, "ref": 0}
for d in data:
    key = d["instruction"].strip()
    if key in seen: drop["dup"] += 1; continue
    if not (10 < len(d["output"]) < 500): drop["len"] += 1; continue
    if any(x in d["output"] for x in ["作为AI", "作为一个人工智能", "无法回答", "我不知道"]):
        drop["ref"] += 1; continue
    seen.add(key); clean.append(d)
random.seed(42); random.shuffle(clean)
json.dump(clean, open(os.path.join(BASE, "train_all.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"原始{len(data)} → 清洗后{len(clean)} (去重-{drop['dup']}, 长度-{drop['len']}, 兜底-{drop['ref']})")
print("\n---- 抽检10条 ----")
for d in clean[:10]:
    print("Q:", d["instruction"], "\nA:", d["output"][:60], "\n")