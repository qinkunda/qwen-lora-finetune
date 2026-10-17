# decontaminate.py
import json, os
BASE = os.path.dirname(os.path.abspath(__file__))
train = json.load(open(os.path.join(BASE, "train_all.json"), encoding="utf-8"))
test = json.load(open(os.path.join(BASE, "test.json"), encoding="utf-8"))
eval_qs = [d["instruction"] for d in test]
def sim(a, b):
    A, B = set(a), set(b)
    return len(A & B) / max(len(A | B), 1) > 0.5
kept = [d for d in train if not any(sim(d["instruction"], q) for q in eval_qs)]
json.dump(kept, open(os.path.join(BASE, "train.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"剔了{len(train)-len(kept)}条相似, 最终训练集{len(kept)}条")