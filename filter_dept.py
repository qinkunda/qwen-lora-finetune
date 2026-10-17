# filter_dept.py —— 剔除"部门预算"幻觉（原文只有科目预算，没有部门预算）
import json, os, re
BASE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(BASE, "train.json"), encoding="utf-8"))
pat = re.compile(r"(销售|研发|行政|市场|人事|财务|技术|客服)部(的)?(8月)?预算(是)?[0-9]")
kept = [d for d in data if not pat.search(d["output"])]
json.dump(kept, open(os.path.join(BASE, "train.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"剔了{len(data)-len(kept)}条幻觉, 最终{len(kept)}条")