# gen_sft_data.py —— 模型降级链版，整体替换直接跑
import json, os, time
from openai import OpenAI

ROOT = r"C:\Users\39034\Desktop\AI_Project"
SKIP = ("node_modules", ".git", "models", "__pycache__", "chroma_db", "venv", ".venv", ".idea")

# 与你RAG生产系统同一条降级链
MODEL_FALLBACK_LIST = [
    "qwen-plus",                # 主力
    "qwen3.7-plus",             # 新一代旗舰
    "qwen-plus-2025-07-28",     # 主力模型快照
    "qwen3.6-plus",
    "glm-5",
    "qwen3-32b",
    "qwen3.5-35b-a3b",
    "glm-4.5-air",
    "qwen3.6-flash",            # 快而便宜
    "qwen3.7-flash-2026-07-15",
    "qwen-long",                # 兜底
]

def key_from_file(p):
    try:
        for line in open(p, encoding="utf-8", errors="ignore"):
            line = line.strip()
            if line.startswith("DASHSCOPE_API_KEY") and "=" in line:
                v = line.split("=", 1)[1].strip().strip('"').strip("'")
                if len(v) > 10:
                    return v
    except Exception:
        pass
    return None

env_path, api_key = None, None
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in SKIP]
    for f in sorted(filenames):
        if f.endswith(".env"):
            v = key_from_file(os.path.join(dirpath, f))
            if v:
                env_path, api_key = os.path.join(dirpath, f), v
                break
    if api_key:
        break
if not api_key:
    raise SystemExit("没找到key，在 Lora-Finetune 里新建 .env 写 DASHSCOPE_API_KEY=你的key 后重跑")
print("key文件:", env_path)

DOC_DIR = os.path.join(os.path.dirname(env_path), "knowledge")
if not (os.path.isdir(DOC_DIR) and any(f.endswith(".txt") for f in os.listdir(DOC_DIR))):
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP]
        if os.path.basename(dirpath).lower() == "knowledge" and any(f.endswith(".txt") for f in filenames):
            DOC_DIR = dirpath
            break
n_docs = len([f for f in os.listdir(DOC_DIR) if f.endswith((".txt", ".md"))])
assert n_docs > 0, "没找到语料"
print("语料目录:", DOC_DIR, "→", n_docs, "个文档")

client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

class FallbackCaller:
    """复用RAG生产系统的模型降级链：一个模型403/失败，秒切下一个"""
    def __init__(self):
        self.idx = 0
        self.stats = {}
    def chat(self, content):
        for _ in range(len(MODEL_FALLBACK_LIST)):
            model = MODEL_FALLBACK_LIST[self.idx]
            try:
                r = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.85)
                self.stats[model] = self.stats.get(model, 0) + 1
                return r.choices[0].message.content, model
            except Exception as e:
                print(f"  × {model} 失败({str(e)[:50]}...), 切换")
                self.idx = (self.idx + 1) % len(MODEL_FALLBACK_LIST)
        raise RuntimeError("全部模型都失败——免费额度可能全用完了，去百炼控制台关'仅免费额度'(全程不到1块钱)")

caller = FallbackCaller()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sft_raw.json")

ANGLES = [
    ("事实查询",   "员工查标准/额度/规定的直白问题"),
    ("费用计算",   "给出具体天数、城市、票价，需要按制度算出金额或是否合规"),
    ("超标处理",   "超过标准怎么办、费用谁承担、走什么流程"),
    ("审批流程",   "审批链条、加签规则、出差申请和报销的先后关系"),
    ("权限与拒答", "普通员工打听不该看的数据(工资/成本台账/他人报销单)，标准做法是有礼貌地拒绝并说明权限"),
    ("易混点辨析", "容易混淆的条款对比，如城市分级差异、职级差异、事前申请与事后报销的区别"),
]

PROMPT = """你是财务制度培训专家。基于以下制度片段，从「{angle}」角度生成5条问答指令数据（{desc}）。
要求：问题是员工真实会问的口吻；答案忠实原文、口语化、200字内；各条之间问法尽量不同。
严格输出JSON数组：[{{"instruction": "问题", "output": "答案"}}]，不要任何其他文字。

制度片段：
{chunk}"""

def chunks(text, size=800):
    return [text[i:i+size] for i in range(0, len(text), size)]

data, todo = [], []
for fn in [f for f in os.listdir(DOC_DIR) if f.endswith((".txt", ".md"))]:
    text = open(os.path.join(DOC_DIR, fn), encoding="utf-8", errors="ignore").read()
    for ck in chunks(text):
        if len(ck.strip()) > 100:
            todo.append((fn, ck))
calls = [(fn, ck, ang) for fn, ck in todo for ang in ANGLES]
print(f"{len(todo)}个片段 × {len(ANGLES)}个角度 = {len(calls)}次调用, 预计生成{len(calls)*5}条")

for i, (fn, ck, (ang, desc)) in enumerate(calls):
    try:
        content, model = caller.chat(PROMPT.format(angle=ang, desc=desc, chunk=ck))
        content = content.strip()
        for junk in ("```json", "```"):
            content = content.replace(junk, "")
        for item in json.loads(content.strip()):
            data.append({"instruction": item["instruction"], "input": "", "output": item["output"]})
    except Exception as e:
        print(f"[{i}] skip: {str(e)[:80]}")
    if i % 5 == 0:
        cur = MODEL_FALLBACK_LIST[caller.idx]
        print(f"[{i}/{len(calls)}] {ang} | 当前模型:{cur} | 已生成{len(data)}条")
        json.dump(data, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    time.sleep(0.8)

json.dump(data, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("完成, 共生成", len(data), "条")
print("各模型贡献:", caller.stats)   # 这行记一下，README里写真实生成模型