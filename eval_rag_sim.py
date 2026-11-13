import json, glob, os, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = Path(__file__).resolve().parent
MODEL = os.getenv("BASE_MODEL_PATH", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER = os.getenv("LORA_ADAPTER_PATH", str(BASE / "saves/qwen1.5b-finance-lora-v2"))

docs = []
for f in sorted(glob.glob(str(BASE / "knowledge/*.txt")) + glob.glob(str(BASE / "knowledge/*.md"))):
    docs.append(open(f, encoding="utf-8").read())
CTX = "\n\n".join(docs)
print(f"知识库 {len(docs)} 个文档, 共 {len(CTX)} 字", flush=True)

tests = json.load(open(BASE / "data/test.json", encoding="utf-8"))
tok = AutoTokenizer.from_pretrained(MODEL)

def gen(model, q):
    prompt = f"以下是公司差旅制度原文：\n{CTX}\n\n请严格依据原文回答问题，原文没有的就说不知道。问题：{q}"
    msgs = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
    out = model.generate(**ids, max_new_tokens=256, do_sample=False)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

results = []
print("=== 原模型+制度原文 ===", flush=True)
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
for i, t in enumerate(tests):
    print(f"base {i+1}/10", flush=True)
    results.append({"q": t["instruction"], "ref": t["output"], "base_rag": gen(base, t["instruction"])})
del base
torch.cuda.empty_cache()

print("=== 微调模型+制度原文 ===", flush=True)
ft = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
ft = PeftModel.from_pretrained(ft, ADAPTER)
for i, t in enumerate(tests):
    print(f"ft {i+1}/10", flush=True)
    results[i]["ft_rag"] = gen(ft, t["instruction"])

output_path = BASE / "results/eval_rag_sim.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(output_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"=== 完成,结果在 {output_path} ===", flush=True)