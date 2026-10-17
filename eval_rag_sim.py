import json, glob, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = "/mnt/workspace/.cache/modelscope/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master"
ADAPTER = "saves/qwen1.5b-finance-lora-v2"

docs = []
for f in sorted(glob.glob("knowledge/*.txt") + glob.glob("knowledge/*.md")):
    docs.append(open(f, encoding="utf-8").read())
CTX = "\n\n".join(docs)
print(f"知识库 {len(docs)} 个文档, 共 {len(CTX)} 字", flush=True)

tests = json.load(open("test.json"))
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

json.dump(results, open("eval_rag_sim.json", "w"), ensure_ascii=False, indent=2)
print("=== 完成 ===", flush=True)