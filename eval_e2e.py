import json, glob, os, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dashscope import TextEmbedding
import chromadb

BASE = Path(__file__).resolve().parent
MODEL = os.getenv("BASE_MODEL_PATH", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER = os.getenv("LORA_ADAPTER_PATH", str(BASE / "saves/qwen1.5b-finance-lora-v2"))

chunks = []
for f in sorted(glob.glob(str(BASE / "knowledge/*.txt")) + glob.glob(str(BASE / "knowledge/*.md"))):
    buf = ""
    for line in open(f, encoding="utf-8").read().split("\n"):
        if len(buf) + len(line) > 500 and buf:
            chunks.append(buf.strip()); buf = ""
        buf += line + "\n"
    if buf.strip(): chunks.append(buf.strip())
print(f"共 {len(chunks)} 个切片", flush=True)

def embed(texts):
    r = TextEmbedding.call(model="text-embedding-v2", input=texts)
    assert r.status_code == 200, r
    return [e["embedding"] for e in r.output["embeddings"]]

col = chromadb.Client().create_collection("kb_finance")
col.add(ids=[str(i) for i in range(len(chunks))], embeddings=embed(chunks), documents=chunks)
print("向量索引完成", flush=True)

tests = json.load(open(BASE / "data/test.json", encoding="utf-8"))
tok = AutoTokenizer.from_pretrained(MODEL)

def gen(model, q):
    hits = col.query(query_embeddings=embed([q]), n_results=3)["documents"][0]
    prompt = f"以下是公司差旅制度相关条款：\n{chr(10).join(hits)}\n\n请严格依据以上条款回答问题，条款没有的就说不知道。问题：{q}"
    msgs = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
    out = model.generate(**ids, max_new_tokens=256, do_sample=False)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

results = []
print("=== 原模型+端到端检索 ===", flush=True)
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
for i, t in enumerate(tests):
    print(f"base {i+1}/10", flush=True)
    results.append({"q": t["instruction"], "ref": t["output"], "base_e2e": gen(base, t["instruction"])})
del base
torch.cuda.empty_cache()

print("=== 微调模型+端到端检索 ===", flush=True)
ft = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
ft = PeftModel.from_pretrained(ft, ADAPTER)
for i, t in enumerate(tests):
    print(f"ft {i+1}/10", flush=True)
    results[i]["ft_e2e"] = gen(ft, t["instruction"])

output_path = BASE / "results/eval_e2e.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(output_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"=== 完成,结果在 {output_path} ===", flush=True)