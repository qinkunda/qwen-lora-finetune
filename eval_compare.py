import json, os, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = Path(__file__).resolve().parent
MODEL = os.getenv("BASE_MODEL_PATH", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER = os.getenv("LORA_ADAPTER_PATH", str(BASE / "saves/qwen1.5b-finance-lora-v2"))

tests = json.load(open(BASE / "data/test.json", encoding="utf-8"))
tok = AutoTokenizer.from_pretrained(MODEL)

def gen(model, q):
    msgs = [{"role": "user", "content": q}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
    out = model.generate(**ids, max_new_tokens=256, do_sample=False)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

print("=== 加载原模型 ===", flush=True)
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
results = []
for i, t in enumerate(tests):
    print(f"base 答题 {i+1}/{len(tests)}", flush=True)
    results.append({"q": t["instruction"], "ref": t["output"], "base": gen(base, t["instruction"])})
del base
torch.cuda.empty_cache()

print("=== 加载微调模型 ===", flush=True)
ft = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
ft = PeftModel.from_pretrained(ft, ADAPTER)
for i, t in enumerate(tests):
    print(f"ft 答题 {i+1}/{len(tests)}", flush=True)
    results[i]["ft"] = gen(ft, t["instruction"])

output_path = BASE / "results/eval_compare_v2.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(output_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"=== 完成,结果在 {output_path} ===", flush=True)