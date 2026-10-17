import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = "/mnt/workspace/.cache/modelscope/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master"
ADAPTER = "saves/qwen1.5b-finance-lora-v2"

tests = json.load(open("test.json"))
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

json.dump(results, open("eval_compare_v2.json", "w"), ensure_ascii=False, indent=2)
print("=== 完成,结果在 eval_compare_v2.json ===", flush=True)