# gen_sft_data.py —— 基于当前项目 knowledge/ 生成合成 SFT 数据。
import json
import os
import time
from pathlib import Path

from openai import OpenAI

BASE = Path(__file__).resolve().parent
DOC_DIR = BASE / "knowledge"
OUT = BASE / "sft_raw.json"

# 与 RAG 项目一致的候选模型降级链；实际可用性以调用方账号为准。
MODEL_FALLBACK_LIST = [
    "qwen-plus",
    "qwen3.7-plus",
    "qwen-plus-2025-07-28",
    "qwen3.6-plus",
    "glm-5",
    "qwen3-32b",
    "qwen3.5-35b-a3b",
    "glm-4.5-air",
    "qwen3.6-flash",
    "qwen3.7-flash-2026-07-15",
    "qwen-long",
]

ANGLES = [
    ("事实查询", "员工查标准/额度/规定的直白问题"),
    ("费用计算", "给出具体天数、城市、票价，需要按制度算出金额或是否合规"),
    ("超标处理", "超过标准怎么办、费用谁承担、走什么流程"),
    ("审批流程", "审批链条、加签规则、出差申请和报销的先后关系"),
    ("权限与拒答", "普通员工打听不该看的数据(工资/成本台账/他人报销单)，标准做法是有礼貌地拒绝并说明权限"),
    ("易混点辨析", "容易混淆的条款对比，如城市分级差异、职级差异、事前申请与事后报销的区别"),
]

PROMPT = """你是财务制度培训专家。基于以下制度片段，从「{angle}」角度生成5条问答指令数据（{desc}）。
要求：问题是员工真实会问的口吻；答案忠实原文、口语化、200字内；各条之间问法尽量不同。
严格输出JSON数组：[{{"instruction": "问题", "output": "答案"}}]，不要任何其他文字。

制度片段：
{chunk}"""


def key_from_file(path: Path):
    """只读取调用方明确指定的当前项目 .env，不扫描其他目录。"""
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("DASHSCOPE_API_KEY") and "=" in line:
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if len(value) > 10:
                    return value
    return None


def load_api_key():
    """优先读环境变量，其次只读当前项目根目录 .env。"""
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if api_key:
        return api_key, "环境变量 DASHSCOPE_API_KEY"

    env_path = BASE / ".env"
    if env_path.is_file():
        api_key = key_from_file(env_path)
        if api_key:
            return api_key, str(env_path)

    raise SystemExit(
        "未找到密钥：请设置环境变量 DASHSCOPE_API_KEY，"
        "或在当前项目根目录 .env 中写入 DASHSCOPE_API_KEY=你的key。"
    )


class FallbackCaller:
    """一个候选模型调用失败后切换下一个；不在模块导入时发起网络请求。"""

    def __init__(self, client):
        self.client = client
        self.idx = 0
        self.stats = {}

    def chat(self, content):
        for _ in range(len(MODEL_FALLBACK_LIST)):
            model = MODEL_FALLBACK_LIST[self.idx]
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.85,
                )
                self.stats[model] = self.stats.get(model, 0) + 1
                return response.choices[0].message.content, model
            except Exception as exc:
                print(f"  × {model} 失败({str(exc)[:50]}...), 切换")
                self.idx = (self.idx + 1) % len(MODEL_FALLBACK_LIST)
        raise RuntimeError("全部候选模型都调用失败，请检查账号额度、模型权限和网络。")


def chunks(text, size=800):
    return [text[i:i + size] for i in range(0, len(text), size)]


def write_json(path, data):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=1)


def generate(caller, doc_dir=DOC_DIR, output=OUT):
    documents = sorted(
        path for path in doc_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}
    )
    if not documents:
        raise SystemExit(f"没找到语料：{doc_dir} 下需要至少一个 .txt 或 .md 文件")
    print(f"语料目录: {doc_dir} → {len(documents)} 个文档")

    todo = []
    for path in documents:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for chunk in chunks(text):
            if len(chunk.strip()) > 100:
                todo.append((path.name, chunk))

    calls = [(fn, chunk, angle) for fn, chunk in todo for angle in ANGLES]
    print(f"{len(todo)}个片段 × {len(ANGLES)}个角度 = {len(calls)}次调用，预计最多生成{len(calls) * 5}条")

    data = []
    for i, (fn, chunk, (angle, desc)) in enumerate(calls):
        try:
            content, model = caller.chat(PROMPT.format(angle=angle, desc=desc, chunk=chunk))
            content = content.strip()
            for junk in ("```json", "```"):
                content = content.replace(junk, "")
            for item in json.loads(content.strip()):
                data.append({
                    "instruction": item["instruction"],
                    "input": "",
                    "output": item["output"],
                })
        except Exception as exc:
            print(f"[{i}] skip: {str(exc)[:80]}")
        if i % 5 == 0:
            current = MODEL_FALLBACK_LIST[caller.idx]
            print(f"[{i}/{len(calls)}] {angle} | 当前模型:{current} | 已生成{len(data)}条")
            write_json(output, data)
        time.sleep(0.8)

    write_json(output, data)
    print("完成，共生成", len(data), "条")
    print("各模型贡献:", caller.stats)
    return data


def main():
    api_key, key_source = load_api_key()
    print("密钥来源:", key_source)  # 只显示来源，不打印密钥。
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    generate(FallbackCaller(client))


if __name__ == "__main__":
    main()
