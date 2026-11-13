# qwen-lora-finetune

企业制度问答场景下的 LoRA 微调**对照实验**：用四线对比验证“微调管表达、RAG 管知识、护栏管边界”的架构分工。本仓另新增企业财务多任务种子数据，覆盖付款、预算和对账，但**尚未训练、尚未产生新指标**。

本项目是三仓架构的微调层：

- **finance-rag-agent**：知识工程、RAG、多场景规则、多 Agent 与 MCP，负责检索、确定性计算和权限审计。
- **dify-rag-workflow**：对话与工作流编排。
- **qwen-lora-finetune（本仓）**：验证微调模型在 RAG 链路中的定位——生成端的表达组件，而非知识载体。

> 数据声明：本仓训练、评测和新增种子数据均使用虚构企业制度或合成业务记录，不含真实企业、人员或交易信息。

## 核心结论（既有实验，保持原记录）

10 题隔离评测集（事实 3 / 计算 1 / 边界流程 4 / 拒答 2），四线对照：

| 线路 | 知识命中 |
| --- | --- |
| 原模型裸答 | 0/10 |
| 微调裸答（v2, 10 epochs） | 1/10 |
| 原模型 + 检索上下文 | 5/10 |
| **微调 + 检索上下文** | **7/10** |

一句话：**SFT 教“怎么答”立竿见影，教“答什么”不可靠且剂量越大幻觉越自信；知识的正解是检索，微调在检索之上进一步提升任务对齐。**

这些数字仅对应原 115 条训练集和原 10 题评测集，不适用于本次新增种子数据。

## 新增：财务多任务种子数据（未训练）

新增文件与原实验数据完全分开：

| 文件 | 内容 | 状态 |
| --- | --- | --- |
| `data/finance_seed_train.json` | 32 条训练种子 | 已做静态校验，未训练 |
| `data/finance_seed_test.json` | 10 条隔离测试种子 | 已做静态校验，未跑模型 |
| `data/finance_seed_manifest.json` | 数量、任务分布、SHA-256、状态 | `seed_only_not_trained` |
| `build_finance_seed.py` | 确定性生成器；金额由 `Decimal` 计算 | 不调用外部模型 |
| `validate_finance_seed.py` | 结构、JSON、金额、任务输出、隔离性和清单校验 | 已通过 |

任务分布：

| 任务 | 样本数 | 目标 |
| --- | ---: | --- |
| 意图分类 | 16 | 区分制度问答、付款预检、预算预检、对账和越界问题，抽取 PR/BR 编号 |
| 付款条款抽取 | 8 | 从合成合同条款抽取金额、付款比例和验收要求，输出结构化 JSON |
| 预算预检解释 | 10 | 忠实解释“年度预算－已承诺－已发生－本次申请”，不自行改数 |
| 对账差异解释 | 8 | 把金额差异、缺失、未达和重复流水转成简洁中文说明 |

```bash
# 确定性重建 32 + 10 条种子数据；不会修改 data/train.json 或 data/test.json
python3 build_finance_seed.py

# 校验标准 Alpaca 字段、任务 schema、金额、输出证据和训练/测试隔离
python3 validate_finance_seed.py
```

**使用边界**：种子数据尚未合并进 LLaMA-Factory 的 `my_finance` 数据集。后续实验应单独建立数据集配置和结果目录，不能覆盖原实验，也不能把“校验通过”写成“训练通过”。意图/结构化任务适合做任务对齐；企业实际路由和金额计算仍应由核心仓确定性程序负责。

## 既有实验的关键发现

1. **风格迁移完全成功**：微调后回答从“百科废话腔”变为简洁制度腔，主动引用条款；数据风格会传导。
2. **知识注入不可靠，且随训练量幻觉自信化**：3 epochs（24 步，loss 2.57）欠拟合 0/10；10 epochs（80 步，loss 0.60）仅 1/10，但对未学问题更敢编。
3. **检索是知识的正解**：注入制度原文后，微调模型 1/10 → 7/10，且超过同条件原模型（5/10）。
4. **表格行定位是小模型短板**：两模型面对补助标准表都发生错抄，这是 RAG 切片粒度/表格结构化的优化点。
5. **拒答边界不能只靠 SFT**：边界必须由输入输出护栏、权限和业务规则共同保证。

## 原实验设计（120 → 115 条）

```text
gen_sft_data.py     多角度合成：6 角度 × 每角度 5 条，候选模型降级链
clean_sft.py        去重 + 长度过滤(10-500字符) + 兜底话术过滤
decontaminate.py    与评测集字符 Jaccard 相似度 >0.5 剔除（-2 条）
filter_dept.py      幻觉正则过滤：部门预算类张冠李戴（-3 条）
```

### 密钥读取安全修复

`gen_sft_data.py` 现在：

- 优先读取环境变量 `DASHSCOPE_API_KEY`；
- 其次只读取**当前项目根目录** `.env`；
- 不再扫描 Desktop、其他项目或任意本机目录；
- 增加 `main()` 入口保护，导入脚本不会自动找密钥或发起模型调用；
- 不打印密钥内容。

### 训练配置（LLaMA-Factory 0.9.5，Qwen2.5-1.5B-Instruct）

| 配置 | v1（欠拟合） | v2（正式版） |
| --- | --- | --- |
| epochs / 步数 | 3 / 24 | 10 / 80 |
| learning_rate | 1e-4 | 2e-4 |
| loss 曲线 | 2.82 → 2.57 | 2.69 → 0.60 |
| LoRA | rank 8 / alpha 16 / target all | 同左 |
| 其他 | bf16, cutoff 1024, batch 4 × grad_accum 4 | 同左 |
| 训练耗时（A10） | 23 秒 | 约 1 分钟 |

原评测脚本见 `eval_compare.py`（裸答）、`eval_rag_sim.py`（注入制度原文）和 `eval_e2e.py`（向量检索端到端）。三者现在使用项目相对的数据与输出路径，模型和 Adapter 可通过环境变量指定，不再依赖原机器的 `/mnt/workspace/...` 绝对路径。

## 文件结构

```text
build_finance_seed.py          # 新增：财务多任务种子数据确定性生成
validate_finance_seed.py       # 新增：种子数据校验
gen_sft_data.py                # 原 SFT 数据合成（已修复密钥读取与入口保护）
clean_sft.py                   # 原清洗
decontaminate.py               # 原评测集去污染
filter_dept.py                 # 原幻觉正则过滤
train.yaml                     # 原 LLaMA-Factory v2 配置
eval_compare.py                # 原裸答对比
eval_rag_sim.py                # 原 gold-context 对比
eval_e2e.py                    # 原端到端向量检索对比
data/train.json                # 原 115 条训练集（未修改）
data/test.json                 # 原 10 题隔离评测集（未修改）
data/finance_seed_*.json       # 新增 32 + 10 条种子数据及清单（未训练）
results/                       # 原评测原始输出（未修改）
images/                        # 原 v1/v2 loss 曲线（未修改）
```

## 复现原实验

```bash
pip install llamafactory==0.9.5
export USE_MODELSCOPE_HUB=1
llamafactory-cli train train.yaml

# 可选：模型路径和 Adapter 路径；不传时使用 train.yaml 对应的模型名和默认 saves 路径
export BASE_MODEL_PATH=/your/path/Qwen2.5-1.5B-Instruct
export LORA_ADAPTER_PATH=/your/path/saves/qwen1.5b-finance-lora-v2

python3 eval_compare.py       # 裸答对比，输出到 results/
python3 eval_rag_sim.py       # +制度原文，输出到 results/
python3 eval_e2e.py           # +向量检索；另需 DASHSCOPE_API_KEY 和相关依赖
```

原实验使用单卡 24G（A10/3090/4090）。LoRA 权重未上传；adapter 配置见 `train.yaml`。本次仅做脚本静态编译和种子数据校验，未重新执行 GPU 训练或以上三项模型评测。

## 后续路线

- [ ] 为新种子数据单独建立 LLaMA-Factory 数据集配置，训练后使用新结果目录，不回写原 0/1/5/7 实验。
- [ ] 微调模型通过 OpenAI 兼容服务嵌回核心仓，仅承担表达或结构化任务，不替代权限与规则引擎。
- [ ] 做端到端“向量检索 + 微调生成”复测，保留 gold-context 与真实检索结果的口径差异。
- [ ] 表格类知识切片结构化。
