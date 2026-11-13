"""确定性生成企业财务多任务种子数据；不调用模型，不修改原 115 条训练集。"""
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
TRAIN_PATH = DATA_DIR / "finance_seed_train.json"
TEST_PATH = DATA_DIR / "finance_seed_test.json"
MANIFEST_PATH = DATA_DIR / "finance_seed_manifest.json"

INTENT = "【意图分类】"
EXTRACT = "【付款条款抽取】"
BUDGET = "【预算预检解释】"
RECON = "【对账差异解释】"


def record(prefix, instruction, input_text, output):
    return {
        "instruction": f"{prefix}{instruction}",
        "input": input_text,
        "output": output,
    }


def intent_record(question, intent, request_id=None):
    output = json.dumps(
        {"intent": intent, "request_id": request_id},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return record(INTENT, "识别以下企业财务请求的意图和申请编号，仅输出 JSON。", question, output)


def extraction_record(contract_id, total, advance, delivery, retention, extra_clause):
    input_text = (
        f"合同{contract_id}含税总额{total}元。预付款{advance}%，到货款{delivery}%，"
        f"质保金{retention}%。{extra_clause}"
    )
    output = json.dumps({
        "contract_id": contract_id,
        "total_amount": f"{Decimal(total):.2f}",
        "advance_payment_rate": f"{advance}%",
        "delivery_payment_rate": f"{delivery}%",
        "retention_rate": f"{retention}%",
        "acceptance_required": True,
    }, ensure_ascii=False, separators=(",", ":"))
    return record(EXTRACT, "从 input 的合成合同条款中提取付款节点，仅输出 JSON。", input_text, output)


def money(value):
    return f"{Decimal(str(value)).quantize(Decimal('0.01')):.2f}"


def budget_record(project_id, annual, committed, actual, requested):
    annual, committed, actual, requested = map(Decimal, map(str, (annual, committed, actual, requested)))
    available = annual - committed - actual
    remaining = available - requested
    input_text = json.dumps({
        "project_id": project_id,
        "annual_budget": money(annual),
        "committed": money(committed),
        "actual": money(actual),
        "requested_amount": money(requested),
    }, ensure_ascii=False, separators=(",", ":"))
    output = (
        f"{project_id}可用预算为{money(available)}元"
        f"（{money(annual)}-{money(committed)}-{money(actual)}）。"
        f"本次申请{money(requested)}元，"
    )
    if remaining >= 0:
        output += f"未超过可用预算，预检通过，剩余{money(remaining)}元。"
    else:
        output += f"超过可用预算{money(-remaining)}元，需进入预算调整或专项审批流程。"
    return record(BUDGET, "根据 input 的合成预算数据解释预检结果，金额不得改动。", input_text, output)


def reconciliation_record(matched_count, findings):
    input_text = json.dumps({
        "matched_count": matched_count,
        "findings": findings,
    }, ensure_ascii=False, separators=(",", ":"))
    parts = [f"对账完成：{matched_count}笔精确匹配，发现{len(findings)}项差异。"]
    for finding in findings:
        code = finding["code"]
        record_id = finding["record_id"]
        if code == "amount_mismatch":
            bank = Decimal(str(finding["bank_amount"]))
            ledger = Decimal(str(finding["ledger_amount"]))
            parts.append(
                f"{record_id}银行金额{money(bank)}元、总账金额{money(ledger)}元，"
                f"差额{money(abs(bank - ledger))}元，需核对记账凭证。"
            )
        elif code == "missing_in_ledger":
            parts.append(f"{record_id}银行有记录而总账缺失，需检查是否漏记凭证。")
        elif code == "missing_in_bank":
            parts.append(f"{record_id}总账有记录而银行流水缺失，列为银行未达事项。")
        elif code == "duplicate_bank_record":
            parts.append(f"{record_id}存在重复银行流水，需去重后复核。")
        else:
            raise ValueError(f"未知对账差异类型：{code}")
    if not findings:
        parts.append("无需生成差异工单。")
    return record(RECON, "根据 input 的合成对账结果生成简洁中文说明，不得新增差异。", input_text, "".join(parts))


def build():
    train = []
    test = []

    intent_train = [
        ("预付款最多可以付多少比例？", "policy_qa", None),
        ("电子发票需要上传什么文件？", "policy_qa", None),
        ("深圳普通员工住宿上限是多少？", "policy_qa", None),
        ("请预检付款申请 PR-001", "payment_precheck", "PR-001"),
        ("付款申请 PR-002 为什么需要人工复核？", "payment_precheck", "PR-002"),
        ("我想办理一笔付款预检", "payment_precheck", None),
        ("检查预算申请 BR-001", "budget_precheck", "BR-001"),
        ("BR-002 是否超出可用预算？", "budget_precheck", "BR-002"),
        ("帮我做一下预算预检", "budget_precheck", None),
        ("跑一下银行与总账对账并列出差异", "reconciliation", None),
        ("这次对账有哪些金额不一致？", "reconciliation", None),
        ("今天适合去哪跑步？", "out_of_scope", None),
    ]
    intent_test = [
        ("质保金什么时候可以申请？", "policy_qa", None),
        ("核对付款单 PR-003 的里程碑金额", "payment_precheck", "PR-003"),
        ("预算单 BR-003 能否通过预检？", "budget_precheck", "BR-003"),
        ("请推荐一家好吃的餐厅", "out_of_scope", None),
    ]
    train.extend(intent_record(*item) for item in intent_train)
    test.extend(intent_record(*item) for item in intent_test)

    extraction_train = [
        ("HT-2026-101", "80000", 20, 70, 10, "到货款支付前必须取得验收单。"),
        ("HT-2026-102", "120000", 30, 60, 10, "预付款凭合同与发票申请，到货款另附验收材料。"),
        ("HT-2026-103", "50000", 10, 80, 10, "到货验收完成后方可申请到货款。"),
        ("HT-2026-104", "200000", 25, 65, 10, "验收单为到货款必要附件。"),
        ("HT-2026-105", "60000", 30, 50, 20, "到货款审核时核验验收单与发票。"),
        ("HT-2026-106", "90000", 15, 75, 10, "未提供验收单不得进入到货款审批。"),
    ]
    extraction_test = [
        ("HT-2026-201", "75000", 20, 60, 20, "到货款申请必须附验收单。"),
        ("HT-2026-202", "150000", 30, 65, 5, "验收完成后方可申请到货节点款项。"),
    ]
    train.extend(extraction_record(*item) for item in extraction_train)
    test.extend(extraction_record(*item) for item in extraction_test)

    budget_train = [
        ("LOG-2026", 200000, 80000, 50000, 50000),
        ("LOG-2026-A", 200000, 80000, 50000, 90000),
        ("IT-2026", 150000, 30000, 20000, 40000),
        ("IT-2026-A", 150000, 30000, 20000, 120000),
        ("MKT-2026", 100000, 10000, 20000, 70000),
        ("MKT-2026-A", 100000, 10000, 20000, 80000),
        ("RND-2026", 300000, 50000, 50000, 100000),
        ("RND-2026-A", 300000, 50000, 50000, 250000),
    ]
    budget_test = [
        ("HR-2026", 120000, 20000, 10000, 95000),
        ("OPS-2026", 180000, 20000, 10000, 120000),
    ]
    train.extend(budget_record(*item) for item in budget_train)
    test.extend(budget_record(*item) for item in budget_test)

    recon_train = [
        (12, []),
        (8, [{"code": "amount_mismatch", "record_id": "BK-101", "bank_amount": "1000", "ledger_amount": "900"}]),
        (6, [{"code": "missing_in_ledger", "record_id": "BK-102"}]),
        (9, [{"code": "missing_in_bank", "record_id": "BK-103"}]),
        (7, [{"code": "duplicate_bank_record", "record_id": "BK-104"}]),
        (4, [
            {"code": "amount_mismatch", "record_id": "BK-105", "bank_amount": "2600", "ledger_amount": "2500"},
            {"code": "missing_in_ledger", "record_id": "BK-106"},
        ]),
    ]
    recon_test = [
        (5, [{"code": "duplicate_bank_record", "record_id": "BK-201"}]),
        (3, [
            {"code": "amount_mismatch", "record_id": "BK-202", "bank_amount": "4800", "ledger_amount": "4500"},
            {"code": "missing_in_bank", "record_id": "BK-203"},
        ]),
    ]
    train.extend(reconciliation_record(*item) for item in recon_train)
    test.extend(reconciliation_record(*item) for item in recon_test)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path, rows in ((TRAIN_PATH, train), (TEST_PATH, test)):
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    all_rows = train + test
    task_counts = Counter(row["instruction"].split("】", 1)[0] + "】" for row in all_rows)
    manifest = {
        "dataset_status": "seed_only_not_trained",
        "data_source": "synthetic_demo",
        "description": "企业财务多任务种子数据；未纳入原 data/train.json，未产生新训练或评测指标。",
        "train_records": len(train),
        "test_records": len(test),
        "task_counts": dict(sorted(task_counts.items())),
        "files": {},
    }
    for path in (TRAIN_PATH, TEST_PATH):
        manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build()
