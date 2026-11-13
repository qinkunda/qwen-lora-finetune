"""校验财务种子数据结构、金额、任务输出、隔离性和清单；仅使用标准库。"""
import hashlib
import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
TRAIN_PATH = DATA_DIR / "finance_seed_train.json"
TEST_PATH = DATA_DIR / "finance_seed_test.json"
MANIFEST_PATH = DATA_DIR / "finance_seed_manifest.json"
TASKS = {"【意图分类】", "【付款条款抽取】", "【预算预检解释】", "【对账差异解释】"}
INTENTS = {"policy_qa", "payment_precheck", "budget_precheck", "reconciliation", "out_of_scope"}
MONEY = Decimal("0.01")


def load_rows(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(rows, list) and rows, f"{path.name} 必须是非空数组"
    return rows


def task_of(row):
    return row["instruction"].split("】", 1)[0] + "】"


def char_jaccard(left, right):
    a, b = set(left), set(right)
    return len(a & b) / max(len(a | b), 1)


def validate_common(rows):
    seen = set()
    for index, row in enumerate(rows):
        assert set(row) == {"instruction", "input", "output"}, f"第{index}条字段不符"
        assert all(isinstance(row[key], str) for key in row), f"第{index}条必须是字符串"
        assert all(row[key].strip() for key in row), f"第{index}条存在空字段"
        task = task_of(row)
        assert task in TASKS, f"未知任务前缀：{task}"
        key = tuple(row[field].strip() for field in ("instruction", "input", "output"))
        assert key not in seen, f"存在完全重复样本：第{index}条"
        seen.add(key)


def validate_intent(row):
    payload = json.loads(row["output"])
    assert set(payload) == {"intent", "request_id"}
    assert payload["intent"] in INTENTS
    request_id = payload["request_id"]
    if request_id is not None:
        assert re.fullmatch(r"(PR|BR)-\d{3}", request_id)
        expected = "payment_precheck" if request_id.startswith("PR-") else "budget_precheck"
        assert payload["intent"] == expected
        assert request_id in row["input"]
    if payload["intent"] in {"policy_qa", "reconciliation", "out_of_scope"}:
        assert request_id is None


def validate_extraction(row):
    payload = json.loads(row["output"])
    expected = {
        "contract_id", "total_amount", "advance_payment_rate", "delivery_payment_rate",
        "retention_rate", "acceptance_required",
    }
    assert set(payload) == expected
    assert payload["contract_id"] in row["input"]
    assert Decimal(payload["total_amount"]) > 0
    rates = [Decimal(payload[key].rstrip("%")) for key in (
        "advance_payment_rate", "delivery_payment_rate", "retention_rate",
    )]
    assert all(Decimal("0") <= rate <= Decimal("100") for rate in rates)
    assert sum(rates) == Decimal("100"), "付款节点比例合计必须为 100%"
    assert payload["acceptance_required"] is True


def validate_budget(row):
    payload = json.loads(row["input"])
    annual = Decimal(payload["annual_budget"])
    committed = Decimal(payload["committed"])
    actual = Decimal(payload["actual"])
    requested = Decimal(payload["requested_amount"])
    available = (annual - committed - actual).quantize(MONEY)
    remaining = (available - requested).quantize(MONEY)
    assert payload["project_id"] in row["output"]
    for value in (available, requested):
        assert f"{value:.2f}" in row["output"], f"金额未忠实呈现：{value}"
    if remaining >= 0:
        assert "预检通过" in row["output"] and f"{remaining:.2f}" in row["output"]
    else:
        assert "预算调整或专项审批" in row["output"]
        assert f"{-remaining:.2f}" in row["output"]


def validate_reconciliation(row):
    payload = json.loads(row["input"])
    findings = payload["findings"]
    allowed = {"amount_mismatch", "missing_in_ledger", "missing_in_bank", "duplicate_bank_record"}
    assert isinstance(payload["matched_count"], int) and payload["matched_count"] >= 0
    assert f"{payload['matched_count']}笔精确匹配" in row["output"]
    assert f"发现{len(findings)}项差异" in row["output"]
    for finding in findings:
        assert finding["code"] in allowed
        assert finding["record_id"] in row["output"]
        if finding["code"] == "amount_mismatch":
            bank = Decimal(str(finding["bank_amount"]))
            ledger = Decimal(str(finding["ledger_amount"]))
            difference = abs(bank - ledger).quantize(MONEY)
            assert f"{difference:.2f}" in row["output"]


def validate_isolation(train, test):
    """意图问题做字符 Jaccard 去污染；结构化任务检查业务主键和输入复用。"""
    for train_row in train:
        for test_row in test:
            if task_of(train_row) != task_of(test_row):
                continue
            task = task_of(train_row)
            if task == "【意图分类】":
                similarity = char_jaccard(train_row["input"], test_row["input"])
                assert similarity <= 0.5, f"意图训练/测试问题过近：{similarity:.2f}"
            elif task == "【付款条款抽取】":
                train_id = json.loads(train_row["output"])["contract_id"]
                test_id = json.loads(test_row["output"])["contract_id"]
                assert train_id != test_id and train_row["input"] != test_row["input"]
            elif task == "【预算预检解释】":
                train_input = json.loads(train_row["input"])
                test_input = json.loads(test_row["input"])
                assert train_input["project_id"] != test_input["project_id"]
                assert train_row["input"] != test_row["input"]
            else:
                train_ids = {x["record_id"] for x in json.loads(train_row["input"])["findings"]}
                test_ids = {x["record_id"] for x in json.loads(test_row["input"])["findings"]}
                assert not train_ids & test_ids
                assert train_row["input"] != test_row["input"]


def validate_manifest(train, test):
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["dataset_status"] == "seed_only_not_trained"
    assert manifest["data_source"] == "synthetic_demo"
    assert manifest["train_records"] == len(train)
    assert manifest["test_records"] == len(test)
    counts = Counter(task_of(row) for row in train + test)
    assert manifest["task_counts"] == dict(sorted(counts.items()))
    for path in (TRAIN_PATH, TEST_PATH):
        assert manifest["files"][path.name] == hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    train = load_rows(TRAIN_PATH)
    test = load_rows(TEST_PATH)
    validate_common(train)
    validate_common(test)
    validators = {
        "【意图分类】": validate_intent,
        "【付款条款抽取】": validate_extraction,
        "【预算预检解释】": validate_budget,
        "【对账差异解释】": validate_reconciliation,
    }
    for row in train + test:
        validators[task_of(row)](row)
    validate_isolation(train, test)
    validate_manifest(train, test)
    counts = Counter(task_of(row) for row in train + test)
    print("任务分布:", json.dumps(dict(sorted(counts.items())), ensure_ascii=False))
    print(f"SEED_VALIDATION_OK: train={len(train)}, test={len(test)}")
    print("边界：种子数据校验通过，不代表已训练或已取得新模型指标。")


if __name__ == "__main__":
    main()
