"""评测数据构造脚本。"""

from __future__ import annotations

import json
from pathlib import Path

from src.agent.extractor import extract_keywords

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = BASE_DIR / "data" / "eval_data.jsonl"


SAMPLES = [
    "我的订单888到货后发现破损，想申请退款",
    "U1001是黄金VIP，退货规则是什么",
    "999号订单还在运输中，是否可以取消",
    "发票怎么开，需要提供什么信息",
]


def build_samples() -> list[dict]:
    rows = []
    for text in SAMPLES:
        ext = extract_keywords(text)
        rows.append(
            {
                "input_text": text,
                "expected": ext.to_dict(),
            }
        )
    return rows


def main() -> int:
    rows = build_samples()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"saved: {OUTPUT_PATH}")
    print(f"count: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
