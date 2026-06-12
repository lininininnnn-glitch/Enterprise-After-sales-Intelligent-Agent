"""批量离线评测抽取与问答管线。"""

from __future__ import annotations

import json
from pathlib import Path

from src.agent.extractor import extract_keywords
from src.eval.metrics import evaluate, load_samples

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "eval_data.jsonl"
OUTPUT_PATH = BASE_DIR / "data" / "eval_report.json"


def main() -> int:
    if not DATA_PATH.exists():
        print(f"缺少评测数据：{DATA_PATH}")
        return 1

    samples = load_samples(str(DATA_PATH))
    for sample in samples:
        sample.predicted = extract_keywords(sample.input_text).to_dict()

    report = evaluate(samples)
    OUTPUT_PATH.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(report.to_dict())
    print(f"report saved: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
