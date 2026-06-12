"""一键执行：构建样本、评测并输出报告。"""

from __future__ import annotations

from pathlib import Path

from scripts.build_eval_data import main as build_eval_main
from scripts.evaluate_pipeline import main as evaluate_main

BASE_DIR = Path(__file__).resolve().parents[1]


def main() -> int:
    print("[1/2] 生成评测数据")
    if build_eval_main() != 0:
        return 1

    print("[2/2] 执行离线评测")
    if evaluate_main() != 0:
        return 1

    print("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
