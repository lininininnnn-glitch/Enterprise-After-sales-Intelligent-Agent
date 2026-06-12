"""大模型输出校验与重试提示生成。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ValidationResult:
    ok: bool
    reason: str = ""
    parsed: Optional[dict[str, Any]] = None


def validate_answer(answer: str) -> ValidationResult:
    text = (answer or "").strip()
    if not text:
        return ValidationResult(ok=False, reason="空输出")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return ValidationResult(ok=True, parsed=parsed)
        return ValidationResult(ok=False, reason="JSON 不是对象")
    except Exception:
        if len(text) < 2:
            return ValidationResult(ok=False, reason="输出过短")
        return ValidationResult(ok=True, reason="非JSON文本输出")


def build_fix_prompt(previous_answer: str, reason: str) -> str:
    return (
        "你上一轮输出存在问题，需要修正。\n"
        f"问题原因：{reason}\n"
        f"上一轮输出：{previous_answer}\n"
        "请仅基于已提供的工具结果、政策与上下文重新生成最终回复。"
        "如果仍然无法解决，请明确输出需要人工客服介入。"
    )
