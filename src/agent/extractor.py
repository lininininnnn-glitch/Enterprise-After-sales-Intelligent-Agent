"""结构化信息抽取与 JSON 解析工具。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ExtractionResult:
    intent: str = "unknown"
    product: str = ""
    order_id: str = ""
    vip_level: str = ""
    issue: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "intent": self.intent,
            "product": self.product,
            "order_id": self.order_id,
            "vip_level": self.vip_level,
            "issue": self.issue,
            "reason": self.reason,
        }


_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _clean_json_text(text: str) -> str:
    text = text.strip()
    match = _JSON_RE.search(text)
    if match:
        return match.group(0)
    return text


def strict_json_parse(text: str) -> Optional[dict[str, Any]]:
    try:
        return json.loads(_clean_json_text(text))
    except Exception:
        return None


def relaxed_json_parse(text: str) -> Optional[dict[str, Any]]:
    data = strict_json_parse(text)
    if data is not None:
        return data
    cleaned = text.strip()
    cleaned = cleaned.replace("'", '"')
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        return json.loads(_clean_json_text(cleaned))
    except Exception:
        return None


def extract_keywords(text: str) -> ExtractionResult:
    order_match = re.search(r"\b\d{3,}\b", text)
    vip_level = ""
    for level in ("SVIP", "黑钻会员", "白金VIP", "黄金VIP", "普通用户"):
        if level in text:
            vip_level = level
            break

    intent = "general"
    for key, name in [
        ("退款", "refund"),
        ("退货", "return"),
        ("换货", "exchange"),
        ("保修", "warranty"),
        ("发票", "invoice"),
        ("会员", "vip"),
    ]:
        if key in text:
            intent = name
            break

    issue = ""
    for keyword in ("质量问题", "破损", "发错", "少件", "延迟", "退款", "退货", "换货", "保修", "发票"):
        if keyword in text:
            issue = keyword
            break

    return ExtractionResult(
        intent=intent,
        product="",
        order_id=order_match.group(0) if order_match else "",
        vip_level=vip_level,
        issue=issue,
        reason=text.strip(),
    )
