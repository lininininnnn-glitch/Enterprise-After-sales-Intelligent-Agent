"""政策文件解析与规则抽取。"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class PolicyItem:
    category: str
    description: str
    trigger_keywords: list[str]


@dataclass
class PolicyRuleSet:
    policies: list[PolicyItem]

    def match(self, text: str) -> Optional[PolicyItem]:
        for item in self.policies:
            if any(keyword in text for keyword in item.trigger_keywords):
                return item
        return None

    def to_dict(self) -> dict:
        return {"policies": [asdict(item) for item in self.policies]}


def _parse_json_policy(content: str) -> PolicyRuleSet:
    data = json.loads(content)
    policies: list[PolicyItem] = []
    for item in data if isinstance(data, list) else data.get("policies", []):
        policies.append(
            PolicyItem(
                category=item.get("category", "未分类"),
                description=item.get("description", ""),
                trigger_keywords=item.get("trigger_keywords", []) or [],
            )
        )
    return PolicyRuleSet(policies=policies)


def _parse_text_policy(content: str) -> PolicyRuleSet:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    policies: list[PolicyItem] = []
    for idx, line in enumerate(lines):
        policies.append(PolicyItem(category=f"规则{idx + 1}", description=line, trigger_keywords=line.split()[:3]))
    return PolicyRuleSet(policies=policies)


def load_and_extract_policy_rules(path: Path) -> PolicyRuleSet:
    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return _parse_json_policy(content)
    return _parse_text_policy(content)
