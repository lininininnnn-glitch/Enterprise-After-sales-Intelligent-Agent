"""政策规则版本存储。"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from uuid import uuid4

from src.policy.policy_loader import PolicyRuleSet, PolicyItem


@dataclass
class PolicyRecord:
    policy_id: str
    file_name: str
    created_at: str
    source_path: str
    rule_set_path: str


class PolicyStore:
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(__file__).resolve().parents[2] / "data" / "policies"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "index.json"
        if not self.index_path.exists():
            self.index_path.write_text(json.dumps([], ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_index(self) -> list[dict]:
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _save_index(self, records: list[dict]) -> None:
        self.index_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    def register(self, file_name: str, rule_set: PolicyRuleSet, source_path: str) -> PolicyRecord:
        policy_id = uuid4().hex[:12]
        created_at = __import__("datetime").datetime.now().isoformat(timespec="seconds")
        rule_set_path = self.base_dir / f"{policy_id}.json"
        rule_set_path.write_text(json.dumps(rule_set.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        record = PolicyRecord(policy_id=policy_id, file_name=file_name, created_at=created_at, source_path=source_path, rule_set_path=str(rule_set_path))
        records = self._load_index()
        records.append(asdict(record))
        self._save_index(records)
        return record

    def list_records(self) -> list[PolicyRecord]:
        return [PolicyRecord(**item) for item in self._load_index()]

    def latest(self) -> Optional[PolicyRecord]:
        records = self.list_records()
        return records[-1] if records else None

    def get(self, policy_id: str) -> Optional[PolicyRecord]:
        for record in self.list_records():
            if record.policy_id == policy_id:
                return record
        return None

    def load_rule_set(self, policy_id: str) -> Optional[PolicyRuleSet]:
        record = self.get(policy_id)
        if not record:
            return None
        data = json.loads(Path(record.rule_set_path).read_text(encoding="utf-8"))
        return PolicyRuleSet(policies=[PolicyItem(**item) for item in data.get("policies", [])])
