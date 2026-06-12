"""离线评测指标。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractionSample:
    input_text: str
    expected: dict[str, Any]
    predicted: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricReport:
    exact_match: float
    field_recall: float
    key_consistency: float
    total: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact_match": self.exact_match,
            "field_recall": self.field_recall,
            "key_consistency": self.key_consistency,
            "total": self.total,
        }


def exact_match_score(expected: dict[str, Any], predicted: dict[str, Any]) -> float:
    return 1.0 if expected == predicted else 0.0


def field_recall_score(expected: dict[str, Any], predicted: dict[str, Any]) -> float:
    if not expected:
        return 1.0
    hits = 0
    total = 0
    for key, value in expected.items():
        total += 1
        if predicted.get(key) == value and value not in (None, ""):
            hits += 1
    return hits / total if total else 1.0


def key_consistency_score(expected: dict[str, Any], predicted: dict[str, Any]) -> float:
    if not expected:
        return 1.0
    expected_keys = {k for k, v in expected.items() if v not in (None, "")}
    predicted_keys = {k for k, v in predicted.items() if v not in (None, "")}
    if not expected_keys:
        return 1.0
    return len(expected_keys & predicted_keys) / len(expected_keys)


def evaluate(samples: list[ExtractionSample]) -> MetricReport:
    if not samples:
        return MetricReport(exact_match=0.0, field_recall=0.0, key_consistency=0.0, total=0)

    exact_total = 0.0
    recall_total = 0.0
    key_total = 0.0
    for sample in samples:
        exact_total += exact_match_score(sample.expected, sample.predicted)
        recall_total += field_recall_score(sample.expected, sample.predicted)
        key_total += key_consistency_score(sample.expected, sample.predicted)

    total = len(samples)
    return MetricReport(
        exact_match=exact_total / total,
        field_recall=recall_total / total,
        key_consistency=key_total / total,
        total=total,
    )


def load_samples(path: str) -> list[ExtractionSample]:
    samples: list[ExtractionSample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            samples.append(ExtractionSample(input_text=obj["input_text"], expected=obj["expected"], predicted=obj.get("predicted", {})))
    return samples
