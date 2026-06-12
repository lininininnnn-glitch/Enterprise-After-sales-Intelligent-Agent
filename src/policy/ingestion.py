"""政策知识摄入分级与入库计划。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


PolicyComplexity = Literal["simple", "complex"]


@dataclass
class IngestionPlan:
    complexity: PolicyComplexity
    use_small_model: bool
    use_langchain_split: bool
    write_to_es: bool
    write_to_milvus: bool
    write_to_neo4j: bool
    reason: str


_SIMPLE_HINTS = ["退货", "退款", "发票", "保修", "换货", "7天", "15天", "30天", "会员权益"]
_COMPLEX_HINTS = ["如果", "否则", "例外", "同时", "并且", "且", "多条件", "跨段", "依赖", "嵌套", "但", "除外"]


def detect_policy_complexity(content: str) -> PolicyComplexity:
    text = content.strip()
    if len(text) > 1200:
        return "complex"
    simple_score = sum(1 for hint in _SIMPLE_HINTS if hint in text)
    complex_score = sum(1 for hint in _COMPLEX_HINTS if hint in text)
    if complex_score >= 2 or len(text.splitlines()) > 30:
        return "complex"
    if simple_score >= 2 and complex_score == 0 and len(text) < 800:
        return "simple"
    return "simple" if len(text) < 500 and complex_score == 0 else "complex"


def build_ingestion_plan(path: Path) -> IngestionPlan:
    content = path.read_text(encoding="utf-8")
    complexity = detect_policy_complexity(content)
    if complexity == "simple":
        return IngestionPlan(
            complexity=complexity,
            use_small_model=False,
            use_langchain_split=True,
            write_to_es=True,
            write_to_milvus=True,
            write_to_neo4j=False,
            reason="政策文本较短、条款较独立，适合直接切分并写入检索库。",
        )
    return IngestionPlan(
        complexity=complexity,
        use_small_model=True,
        use_langchain_split=True,
        write_to_es=True,
        write_to_milvus=True,
        write_to_neo4j=True,
        reason="政策包含多条件、例外或依赖关系，建议小模型增强摄入并写入图谱。",
    )
