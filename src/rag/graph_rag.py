"""GraphRAG 风格的知识图谱召回辅助模块。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from neo4j import GraphDatabase


@dataclass
class GraphResult:
    entity_name: str
    triples: list[dict[str, Any]] = field(default_factory=list)
    expanded_entities: list[str] = field(default_factory=list)
    summary: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_name": self.entity_name,
            "triples": self.triples,
            "expanded_entities": self.expanded_entities,
            "summary": self.summary,
            "errors": self.errors,
        }


class GraphRAGRetriever:
    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None):
        self.uri = uri or os.getenv("NEO4J_URI", "neo4j+ssc://54d3767b.databases.neo4j.io")
        self.user = user or os.getenv("NEO4J_USER", "54d3767b")
        self.password = password or os.getenv("NEO4J_PASSWORD", "")

    def retrieve(self, entity_name: str, depth: int = 1) -> GraphResult:
        result = GraphResult(entity_name=entity_name)
        query = """
        MATCH (n:Entity)
        WHERE n.name = $entity_name
        OPTIONAL MATCH (n)-[r:RELATED_TO]->(m:Entity)
        RETURN n.name AS source, r.type AS relation, m.name AS target
        UNION
        MATCH (m:Entity)-[r:RELATED_TO]->(n:Entity)
        WHERE n.name = $entity_name
        RETURN m.name AS source, r.type AS relation, n.name AS target
        """

        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            with driver.session() as session:
                records = session.run(query, entity_name=entity_name)
                for record in records:
                    source = record.get("source")
                    relation = record.get("relation")
                    target = record.get("target")
                    if source and relation and target:
                        result.triples.append({"source": source, "relation": relation, "target": target})
                        result.expanded_entities.extend([source, target])
            driver.close()
        except Exception as exc:
            result.errors.append(f"GraphDB 查询失败: {exc}")

        result.expanded_entities = list(dict.fromkeys([item for item in result.expanded_entities if item]))
        if result.triples:
            lines = [f"({item['source']}) -[{item['relation']}]-> ({item['target']})" for item in result.triples[:20]]
            result.summary = "知识图谱召回逻辑如下：\n" + "\n".join(lines)
        else:
            result.summary = f"知识图谱中未查到与实体 '{entity_name}' 相关的逻辑关系。"
        return result
