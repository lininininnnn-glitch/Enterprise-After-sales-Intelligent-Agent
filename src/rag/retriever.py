"""混合 RAG 检索器。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from elasticsearch import Elasticsearch

from src.tools.tools import embed_text


@dataclass
class RetrievalResult:
    query: str
    keyword_hits: list[dict[str, Any]] = field(default_factory=list)
    vector_hits: list[dict[str, Any]] = field(default_factory=list)
    merged_context: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "keyword_hits": self.keyword_hits,
            "vector_hits": self.vector_hits,
            "merged_context": self.merged_context,
            "errors": self.errors,
        }


class HybridRetriever:
    def __init__(self, es_url: str | None = None, es_index: str = "company_index", milvus_db_path: str | None = None, milvus_collection: str = "company_vectors"):
        self.es_url = es_url or os.getenv("ES_URL", "http://localhost:9200")
        self.es_index = es_index
        self.milvus_db_path = milvus_db_path or "./milvus_demo.db"
        self.milvus_collection = milvus_collection

    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        result = RetrievalResult(query=query)
        seen: set[str] = set()

        try:
            es = Elasticsearch(self.es_url)
            if es.indices.exists(index=self.es_index):
                response = es.search(index=self.es_index, size=top_k, query={"match": {"content": {"query": query}}})
                for hit in response.get("hits", {}).get("hits", []):
                    source = hit.get("_source", {})
                    content = source.get("content", "")
                    if content and content not in seen:
                        seen.add(content)
                        result.keyword_hits.append({"score": hit.get("_score", 0), "content": content, "source": source.get("source", "")})
        except Exception as exc:
            result.errors.append(f"ES 检索失败: {exc}")

        try:
            from pymilvus import MilvusClient

            vector = embed_text(query)
            client = MilvusClient(self.milvus_db_path)
            if client.has_collection(self.milvus_collection):
                items = client.search(collection_name=self.milvus_collection, data=[vector], limit=top_k, output_fields=["content", "chunk_id"])
                for item in items[0]:
                    entity = item.get("entity", {})
                    content = entity.get("content", "")
                    if content and content not in seen:
                        seen.add(content)
                        result.vector_hits.append({"distance": item.get("distance", 0), "content": content, "chunk_id": entity.get("chunk_id", "")})
        except Exception as exc:
            result.errors.append(f"Milvus 检索失败: {exc}")

        contexts = []
        for idx, hit in enumerate(result.keyword_hits + result.vector_hits, start=1):
            content = hit.get("content", "")
            if content:
                contexts.append(f"{idx}. {content}")
        result.merged_context = "\n".join(contexts) if contexts else "未检索到有效知识片段。"
        return result
