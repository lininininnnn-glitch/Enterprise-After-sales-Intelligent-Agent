"""本地业务工具模块。"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from elasticsearch import Elasticsearch
from pydantic import BaseModel, Field
from pymilvus import MilvusClient

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

from neo4j import GraphDatabase

from src.services.business_db import get_order_info, get_user_profile, get_vip_policy


class UserProfileInput(BaseModel):
    user_id: str = Field(..., description="用户ID，例如 U1001")


class OrderInfoInput(BaseModel):
    order_id: str = Field(..., description="订单编号，例如 888、999")


class VipPolicyInput(BaseModel):
    vip_level: str = Field(..., description="会员等级，例如 普通用户、黄金VIP、白金VIP、黑钻会员、SVIP")


class RefundEligibilityInput(BaseModel):
    order_id: str = Field(..., description="订单编号")
    reason: str = Field(..., description="用户申请退款/退换货的原因")


class PolicySearchInput(BaseModel):
    query: str = Field(..., description="用户的自然语言问题，例如退换货政策、VIP 权益、发票规则等")


class GraphQueryInput(BaseModel):
    entity_name: str = Field(..., description="要查询的核心实体名称")


def check_refund_eligibility(order_id: str, reason: str) -> str:
    order = get_order_info(order_id)
    if not order.get("found"):
        return f"订单 {order_id} 不存在，无法判断退款资格，请先核对订单号。"
    if any(keyword in reason for keyword in ("质量问题", "破损")):
        return "符合无理由退换条件"
    return "需人工进一步审核"


def _get_embedding_client() -> Any:
    if OpenAI is None:
        raise RuntimeError("当前环境未安装 openai 库，无法执行向量检索。")
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("未找到嵌入模型所需 API Key，请配置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY。")
    base_url = os.getenv("DASHSCOPE_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    return OpenAI(api_key=api_key, base_url=base_url)


def embed_text(text: str, model_name: Optional[str] = None) -> List[float]:
    client = _get_embedding_client()
    embedding_model = model_name or os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
    response = client.embeddings.create(model=embedding_model, input=text)
    return response.data[0].embedding


def search_company_policy(query: str) -> str:
    results: List[str] = []
    seen = set()
    try:
        es_url = os.getenv("ES_URL", "http://localhost:9200")
        es = Elasticsearch(es_url)
        if not es.indices.exists(index="company_index"):
            return "知识库暂未构建或 company_index 不存在，请先运行 build_rag_db.py。"
        es_response = es.search(index="company_index", size=2, query={"match": {"content": {"query": query}}})
        for hit in es_response.get("hits", {}).get("hits", []):
            text = hit.get("_source", {}).get("content", "")
            if text and text not in seen:
                seen.add(text)
                results.append(text)
    except Exception as exc:
        results.append(f"[ES 检索失败：{exc}]")

    try:
        vector = embed_text(query)
        milvus = MilvusClient("./milvus_demo.db")
        if not milvus.has_collection("company_vectors"):
            return "知识库暂未构建或 company_vectors 不存在，请先运行 build_rag_db.py。"
        try:
            milvus.load_collection(collection_name="company_vectors")
            milvus_response = milvus.search(collection_name="company_vectors", data=[vector], limit=2, output_fields=["content"])
            for item in milvus_response[0]:
                text = item.get("entity", {}).get("content", "")
                if text and text not in seen:
                    seen.add(text)
                    results.append(text)
        except Exception as exc:
            results.append(f"[Milvus 检索失败：{exc}]")
    except Exception as exc:
        results.append(f"[Milvus 检索失败：{exc}]")

    if not results:
        return "知识库中未找到相关信息。"
    formatted = [f"{idx + 1}. {text}" for idx, text in enumerate(results[:4])]
    return "知识库参考信息：\n" + "\n".join(formatted)


NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+ssc://54d3767b.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "54d3767b")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")


def query_knowledge_graph(entity_name: str) -> str:
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        query = """
        MATCH (n:Entity {name: $entity_name})-[r:RELATED_TO]->(m:Entity)
        RETURN n.name AS source, r.type AS relation, m.name AS target
        UNION
        MATCH (m:Entity)-[r:RELATED_TO]->(n:Entity {name: $entity_name})
        RETURN m.name AS source, r.type AS relation, n.name AS target
        """
        results = []
        with driver.session() as session:
            records = session.run(query, entity_name=entity_name)
            for record in records:
                results.append(f"({record['source']}) -[{record['relation']}]-> ({record['target']})")
        driver.close()
        if not results:
            return f"知识图谱中未查到与实体 '{entity_name}' 相关的逻辑关系。"
        return "知识图谱召回逻辑如下：\n" + "\n".join(results)
    except Exception as e:
        return f"图谱数据库查询异常: {str(e)}"


TOOL_SPECS = [
    {"type": "function", "function": {"name": "get_user_profile", "description": "查询指定用户的基础信息和会员等级", "parameters": UserProfileInput.model_json_schema()}},
    {"type": "function", "function": {"name": "get_order_info", "description": "查询指定订单的状态、金额和备注信息", "parameters": OrderInfoInput.model_json_schema()}},
    {"type": "function", "function": {"name": "check_refund_eligibility", "description": "根据订单号和退款原因，判断是否符合退款/退换货条件", "parameters": RefundEligibilityInput.model_json_schema()}},
    {"type": "function", "function": {"name": "get_vip_policy", "description": "查询指定会员等级的售后权益和服务说明", "parameters": VipPolicyInput.model_json_schema()}},
    {"type": "function", "function": {"name": "search_company_policy", "description": "查询公司的退换货、VIP、发票、保修等内部政策规范", "parameters": PolicySearchInput.model_json_schema()}},
    {"type": "function", "function": {"name": "query_knowledge_graph", "description": "查询企业知识图谱", "parameters": GraphQueryInput.model_json_schema()}},
]

TOOL_REGISTRY: Dict[str, Callable[..., Any]] = {
    "get_user_profile": get_user_profile,
    "get_order_info": get_order_info,
    "get_vip_policy": get_vip_policy,
    "check_refund_eligibility": check_refund_eligibility,
    "search_company_policy": search_company_policy,
    "query_knowledge_graph": query_knowledge_graph,
}

TOOL_INPUT_MODELS: Dict[str, type[BaseModel]] = {
    "get_user_profile": UserProfileInput,
    "get_order_info": OrderInfoInput,
    "get_vip_policy": VipPolicyInput,
    "check_refund_eligibility": RefundEligibilityInput,
    "search_company_policy": PolicySearchInput,
    "query_knowledge_graph": GraphQueryInput,
}
