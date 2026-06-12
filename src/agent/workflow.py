"""将抽取、路由、RAG 和工具调用串联的统一工作流。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.agent.agent import OmniAgent, AgentRunResult
from src.agent.extractor import ExtractionResult, extract_keywords, relaxed_json_parse, strict_json_parse
from src.rag.graph_rag import GraphRAGRetriever, GraphResult
from src.rag.retriever import HybridRetriever, RetrievalResult
from src.agent.router import ASRouter, RouterContext, RouteDecision


@dataclass
class WorkflowResult:
    extraction: ExtractionResult
    route: RouteDecision
    retrieval: Optional[RetrievalResult]
    graph_result: Optional[GraphResult]
    agent_result: AgentRunResult
    meta: dict[str, Any] = field(default_factory=dict)


class AfterSalesWorkflow:
    def __init__(self, agent: OmniAgent, retriever: Optional[HybridRetriever] = None):
        self.agent = agent
        self.router = ASRouter(policy_rules=agent.policy_rules)
        self.retriever = retriever or HybridRetriever()

    def run(self, user_input: str, max_tool_rounds: int = 4) -> WorkflowResult:
        extraction = extract_keywords(user_input)
        route = self.router.route(
            RouterContext(
                user_input=user_input,
                user_id=self.agent._extract_user_id(user_input),
                order_id=extraction.order_id or self.agent._extract_order_id(user_input),
                vip_level=extraction.vip_level or self.agent._extract_vip_level(user_input),
                policy_rules=self.agent.policy_rules,
            )
        )

        retrieval: Optional[RetrievalResult] = None
        graph_result: Optional[GraphResult] = None
        if route.needs_rag:
            retrieval = self.retriever.retrieve(user_input)
        if route.needs_graph:
            graph_entity = extraction.product or extraction.issue or extraction.intent or user_input
            graph_result = GraphRAGRetriever().retrieve(graph_entity)

        answer_prefix = ""
        if retrieval and retrieval.merged_context:
            answer_prefix += f"以下是检索到的参考信息：\n{retrieval.merged_context}\n\n"
        if graph_result and graph_result.summary:
            answer_prefix += f"以下是图谱召回信息：\n{graph_result.summary}\n\n"

        agent_input = user_input
        if answer_prefix:
            agent_input = f"{answer_prefix}\n用户问题：{user_input}"
        agent_result = self.agent.run(agent_input, max_tool_rounds=max_tool_rounds)
        meta = {
            "strict_json": strict_json_parse(user_input),
            "relaxed_json": relaxed_json_parse(user_input),
            "answer_prefix": answer_prefix,
        }
        return WorkflowResult(extraction=extraction, route=route, retrieval=retrieval, graph_result=graph_result, agent_result=agent_result, meta=meta)
