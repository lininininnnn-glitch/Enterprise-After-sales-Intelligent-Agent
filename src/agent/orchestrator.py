"""复杂任务总调度器：先路由，再由大模型主导，再由小模型抽取与工具调用。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.agent.agent import AgentRunResult, OmniAgent
from src.agent.extractor import ExtractionResult, extract_keywords
from src.agent.router import ASRouter, RouterContext, RouteDecision
from src.agent.validator import build_fix_prompt, validate_answer
from src.rag.graph_rag import GraphRAGRetriever, GraphResult
from src.rag.retriever import HybridRetriever, RetrievalResult


@dataclass
class OrchestratorResult:
    route: RouteDecision
    extraction: ExtractionResult
    retrieval: Optional[RetrievalResult]
    graph_result: Optional[GraphResult]
    agent_result: AgentRunResult
    answer: str
    needs_human: bool = False


class LLMFirstOrchestrator:
    def __init__(self, agent: OmniAgent, retriever: Optional[HybridRetriever] = None):
        self.agent = agent
        self.router = ASRouter(policy_rules=agent.policy_rules)
        self.retriever = retriever or HybridRetriever()

    def run(self, user_input: str) -> OrchestratorResult:
        route = self.router.route(RouterContext(user_input=user_input, policy_rules=self.agent.policy_rules))
        if route.simple_query_only:
            # 简单任务直接由规则/数据库层处理，不进入复杂大模型链路
            extraction = extract_keywords(user_input)
            agent_result = self.agent.run(user_input, max_tool_rounds=1, max_answer_retries=1)
            return OrchestratorResult(route=route, extraction=extraction, retrieval=None, graph_result=None, agent_result=agent_result, answer=agent_result.answer)

        # 复杂任务交给大模型主导
        extraction = extract_keywords(user_input)
        retrieval = self.retriever.retrieve(user_input) if route.needs_rag else None
        graph_result = GraphRAGRetriever().retrieve(extraction.issue or extraction.intent or user_input) if route.needs_graph else None

        # 先把检索结果回填到 agent 的上下文中
        prefix_parts = []
        if retrieval and retrieval.merged_context:
            prefix_parts.append(f"检索上下文：\n{retrieval.merged_context}")
        if graph_result and graph_result.summary:
            prefix_parts.append(f"图谱上下文：\n{graph_result.summary}")
        if extraction.to_dict():
            prefix_parts.append(f"抽取结果：{extraction.to_dict()}")
        if prefix_parts:
            self.agent.messages.append({"role": "system", "content": "\n\n".join(prefix_parts)})

        agent_result = self.agent.run(user_input, max_tool_rounds=4, max_answer_retries=3)
        validation = validate_answer(agent_result.answer)
        needs_human = not validation.ok and agent_result.state.name == "ERROR"
        if needs_human:
            agent_result.answer = "请联系人工客服进一步处理。"
        return OrchestratorResult(route=route, extraction=extraction, retrieval=retrieval, graph_result=graph_result, agent_result=agent_result, answer=agent_result.answer, needs_human=needs_human)
