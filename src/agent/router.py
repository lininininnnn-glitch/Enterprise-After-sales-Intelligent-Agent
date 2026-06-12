"""售后 Agent 路由器。

职责：
1. 根据用户输入、政策规则和上下文生成路由建议；
2. 不直接替代模型决策，但会给出可解释的先验；
3. 为模型提供政策片段、关键词、意图分层和工具建议。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from src.policy.policy_loader import PolicyRuleSet


@dataclass
class RouteDecision:
    is_after_sales: bool = False
    intent: str = "general"
    confidence: float = 0.0
    needs_rag: bool = False
    needs_tool_call: bool = False
    needs_graph: bool = False
    needs_extraction: bool = False
    route_to_llm: bool = False
    simple_query_only: bool = False
    matched_policy_category: Optional[str] = None
    matched_policy_description: Optional[str] = None
    matched_keywords: list[str] = field(default_factory=list)
    context_hint: str = ""
    suggested_tools: list[str] = field(default_factory=list)


KEYWORDS_AFTER_SALES = ["退款", "退货", "换货", "保修", "维修", "投诉", "售后", "发票", "破损", "发错", "质量问题", "换新"]
KEYWORDS_POLICY = ["规则", "政策", "标准", "条件", "怎么办", "是否可以", "怎么处理", "能不能", "是否能"]
KEYWORDS_GRAPH = ["关系", "流转", "约束", "影响", "依赖", "组合", "包含", "多跳"]
KEYWORDS_EXTRACTION = ["订单号", "手机号", "用户ID", "会员", "金额", "时间", "地址", "商品", "原因"]


@dataclass
class RouterContext:
    user_input: str
    user_id: Optional[str] = None
    order_id: Optional[str] = None
    vip_level: Optional[str] = None
    policy_rules: Optional[PolicyRuleSet] = None
    history_summary: Optional[str] = None
    llm_decision: Optional[dict[str, Any]] = None


class ASRouter:
    def __init__(self, policy_rules: Optional[PolicyRuleSet] = None):
        self.policy_rules = policy_rules

    def _score_intent(self, text: str) -> tuple[str, float, list[str]]:
        matched_keywords = [k for k in KEYWORDS_AFTER_SALES + KEYWORDS_POLICY + KEYWORDS_GRAPH + KEYWORDS_EXTRACTION if k in text]
        score = len(matched_keywords)
        if any(k in text for k in KEYWORDS_AFTER_SALES):
            intent = "after_sales"
        elif any(k in text for k in KEYWORDS_POLICY):
            intent = "policy_qa"
        elif any(k in text for k in KEYWORDS_GRAPH):
            intent = "graph_reasoning"
        elif any(k in text for k in KEYWORDS_EXTRACTION):
            intent = "structured_extraction"
        else:
            intent = "general"
        confidence = min(0.95, 0.2 + score * 0.12)
        return intent, confidence, matched_keywords

    def _rule_route(self, ctx: RouterContext) -> RouteDecision:
        text = ctx.user_input
        intent, confidence, matched_keywords = self._score_intent(text)
        decision = RouteDecision()
        decision.intent = intent
        decision.confidence = confidence
        decision.is_after_sales = intent != "general"
        decision.needs_extraction = intent in {"after_sales", "structured_extraction"}
        decision.needs_rag = intent in {"after_sales", "policy_qa", "graph_reasoning"}
        decision.needs_graph = intent == "graph_reasoning" or any(k in text for k in KEYWORDS_GRAPH)
        decision.simple_query_only = intent == "general" and not any(ch in text for ch in ("怎么办", "是否", "能不能", "为什么", "如何", "原因", "处理"))
        decision.route_to_llm = not decision.simple_query_only
        decision.needs_tool_call = decision.route_to_llm and (decision.is_after_sales or bool(ctx.order_id or ctx.user_id or ctx.vip_level))
        decision.matched_keywords = matched_keywords

        if decision.needs_tool_call:
            suggested = ["search_company_policy"]
            if ctx.user_id:
                suggested.append("get_user_profile")
            if ctx.order_id:
                suggested.extend(["get_order_info", "check_refund_eligibility"])
            if ctx.vip_level:
                suggested.append("get_vip_policy")
            if decision.needs_graph:
                suggested.append("query_knowledge_graph")
            decision.suggested_tools = list(dict.fromkeys(suggested))

        if self.policy_rules:
            matched = self.policy_rules.match(text)
            if matched:
                decision.matched_policy_category = matched.category
                decision.matched_policy_description = matched.description
                hints = [f"已匹配政策规则: {matched.category}"]
                if matched.description:
                    hints.append(f"规则描述: {matched.description}")
                if matched.trigger_keywords:
                    hints.append(f"命中关键词: {', '.join(matched.trigger_keywords)}")
                decision.context_hint = "\n".join(hints)
            else:
                decision.context_hint = "未命中具体政策规则，但可参考售后政策文件中的通用规则。"
        else:
            decision.context_hint = "当前未加载政策文件，可根据通用售后知识库和工具信息进行判断。"
        return decision

    def route(self, ctx: RouterContext) -> RouteDecision:
        return self._rule_route(ctx)

    def route_to_json(self, decision: RouteDecision) -> str:
        return json.dumps(decision.__dict__, ensure_ascii=False)
