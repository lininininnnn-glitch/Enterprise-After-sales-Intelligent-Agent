"""售后 Agent 路由器。

职责：
1. 根据用户输入和政策规则生成上下文提示；
2. 不直接替代模型决策；
3. 为模型提供政策片段、关键词与路由建议。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from src.policy.policy_loader import PolicyRuleSet


@dataclass
class RouteDecision:
    is_after_sales: bool = False
    matched_policy_category: Optional[str] = None
    matched_policy_description: Optional[str] = None
    matched_keywords: list[str] = field(default_factory=list)
    context_hint: str = ""


KEYWORDS_AFTER_SALES = ["退款", "退货", "换货", "保修", "维修", "投诉", "售后", "发票", "破损", "发错", "质量问题", "换新"]
KEYWORDS_POLICY = ["规则", "政策", "标准", "条件", "怎么办", "是否可以", "怎么处理"]
KEYWORDS_GRAPH = ["关系", "流转", "约束", "影响", "依赖", "组合", "包含"]


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

    def _rule_route(self, ctx: RouterContext) -> RouteDecision:
        text = ctx.user_input
        decision = RouteDecision()
        decision.is_after_sales = any(k in text for k in KEYWORDS_AFTER_SALES)
        matched_keywords = [k for k in KEYWORDS_AFTER_SALES + KEYWORDS_POLICY + KEYWORDS_GRAPH if k in text]
        if matched_keywords:
            decision.matched_keywords = matched_keywords

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
