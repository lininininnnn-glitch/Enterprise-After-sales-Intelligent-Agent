"""Agent 核心工作流。

实现能力：
- Tool Calling
- 多轮对话
- RAG
- SSE 配合上层 API
- 结构化输出
- ReAct trace
- 任务状态流转
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from src.policy.policy_loader import PolicyRuleSet
from src.agent.router import ASRouter, RouteDecision, RouterContext
from src.agent.validator import build_fix_prompt, validate_answer
from src.tools.tools import TOOL_INPUT_MODELS, TOOL_REGISTRY, TOOL_SPECS

SYSTEM_PROMPT = (
    "你是一个严谨、专业的企业售后智能助理。"
    "你必须优先使用工具获取事实，不要凭空编造订单、会员或政策信息。"
    "你需要遵循 ReAct 风格：先分析是否需要抽取、检索、图谱或工具调用，再回答。"
    "当信息不足时，继续调用合适的工具；如果仍然不足，需要明确说明。"
)


class AgentState(str, Enum):
    INIT = "INIT"
    ROUTED = "ROUTED"
    MODEL_THINKING = "MODEL_THINKING"
    TOOL_RUNNING = "TOOL_RUNNING"
    ANSWERING = "ANSWERING"
    DONE = "DONE"
    ERROR = "ERROR"


@dataclass
class ReActTraceStep:
    step: int
    state: AgentState
    kind: str
    content: Any

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "state": self.state.value, "kind": self.kind, "content": self.content}


@dataclass
class ToolBundle:
    user_profile: Optional[dict] = None
    order_info: Optional[dict] = None
    vip_policy: Optional[dict] = None
    policy_search: Optional[str] = None
    graph_result: Optional[str] = None
    refund_eligibility: Optional[str] = None

    def has_any(self) -> bool:
        return any([self.user_profile, self.order_info, self.vip_policy, self.policy_search, self.graph_result, self.refund_eligibility])

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_profile": self.user_profile,
            "order_info": self.order_info,
            "vip_policy": self.vip_policy,
            "policy_search": self.policy_search,
            "graph_result": self.graph_result,
            "refund_eligibility": self.refund_eligibility,
        }


@dataclass
class AgentRunResult:
    answer: str
    state: AgentState
    route_hint: dict[str, Any]
    tool_bundle: ToolBundle
    trace: list[ReActTraceStep] = field(default_factory=list)
    tool_rounds: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "state": self.state.value,
            "route_hint": self.route_hint,
            "tool_bundle": self.tool_bundle.to_dict(),
            "trace": [item.to_dict() for item in self.trace],
            "tool_rounds": self.tool_rounds,
        }


@dataclass
class OmniAgent:
    client: OpenAI
    model_name: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    max_history_messages: int = 12
    policy_rules: Optional[PolicyRuleSet] = None

    def __post_init__(self) -> None:
        self.router = ASRouter(policy_rules=self.policy_rules)
        if not self.messages:
            self.messages.append({"role": "system", "content": SYSTEM_PROMPT})

    def _trim_history(self) -> None:
        if len(self.messages) <= self.max_history_messages:
            return
        self.messages = self.messages[:1] + self.messages[-(self.max_history_messages - 1):]

    def _append_user_message(self, user_input: str) -> None:
        self.messages.append({"role": "user", "content": user_input})
        self._trim_history()

    def _extract_user_id(self, text: str) -> Optional[str]:
        match = re.search(r"U\d{4,}", text)
        return match.group(0) if match else None

    def _extract_order_id(self, text: str) -> Optional[str]:
        match = re.search(r"\b\d{3,}\b", text)
        return match.group(0) if match else None

    def _extract_vip_level(self, text: str) -> Optional[str]:
        for level in ("SVIP", "黑钻会员", "白金VIP", "黄金VIP", "普通用户"):
            if level in text:
                return level
        return None

    def _route_context(self, user_input: str) -> Tuple[RouteDecision, RouterContext]:
        ctx = RouterContext(
            user_input=user_input,
            user_id=self._extract_user_id(user_input),
            order_id=self._extract_order_id(user_input),
            vip_level=self._extract_vip_level(user_input),
            policy_rules=self.policy_rules,
        )
        return self.router.route(ctx), ctx

    def _build_context_message(self, decision: RouteDecision, bundle: ToolBundle) -> str:
        payload = {
            "route_hint": decision.__dict__,
            "policy_context": decision.context_hint,
            "policy_fragment": decision.matched_policy_description,
            "tool_bundle": bundle.to_dict(),
        }
        return (
            "你是一个企业售后智能助理。下面提供的是路由提示、政策片段和系统已获取的事实信息。\n"
            "请按如下步骤工作：1）识别是否需要结构化抽取；2）必要时调用检索或图谱工具；3）必要时调用业务工具；4）基于工具结果生成最终回答。\n"
            "路由提示只供参考，不替代你的判断。请你通过 function calling 自主决定是否查询用户、订单、会员权益、知识库或知识图谱。\n"
            "如果事实不足，请继续调用工具；如果信息足够，请直接生成最终售后回复。\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )

    def _call_model(self, allow_tools: bool = True, extra_system: str = ""):
        request_messages = list(self.messages)
        if extra_system:
            request_messages = [{"role": "system", "content": extra_system}] + request_messages
        request_kwargs = {"model": self.model_name, "messages": request_messages, "temperature": 0.2}
        if allow_tools:
            request_kwargs["tools"] = TOOL_SPECS
            request_kwargs["tool_choice"] = "auto"
        return self.client.chat.completions.create(**request_kwargs)

    def _merge_assistant_message(self, assistant_message: Any) -> None:
        message: Dict[str, Any] = {"role": "assistant", "content": assistant_message.content or ""}
        if assistant_message.tool_calls:
            normalized_calls = []
            for call in assistant_message.tool_calls:
                normalized_calls.append(
                    {
                        "id": getattr(call, "id", None),
                        "type": getattr(call, "type", "function"),
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                )
            message["tool_calls"] = normalized_calls
        self.messages.append(message)
        self._trim_history()

    def _execute_tool_call(self, tool_call: Any) -> tuple[Dict[str, Any], dict[str, Any]]:
        tool_name = tool_call.function.name
        raw_arguments = tool_call.function.arguments or "{}"
        model_cls = TOOL_INPUT_MODELS.get(tool_name)
        if model_cls is None:
            result = {"error": f"未知工具：{tool_name}"}
            return {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps({"tool_name": tool_name, "arguments": {}, "result": result}, ensure_ascii=False)}, {"tool_name": tool_name, "arguments": {}, "result": result}
        try:
            arguments_dict = json.loads(raw_arguments)
            validated = model_cls(**arguments_dict)
            result = TOOL_REGISTRY[tool_name](**validated.model_dump())
            payload = {"tool_name": tool_name, "arguments": arguments_dict, "result": result}
        except Exception as exc:
            payload = {"tool_name": tool_name, "arguments": raw_arguments, "result": {"error": str(exc)}}
        tool_message = {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(payload, ensure_ascii=False)}
        return tool_message, payload

    def _build_tool_bundle_from_recent_tools(self) -> ToolBundle:
        bundle = ToolBundle()
        for msg in reversed(self.messages):
            if msg.get("role") != "tool":
                continue
            try:
                payload = json.loads(msg.get("content", "{}"))
            except Exception:
                continue
            result = payload.get("result", payload)
            tool_name = payload.get("tool_name")
            if tool_name == "get_user_profile" and bundle.user_profile is None:
                bundle.user_profile = result
            elif tool_name == "get_order_info" and bundle.order_info is None:
                bundle.order_info = result
            elif tool_name == "get_vip_policy" and bundle.vip_policy is None:
                bundle.vip_policy = result
            elif tool_name == "check_refund_eligibility" and bundle.refund_eligibility is None:
                bundle.refund_eligibility = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            elif tool_name == "search_company_policy" and bundle.policy_search is None:
                bundle.policy_search = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            elif tool_name == "query_knowledge_graph" and bundle.graph_result is None:
                bundle.graph_result = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        return bundle

    def _should_force_tool_round(self, decision: RouteDecision, bundle: ToolBundle, user_input: str) -> bool:
        if decision.intent == "structured_extraction" and not bundle.order_info:
            return True
        if any(keyword in user_input for keyword in ("退款", "退货", "换货", "保修", "维修", "发票")) and not bundle.policy_search:
            return True
        if decision.needs_graph and not bundle.graph_result:
            return True
        return False

    def run(self, user_input: str, max_tool_rounds: int = 4, max_answer_retries: int = 3) -> AgentRunResult:
        trace: list[ReActTraceStep] = []
        step = 1
        state = AgentState.INIT
        trace.append(ReActTraceStep(step=step, state=state, kind="input", content=user_input))

        self._append_user_message(user_input)
        decision, _ = self._route_context(user_input)
        step += 1
        state = AgentState.ROUTED
        trace.append(ReActTraceStep(step=step, state=state, kind="route_hint", content=decision.__dict__))

        route_msg = self._build_context_message(decision, ToolBundle())
        state = AgentState.MODEL_THINKING
        step += 1
        trace.append(ReActTraceStep(step=step, state=state, kind="thought", content="模型基于用户输入、政策片段和路由提示判断是否需要调用工具。"))

        response = self._call_model(allow_tools=True, extra_system=route_msg)
        assistant_message = response.choices[0].message
        self._merge_assistant_message(assistant_message)

        tool_rounds = 0
        bundle = self._build_tool_bundle_from_recent_tools()
        while (assistant_message.tool_calls or self._should_force_tool_round(decision, bundle, user_input)) and tool_rounds < max_tool_rounds:
            tool_rounds += 1
            state = AgentState.TOOL_RUNNING
            if not assistant_message.tool_calls:
                suggested_tools = decision.suggested_tools or ["search_company_policy"]
                assistant_message.tool_calls = []
                for idx, tool_name in enumerate(suggested_tools[:2], start=1):
                    fake_call = type("FakeToolCall", (), {})()
                    fake_call.id = f"forced_{tool_rounds}_{idx}"
                    fake_call.function = type("FakeFunction", (), {})()
                    fake_call.function.name = tool_name
                    fake_call.function.arguments = json.dumps({"query": user_input, "entity_name": user_input, "order_id": self._extract_order_id(user_input) or "", "reason": user_input, "user_id": self._extract_user_id(user_input) or "", "vip_level": self._extract_vip_level(user_input) or "普通用户"}, ensure_ascii=False)
                    assistant_message.tool_calls.append(fake_call)
            for tool_call in assistant_message.tool_calls:
                step += 1
                trace.append(ReActTraceStep(step=step, state=state, kind="action", content={"tool_name": tool_call.function.name, "arguments": tool_call.function.arguments or "{}", "round": tool_rounds}))
                tool_message, tool_payload = self._execute_tool_call(tool_call)
                self.messages.append(tool_message)
                step += 1
                trace.append(ReActTraceStep(step=step, state=state, kind="observation", content=tool_payload))
            self._trim_history()

            bundle = self._build_tool_bundle_from_recent_tools()
            next_context = self._build_context_message(decision, bundle)
            state = AgentState.MODEL_THINKING
            step += 1
            trace.append(ReActTraceStep(step=step, state=state, kind="thought", content="工具结果已回填，模型继续判断是否需要更多工具或可以回答。"))
            next_response = self._call_model(allow_tools=True, extra_system=next_context)
            assistant_message = next_response.choices[0].message
            self._merge_assistant_message(assistant_message)
            bundle = self._build_tool_bundle_from_recent_tools()

        validation = validate_answer(assistant_message.content or "")
        retries = 0
        while not validation.ok and retries < max_answer_retries:
            retries += 1
            step += 1
            trace.append(ReActTraceStep(step=step, state=AgentState.MODEL_THINKING, kind="validation_fail", content={"reason": validation.reason, "retry": retries}))
            fix_prompt = build_fix_prompt(assistant_message.content or "", validation.reason)
            fix_messages = list(self.messages) + [{"role": "system", "content": fix_prompt}]
            fix_response = self.client.chat.completions.create(model=self.model_name, messages=fix_messages, temperature=0.2)
            assistant_message = fix_response.choices[0].message
            self._merge_assistant_message(assistant_message)
            validation = validate_answer(assistant_message.content or "")

        final_answer = assistant_message.content or ""
        final_state = AgentState.DONE if final_answer and validation.ok else AgentState.ERROR
        if final_state == AgentState.ERROR:
            final_answer = "抱歉，当前问题暂时无法自动解决，建议您联系人工客服进一步处理。"
        step += 1
        trace.append(ReActTraceStep(step=step, state=AgentState.ANSWERING, kind="final", content=final_answer))
        step += 1
        trace.append(ReActTraceStep(step=step, state=final_state, kind="state", content=final_state.value))

        return AgentRunResult(answer=final_answer, state=final_state, route_hint=decision.__dict__, tool_bundle=self._build_tool_bundle_from_recent_tools(), trace=trace, tool_rounds=tool_rounds)

    def chat(self, user_input: str, max_tool_rounds: int = 4) -> str:
        return self.run(user_input=user_input, max_tool_rounds=max_tool_rounds).answer
