"""售后 Agent Web 演示入口。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAI
from pydantic import BaseModel, Field
from starlette.requests import Request

from src.agent.agent import OmniAgent
from src.agent.orchestrator import LLMFirstOrchestrator
from src.agent.router import ASRouter, RouterContext
from src.agent.workflow import AfterSalesWorkflow
from src.policy.ingestion import build_ingestion_plan
from src.policy.policy_loader import PolicyRuleSet, load_and_extract_policy_rules
from src.policy.policy_store import PolicyStore

BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "data" / "policies" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="企业售后智能体演示", version="1.0.0")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

policy_store = PolicyStore()


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户输入消息")
    policy_id: Optional[str] = Field(default=None, description="指定使用的政策ID；不传则默认最新政策")
    session_id: Optional[str] = Field(default=None, description="会话ID，用于前端多轮对话")


class ChatResponse(BaseModel):
    session_id: str
    policy_id: Optional[str]
    route: dict[str, Any]
    extraction: dict[str, Any]
    retrieval: Optional[dict[str, Any]]
    graph_result: Optional[dict[str, Any]]
    tool_trace: dict[str, Any]
    answer: str


_SESSIONS: dict[str, OmniAgent] = {}


def _get_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("请先配置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY")
    base_url = os.getenv("DASHSCOPE_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    return OpenAI(api_key=api_key, base_url=base_url)


def _get_model_name() -> str:
    return os.getenv("CHAT_MODEL", "qwen-plus")


def _get_agent(session_id: str, policy_rules: Optional[PolicyRuleSet] = None) -> OmniAgent:
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = OmniAgent(client=_get_client(), model_name=_get_model_name(), policy_rules=policy_rules)
    else:
        _SESSIONS[session_id].policy_rules = policy_rules
        _SESSIONS[session_id].router = ASRouter(policy_rules=policy_rules)
    return _SESSIONS[session_id]


def _resolve_policy_rules(policy_id: Optional[str]) -> tuple[Optional[Any], Optional[PolicyRuleSet]]:
    policy_record = None
    policy_rules: Optional[PolicyRuleSet] = None
    if policy_id:
        policy_record = policy_store.get(policy_id)
        if not policy_record:
            raise HTTPException(status_code=404, detail=f"未找到 policy_id={policy_id}")
        policy_rules = policy_store.load_rule_set(policy_id)
    else:
        policy_record = policy_store.latest()
        if policy_record:
            policy_rules = policy_store.load_rule_set(policy_record.policy_id)
    return policy_record, policy_rules


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    latest = policy_store.latest()
    records = policy_store.list_records()
    return templates.TemplateResponse("index.html", {"request": request, "latest_policy": latest, "policies": records})


@app.get("/api/policies")
def list_policies() -> JSONResponse:
    data = []
    for item in policy_store.list_records():
        rule_set = policy_store.load_rule_set(item.policy_id)
        data.append({**item.__dict__, "rule_set": rule_set.to_dict() if rule_set else None})
    return JSONResponse({"policies": data})


@app.post("/api/policies/upload")
async def upload_policy(file: UploadFile = File(...)) -> JSONResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".txt", ".md", ".json"}:
        raise HTTPException(status_code=400, detail="仅支持 txt / md / json 政策文件，PDF 可在后续扩展")

    file_id = uuid4().hex[:12]
    save_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    content = await file.read()
    save_path.write_bytes(content)

    plan = build_ingestion_plan(save_path)
    rule_set = load_and_extract_policy_rules(save_path)
    record = policy_store.register(file_name=file.filename, rule_set=rule_set, source_path=str(save_path))
    return JSONResponse({"ok": True, "policy_id": record.policy_id, "file_name": record.file_name, "created_at": record.created_at, "rule_count": len(rule_set.policies), "rule_set": rule_set.to_dict(), "ingestion_plan": plan.__dict__})


@app.get("/api/policies/{policy_id}")
def get_policy(policy_id: str) -> JSONResponse:
    record = policy_store.get(policy_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"未找到 policy_id={policy_id}")
    rule_set = policy_store.load_rule_set(policy_id)
    return JSONResponse({"record": record.__dict__, "rule_set": rule_set.to_dict() if rule_set else None})


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    policy_record, policy_rules = _resolve_policy_rules(payload.policy_id)
    session_id = payload.session_id or uuid4().hex[:12]
    agent = _get_agent(session_id, policy_rules=policy_rules)
    orchestrator = LLMFirstOrchestrator(agent)
    result = orchestrator.run(payload.message)
    route = result.route
    tool_trace = {
        "route": route.__dict__,
        "policy_context": route.context_hint,
        "policy_fragment": route.matched_policy_description,
        "bundle": result.agent_result.tool_bundle.to_dict(),
        "state": result.agent_result.state.value,
        "tool_rounds": result.agent_result.tool_rounds,
        "react_trace": [item.to_dict() for item in result.agent_result.trace],
        "extraction": result.extraction.to_dict(),
        "retrieval": result.retrieval.to_dict() if result.retrieval else None,
        "graph_result": result.graph_result.to_dict() if result.graph_result else None,
        "needs_human": result.needs_human,
    }
    return ChatResponse(
        session_id=session_id,
        policy_id=policy_record.policy_id if policy_record else None,
        route=route.__dict__,
        extraction=result.extraction.to_dict(),
        retrieval=result.retrieval.to_dict() if result.retrieval else None,
        graph_result=result.graph_result.to_dict() if result.graph_result else None,
        tool_trace=tool_trace,
        answer=result.answer,
    )


@app.get("/api/chat/stream")
def chat_stream(message: str, policy_id: Optional[str] = None, session_id: Optional[str] = None) -> StreamingResponse:
    policy_record, policy_rules = _resolve_policy_rules(policy_id)
    real_session_id = session_id or uuid4().hex[:12]
    agent = _get_agent(real_session_id, policy_rules=policy_rules)
    workflow = AfterSalesWorkflow(agent)
    route = workflow.router.route(RouterContext(user_input=message, policy_rules=policy_rules))

    def event_stream():
        yield f"event: route\ndata: {json.dumps(route.__dict__, ensure_ascii=False)}\n\n"
        yield f"event: policy\ndata: {json.dumps({'policy_id': policy_record.policy_id if policy_record else None, 'policy_context': route.context_hint, 'policy_fragment': route.matched_policy_description}, ensure_ascii=False)}\n\n"
        result = workflow.run(message)
        yield f"event: extraction\ndata: {json.dumps(result.extraction.to_dict(), ensure_ascii=False)}\n\n"
        yield f"event: retrieval\ndata: {json.dumps(result.retrieval.to_dict() if result.retrieval else None, ensure_ascii=False)}\n\n"
        yield f"event: tool_trace\ndata: {json.dumps({'bundle': result.agent_result.tool_bundle.to_dict(), 'state': result.agent_result.state.value, 'tool_rounds': result.agent_result.tool_rounds, 'react_trace': [item.to_dict() for item in result.agent_result.trace]}, ensure_ascii=False)}\n\n"
        final_answer = result.agent_result.answer
        if result.meta.get("answer_prefix"):
            final_answer = result.meta["answer_prefix"] + final_answer
        yield f"event: message\ndata: {json.dumps({'session_id': real_session_id, 'policy_id': policy_record.policy_id if policy_record else None, 'answer': final_answer}, ensure_ascii=False)}\n\n"
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/debug/state")
def debug_state() -> JSONResponse:
    return JSONResponse({"sessions": list(_SESSIONS.keys())})
