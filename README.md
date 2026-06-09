# 企业售后智能体系统

> 这是一个正在重构为标准目录结构的 Python 售后 Agent 项目。

## 目录结构

```text
agent6.8/
├── src/
│   ├── api/
│   ├── agent/
│   ├── tools/
│   ├── policy/
│   └── services/
├── templates/
├── data/
├── scripts/
└── README.md
```

## 说明

- `src/api/`：FastAPI 接口与 SSE 流式输出
- `src/agent/`：Agent 编排、多轮对话、ReAct trace、任务状态流转
- `src/tools/`：工具调用、RAG 检索、知识图谱、业务查询
- `src/policy/`：政策解析、规则抽取、版本管理
- `src/services/`：业务服务、数据库初始化、评测与训练辅助
- `templates/`：前端页面
- `data/`：政策文件、训练数据、评测结果等
- `scripts/`：启动脚本、导入脚本、构建脚本

## 启动方式

后续建议统一通过 `src/api/app.py` 启动服务，例如：

```bash
uvicorn src.api.app:app --reload
```

