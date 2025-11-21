# NLP Agent Backend

基于 FastAPI + LangChain 的多源 RAG Agent，复刻 `NLP_ISE` demo
里的 ChatOpenAI + FAISS + Tavily 方案，适配 Python 3.11+。

## 功能特性

- FastAPI + uvicorn 暴露 `/api/agent/answer`；
- LangChain `ChatOpenAI` 负责生成，Prompt 约束输出 JSON；
- 本地 RAG：RecursiveCharacterTextSplitter 切块、OpenAI Embeddings、
  LangChain FAISS 向量库（启动时自动构建/加载）；
- Web RAG：Tavily Search + TTL 内存缓存，覆盖时效问题；
- 路由策略沿用 `local/web/hybrid` 启发式；
- 自带延迟分解、结构化日志、引用 chunk_id / URL。

## 环境准备

```bash
cd backend
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell
pip install -r requirements.txt
cp dot_env_example .env   # 填写 LLM / Tavily 密钥
```

`.env` 需要至少包含：

```
LLM_BASE_URL=https://api.zhizengzeng.com/v1
LLM_API_KEY=sk-xxx
LLM_MODEL=gpt-4o-mini
TAVILY_API_KEY=tvly-xxx
```

## 知识库

- Markdown 放在 `storage/data/fictional_knowledge_base.md`；
- FastAPI 启动时会用 demo 同款 splitter + OpenAI Embeddings 切块，
  并将 FAISS 存在 `storage/indexes/faiss_md_index/`；
- 更新 Markdown 后删除该目录即可重新构建。

## 启动服务

```bash
uvicorn backend.app:app --reload
```

接口：

- `GET /healthz`
- `POST /api/agent/answer`，形如 `{"q": "Who is Dr. Elara Vance?"}`

示例：

```bash
curl -X POST http://127.0.0.1:8000/api/agent/answer \
     -H "Content-Type: application/json" \
     -d '{"q":"Who is Dr. Elara Vance?"}'
```

## 缓存与降级

- Tavily 搜索结果写入 `backend.utils.cache.TTLMemoryCache`（默认 900s）；
- FAISS 载入失败时会抛错提示补齐依赖；
- 若 `TAVILY_API_KEY` 缺失，则网页检索阶段抛出明确异常；
- 本地召回数量可通过 `LOCAL_TOP_K` 调整。

## 路由策略

- 命中 `Sereleia / Xylos / Elara Vance / Vance Protocol` → `local`
- 包含 `today/latest/price/weather/traffic/2025/now` → `web`
- 同时包含本地和实时关键词 → `hybrid`（由 LLM 判断）
- 其他 → 由 LLM 判断使用 `local/web/hybrid`

## 目录约定

- `.env`：密钥与模型配置；
- `storage/data/`：原始 Markdown；
- `storage/indexes/faiss_md_index/`：LangChain 向量库快照；
- `web/`：前端占位。

欢迎基于此再扩展工具、Agent prompt 或持久化策略。
