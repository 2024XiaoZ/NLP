# NLP Agent - RAG-based Intelligent Question Answering System

基于 RAG（Retrieval-Augmented Generation）的智能问答系统，支持本地知识库和实时网络搜索的混合检索策略。

## 项目结构

```
NLP_agent/
├── backend/              # 后端核心代码
│   ├── agent/           # Agent 编排模块
│   ├── rag/             # RAG 相关模块（向量存储、证据聚合）
│   ├── tools/           # 工具模块（本地RAG、网络搜索）
│   ├── core/            # 核心配置和工具
│   ├── services/        # 服务层（LLM客户端）
│   └── schemas/         # 数据模型
├── requirements.txt     # Python 依赖
├── test_e2e.py         # 端到端测试脚本
└── WORKFLOW_ANALYSIS.md # 工作流分析文档
```

## 核心功能

### ✅ 已完成功能

1. **智能路由系统**
   - 基于规则和 LLM 的意图识别
   - 支持三种路由策略：`local`、`web`、`hybrid`
   - 自动选择最合适的知识源

2. **本地 RAG 集成**
   - FAISS 向量数据库
   - Markdown 知识库自动分块和嵌入
   - 相似度搜索和检索

3. **实时网络搜索**
   - Tavily Search API 集成
   - JSON 格式数据解析和处理
   - 搜索结果标准化

4. **证据聚合与生成**
   - 多源证据整合
   - 上下文构建
   - LLM 驱动的答案生成

### 🚧 进行中功能

- **Reranking（重排序）**：BM25 + 向量相似度混合排序
- **来源可信度评估**：权威来源优先
- **时效性评分**：时间敏感查询优化

### 📋 规划中功能

- **多模态支持**：图像和音频输入处理

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `backend/.env` 文件：

```env
# LLM API 配置（必需）
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.zhizengzeng.com/v1
LLM_MODEL=gpt-4o-mini

# Tavily Search API（Web 策略需要）
TAVILY_API_KEY=your_tavily_key_here

# 可选配置
LOCAL_TOP_K=6
CACHE_TTL_SECONDS=900
```

### 3. 初始化向量库

首次运行时，系统会自动构建 FAISS 索引：

```bash
# 确保知识库文件存在
backend/storage/data/fictional_knowledge_base.md
```

### 4. 启动服务

```bash
uvicorn backend.app:app --reload
```

服务将在 `http://127.0.0.1:8000` 启动。

### 5. 运行测试

```bash
# 确保服务已启动后运行端到端测试
python test_e2e.py
```

## API 端点

### POST `/api/agent/answer`

智能问答主接口。

**请求体**:
```json
{
  "q": "What is Sereleia?"
}
```

**响应**:
```json
{
  "answer": "...",
  "sources": [...],
  "routing": {
    "policy": "local",
    "rationale": "..."
  },
  "latency_ms": {
    "retrieve": 100,
    "rerank": 0,
    "generate": 500,
    "total": 600
  },
  "confidence": 0.95
}
```

### POST `/api/router/intent`

测试意图识别模块（仅返回路由决策）。

### GET `/healthz`

健康检查端点。

## 工作流

详细的工作流分析请参考 [WORKFLOW_ANALYSIS.md](./WORKFLOW_ANALYSIS.md)。

## 技术栈

- **框架**: FastAPI
- **LLM**: LangChain + OpenAI-compatible API
- **向量数据库**: FAISS
- **网络搜索**: Tavily Search
- **数据处理**: Pydantic

## 开发

### 项目依赖

参见 [requirements.txt](./requirements.txt)

### 代码结构说明

- `backend/agent/orchestrator.py`: 主编排逻辑
- `backend/agent/router_llm.py`: LLM 驱动的路由决策
- `backend/agent/synth.py`: 答案生成模块
- `backend/tools/local_rag.py`: 本地 RAG 检索
- `backend/tools/web.py`: 网络搜索工具
- `backend/rag/aggregator.py`: 证据聚合模块

## 许可证

[根据实际情况填写]

