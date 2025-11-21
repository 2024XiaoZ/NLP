# 工作流执行分析

## 当前工作流状态

### ✅ orchestrator.py 完成情况

`orchestrator.py` 模块已经**完整实现**，包含以下功能：

1. **主入口函数** (`answer`):
   - ✅ 路由决策（调用 router）
   - ✅ 策略执行（local/web/hybrid）
   - ✅ 证据聚合（aggregator）
   - ✅ 答案生成（synth）
   - ✅ 响应构建（FinalResponse）
   - ✅ 异常处理和回退

2. **策略执行** (`_execute_policy`):
   - ✅ `local` 策略：仅本地检索
   - ✅ `web` 策略：仅网络检索
   - ✅ `hybrid` 策略：并行执行本地和网络检索
   - ✅ 兜底策略：未知策略时回退到 hybrid

3. **辅助函数**:
   - ✅ `_run_local`: 本地 RAG 检索
   - ✅ `_run_web`: 网络搜索
   - ✅ `_fallback_response`: 错误回退响应

## 完整工作流程

### 1. 请求接收 (`app.py`)
```
POST /api/agent/answer
↓
orchestrator.answer(query)
```

### 2. 意图识别/路由 (`router.py`)
```
router.route(query)
↓
router_llm.llm_route(query)
↓
返回 RoutingDecision {policy, rationale}
```

**路由逻辑**:
- 规则匹配：本地关键词 → `local`，实时关键词 → `web`
- 混合匹配：同时命中 → 调用 LLM 判断
- LLM 判断：未命中规则或混合情况 → 调用 LLM API
- 回退策略：失败 → `hybrid`

### 3. 策略执行 (`orchestrator._execute_policy`)

#### 3.1 Local 策略
```
_run_local(query, top_k)
↓
local_rag.search_local()
↓
返回 local_hits, []
```

#### 3.2 Web 策略
```
_run_web(query, top_k)
↓
web_tool.search_web()
↓
返回 [], web_hits
```

#### 3.3 Hybrid 策略
```
并行执行:
├── _run_local(query, top_k) → local_hits
└── _run_web(query, top_k) → web_hits
↓
返回 local_hits, web_hits
```

### 4. 证据聚合 (`aggregator.aggregate_evidence`)
```
输入: local_hits, web_hits
↓
_normalize_local() → LocalEvidence[]
_normalize_web() → WebEvidence[]
_render_local_block() → local_block (文本)
_render_web_block() → web_block (文本)
↓
返回: {local_sources, web_sources, local_block, web_block}
```

### 5. 答案生成 (`synth.generate_answer`)
```
generate_answer(query, local_block, web_block)
↓
构建 Prompt (System + User)
↓
调用 LLM (ChatOpenAI.ainvoke)
↓
解析 JSON 响应 {answer, sources, confidence}
↓
返回合成结果
```

### 6. 响应构建
```
FinalResponse(
    answer=synth_result["answer"],
    sources=[...local_sources, ...web_sources],
    routing=route_decision,
    latency_ms={retrieve, rerank, generate, total},
    confidence=synth_result["confidence"]
)
```

### 7. 返回响应
```
HTTP 200 OK
{
    "answer": "...",
    "sources": [...],
    "routing": {"policy": "...", "rationale": "..."},
    "latency_ms": {...},
    "confidence": 0.95
}
```

## 工作流执行步骤总结

```
[1] 接收请求
    ↓
[2] 意图识别（router）✅
    ├─ 规则匹配 ✅
    ├─ LLM 判断 ✅
    └─ 返回路由决策 ✅
    ↓
[3] 策略执行（orchestrator）✅
    ├─ local: 本地检索 ✅
    ├─ web: 网络检索 ✅
    └─ hybrid: 并行检索 ✅
    ↓
[4] 证据聚合（aggregator）✅
    ├─ 标准化证据 ✅
    └─ 生成上下文块 ✅
    ↓
[5] 答案生成（synth）✅
    ├─ 构建 Prompt ✅
    ├─ 调用 LLM ✅
    └─ 解析响应 ✅
    ↓
[6] 构建响应 ✅
    ↓
[7] 返回结果 ✅
```

## 当前可执行到的步骤

### ✅ 已完成并可执行的步骤

1. **意图识别模块** ✅
   - 规则匹配逻辑
   - LLM 意图判断
   - 混合问题识别
   - 缓存机制

2. **路由执行模块** ✅
   - Local 策略执行
   - Web 策略执行
   - Hybrid 策略执行（并行）
   - 异常处理

3. **证据聚合模块** ✅
   - 本地证据标准化
   - 网络证据标准化
   - 上下文块生成

4. **答案生成模块** ✅
   - Prompt 构建
   - LLM 调用
   - JSON 解析
   - 置信度计算

5. **完整流程** ✅
   - 端到端执行
   - 延迟统计
   - 错误回退

### 依赖的外部服务

1. **LLM API** (必需)
   - 意图识别调用
   - 答案生成调用
   - 配置在 `.env`: `LLM_API_KEY`, `LLM_BASE_URL`

2. **Tavily API** (Web 策略需要)
   - 网络搜索功能
   - 配置在 `.env`: `TAVILY_API_KEY`
   - 缺失时会抛出异常

3. **本地向量库** (Local/Hybrid 策略需要)
   - FAISS 索引文件
   - 存储在: `backend/storage/indexes/faiss_md_index/`
   - 启动时自动构建/加载

## 测试验证

### 已测试功能

1. ✅ 意图识别测试通过
   - Local 问题识别 ✅
   - Web 问题识别 ✅
   - Hybrid 问题识别 ✅（已修复）

2. ✅ 路由策略
   - 移除了 cascade 策略 ✅
   - 混合问题正确识别 ✅

### 待测试功能

1. 端到端完整流程测试
2. 不同策略的实际执行效果
3. 错误处理和回退机制
4. 性能测试（延迟统计）

## 结论

**orchestrator.py 模块已完成**，工作流可以执行到**最终步骤**（返回完整响应）。

所有核心组件已实现：
- ✅ 路由决策
- ✅ 策略执行
- ✅ 证据聚合
- ✅ 答案生成
- ✅ 响应构建

系统已准备好进行端到端测试！

