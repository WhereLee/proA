# agent-ticket

智能工单处理 Agent。进度见 `docs/agent项目A-工单Agent-开发路线.md`。

- **阶段 0（已完成）**：地基与 Mock —— FastAPI/SSE、LLM provider 封装、结构化输出、tracer、mock 业务系统 + 故障注入。
- **阶段 1（已完成）**：只读工具闭环 —— 工具注册与参数校验、自研 agent loop、多轮会话隔离、任务级 eval。
- **阶段 2（已完成）**：写操作 + 安全 + 幂等 —— 写工具、幂等键、HITL 人工确认、权限分级、注入防护、trace 脱敏。
- **阶段 3（已完成）**：可靠性 —— 显式状态机、SQLite 持久化 checkpoint、进程重启断点续跑、失败路径重试策略。
- **阶段 4（已完成）**：评估与治理 —— LLM-judge、失败分类、CI 回归门禁、成本统计与预算熔断、并发闸门与压测。
- **阶段 5（已完成，P1 加分项）**：上下文压缩、MCP server、自研 harness vs LangGraph 对照。
- **多域扩展（已完成）**：同一套 Agent 平台落地三类域 —— `tickets`（工具型）、`data`（数据分析/Text-to-SQL）、`research`（行业调研/规划型）。

> 可选依赖（阶段 5 的 MCP 与框架对照）：`python -m pip install -r requirements-optional.txt`。未安装时相关测试自动跳过。

> 说明：路线文档目录树以 `agent-ticket/` 为项目根；本仓库直接以仓库根为项目根。

## 运行

```bash
python -m pip install -r requirements.txt

# 主应用（默认 LLM_PROVIDER=fake，可离线跑）
uvicorn app.main:app --reload --port 8000

# Mock 业务系统（另开终端；agent 工具调用需要它）
uvicorn mocks.app:app --reload --port 8100
```

接入真实模型（任一 OpenAI 兼容端点）。已在用的**小米 MiMo** 配置见 `.env`：

```bash
LLM_PROVIDER=openai \
LLM_API_KEY=sk-xxx \
LLM_BASE_URL=https://api.xiaomimimo.com/v1 \
LLM_MODEL=mimo-v2.6-flash \
LLM_FALLBACK_MODELS=mimo-v2.6-pro \
LLM_EXTRA_BODY='{"thinking":{"type":"disabled"}}' \
uvicorn app.main:app --port 8000
```

> MiMo 说明：OpenAI 兼容；鉴权支持 `Authorization: Bearer` 或 `api-key`；`thinking` 默认开启（会强制 temperature/top_p 并产生推理 token），可用 `LLM_EXTRA_BODY` 关闭以求更快更省。成本核算默认按 `mimo-v2.6-pro` 海外价（可用 `LLM_PRICE_*` 覆盖）。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| POST | `/chat`、`/chat/stream` | 已废弃：仅保留 token 流示例（带 usage）；请用 `/agent/*` |
| POST | `/agent/run`、`/agent/run/stream` | 阶段 1：agent 工具循环（含写操作时会挂起待确认） |
| POST | `/agent/confirm` | 阶段 2：对挂起的写操作做出人工确认（approve/reject） |
| POST | `/agent/cancel` | 阶段 3：取消一个 run |
| GET | `/agent/run/{run_id}`、`/agent/runs` | 阶段 3：查询持久化的 run 状态 |
| POST | `/research/submit` | 调研域：提交长任务（后台队列），返回 run_id |
| GET | `/research/{run_id}` | 调研域：查询进度/结果 |
| POST | `/research/{run_id}/cancel`、`/resume` | 调研域：取消 / 从 checkpoint 续跑 |
| POST | `/knowledge/ingest` | RAG：文档入库（分块+向量） |
| GET | `/knowledge/search` | RAG：混合检索（可选 `rerank=true`），返回带来源的片段 |
| GET | `/knowledge/stats` | RAG：文档/分块统计 |
| GET | `/audit/confirmations` | 阶段 2：人工确认审计记录 |
| GET | `/traces/{id}`、`/traces` | 回放 / 列出 trace |

```bash
# 只读：直接完成
curl -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{"message":"帮我查一下订单 O1001"}'

# 写操作：先挂起，再确认
curl -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{"message":"帮我把订单 O1001 退款","role":"agent"}'
# -> {"status":"awaiting_confirmation","run_id":"..."}
curl -X POST http://localhost:8000/agent/confirm \
  -H "Content-Type: application/json" \
  -d '{"run_id":"...","approved":true}'
```

## Agent 协议

模型每轮只输出一个 JSON：`{"action":"tool","tool":"...","args":{...}}` 或 `{"action":"final","answer":"..."}`。

工具（`app/tools/`）：

| 工具 | 类型 | 权限 | 确认 |
|---|---|---|---|
| `get_order` / `get_logistics` / `get_user` / `get_ticket` | 只读 | 任意 | 否 |
| `update_address` / `refund` | 写 | `agent` | 是 |

可靠性/安全守卫（全部在 **代码层** 强制，不依赖模型自觉）：

- 工具白名单 + pydantic **参数校验**（拒绝裸传模型参数）
- 非法 JSON 纠错、重复调用检测（`loop_detected`）、`max_steps`、工具超时
- **权限分级**：角色不符 → 拒绝执行
- **HITL 确认**：危险工具挂起 run，`/agent/confirm` 决定后才执行；确认记录进审计
- **幂等**：写调用带 `Idempotency-Key`（由 run_id+工具+参数派生），失败重试不产生重复副作用
- **注入防护**：工具返回视为不可信数据（扫描 + 包裹为 DATA）
- **脱敏**：trace 元数据/属性自动脱敏（api key/token 等）

## 可靠性与恢复（阶段 3）

- **显式状态机** `app/agent/statemachine.py`：`pending → running → {awaiting_confirmation | completed | failed | cancelled}`，非法迁移直接报错，杜绝隐式状态。
- **持久化 checkpoint** `app/persistence/`（SQLite，stdlib）：每步都写一次 RunContext（含 messages/steps/pending/trace），进程重启可从磁盘 rehydrate。
- **断点续跑**：写操作执行前先把"已批准"写入 checkpoint；若执行中途崩溃，重启后 `resume` 用**同一 `Idempotency-Key`** 重放，副作用不重复。
- **失败路径策略**：幂等/只读工具对瞬时错误（超时、5xx）自动重试（`AGENT_TOOL_ATTEMPTS` + 退避），业务错误（4xx）不重试；重试耗尽后回填给模型决策；`max_steps`/死循环 → `failed`。

> 演练：`test_resume_after_restart_executes_write_exactly_once` 模拟"重启 + 副作用已提交但响应丢失"，验证最终只执行一次。

## 成本与并发治理（阶段 4）

- **成本统计 + 预算熔断**：每次 LLM 调用累计 token/成本；单 run 超 `AGENT_MAX_COST_USD` 立即熔断（`budget_exceeded`，状态 `failed`），run 事件返回 `cost_usd`。
- **并发闸门** `app/obs/ratelimit.py`：`AGENT_MAX_CONCURRENCY` 限制同时运行的 run；`TokenBucketLimiter` + `ThrottledLLM` 模拟模型侧 RPM 限流。
- **压测**：

```bash
python -m app.bench --runs 40 --concurrency 8        # 延迟 P50/P95/P99 + 成本
python -m app.bench --runs 40 --concurrency 8 --rpm 120   # 叠加模型限流
```

## Eval 与回归门禁

```bash
python eval/generate_dataset.py                     # 生成 eval/datasets/readonly_v1.jsonl
python -m app.eval --limit 10                       # 跑评估（需 mock 服务）
python -m app.eval --json > report.json             # 完整报告
python -m app.eval --judge                          # 叠加 LLM-judge 质量评分
python -m app.eval --oracle --in-process-mock --min-pass-rate 0.9   # CI 自检 + 门禁

# v2 硬数据集：多工具链 / 故障注入 / HITL / 注入对抗 / 权限
python -m app.eval --dataset eval/datasets/hard_v2.jsonl --reset-state --oracle --in-process-mock
python scripts/run_baseline.py --dataset eval/datasets/hard_v2.jsonl --reset-state --out baseline_hard.json
```

数据集：
- `readonly_v1.jsonl`：52 条只读基础任务（工具轨迹 + 结果断言）。
- `hard_v2.jsonl`：20 条硬任务，评测框架支持 `role` / `fault`（故障注入）/ `confirm`（HITL 决策）/ `expect_state`（副作用断言）/ `expect_injection` / `forbidden_tools` / `expect_min_attempts`。
- `data_v1.jsonl`：8 条数据分析任务（Text-to-SQL，断言 SQL 结果）。
- `research_v1.jsonl`：3 条调研任务（search/fetch + 报告）。

按域跑评测：

```bash
python -m app.eval --domain data     --dataset eval/datasets/data_v1.jsonl --oracle
python -m app.eval --domain research --dataset eval/datasets/research_v1.jsonl --oracle
python scripts/run_baseline.py --domain data     --dataset eval/datasets/data_v1.jsonl --out baseline_data.json
python scripts/run_baseline.py --domain research --dataset eval/datasets/research_v1.jsonl --out baseline_research.json
```

域级步数预算：`Domain.max_steps`（调研域设为 12，因其规划步骤多）。

- 评分器：规则（工具轨迹子序列 + 结果 path 断言 + 完成性）+ **LLM-judge**（固定 prompt、温度 0、双评、保守仲裁）。
- **失败分类** `app/eval/analysis.py`：tool_missing / tool_selection_error / observation_mismatch / permission / tool_error / loop / max_steps / budget ...
- **回归门禁** `app/eval/gate.py`：`--min-pass-rate` 与 `--baseline-pass-rate + --max-regression` 未过则退出码 1。
- `--oracle` 用确定性回放 LLM 自检 harness（无 API key），其通过率**不是**模型质量指标；真实 baseline 需真实模型。
- CI：`.github/workflows/ci.yml` 跑 pytest + oracle 门禁 + 压测冒烟。

真实模型 baseline（写入 `baseline_report.json`）：

```bash
python scripts/run_baseline.py                 # 全量 52 条，规则评分
python scripts/run_baseline.py --limit 6 --judge   # 叠加 LLM-judge
```

> 已跑（`mimo-v2.6-flash`）：
> - v1 只读：**51–52 / 52（约 98–100%）**，偶发 `tool_missing` 出现在"发货了吗"这类歧义问题上。
> - v2 硬数据集：改进系统提示前 **16/20**（模型只在文字里追问确认、不触发写工具），改进后 **20/20**。
> - data 域：**8/8**；research 域核心能力：**3/3**。
> - LLM-judge 6 条双评一致。
> - 已知短板：research 域模型**不按字面给出 http 链接**（引用格式遵循弱），暂以核心能力（检索/抓取/报告）计分，引用校验列为后续工作。

## 故障注入

```bash
curl -X POST http://localhost:8100/_fault \
  -H "Content-Type: application/json" \
  -d '{"mode":"partial","target":"order","remaining":1}'
#   mode: error | timeout | partial（副作用已提交但调用方看到失败 → 触发幂等重试）
curl -X POST http://localhost:8100/_reset
```

## RAG（A 的语义记忆，完善版）

真实、可离线运行的 RAG 栈（本地模型 + 真向量数据库）：

- **嵌入**：本地 ONNX `bge-base-zh-v1.5`（768 维，`onnxruntime`）——`app/memory/onnx_models.py`。可插拔 `Embedder`，保留 `HashingEmbedder`（离线兜底）。
- **向量库**：**PostgreSQL + pgvector 0.8.2**（`agent_kb` 库，`kb_docs`/`kb_chunks` + HNSW 余弦索引）——`app/memory/pg_store.py`；亦可退回 SQLite。
- **混合检索**：BM25 + 向量，**RRF 融合**——`fusion.py`、`knowledge.py`。
- **重排**：本地 ONNX `bge-reranker-base` CrossEncoder（也可 LLM 重排 / 规则）——`rerank.py`。
- **多查询改写**：可选 `LLMQueryRewriter`；`RagPipeline` 串联「改写→混合检索→重排」——`rag.py`。
- **引用 / 过滤**：结果带 `title/source`；`scope` 按租户隔离。
- **接入**：`/agent/run` 已挂载 domain 注册表（含 `search_knowledge`）+ 情景记忆 + 上下文压缩；也可用 `/knowledge/*` 直接检索。
- **配置**（`.env`）：`EMBEDDING_PROVIDER=onnx`、`VECTOR_STORE=pgvector`、`PG_DSN`、`RERANKER=onnx` 及各模型路径。

```bash
# 检索评估：本地 ONNX 嵌入 + pgvector，对比哈希基线
python -m app.eval.retrieval --dataset eval/datasets/rag_v1.jsonl
# onnx + pgvector : R@1=82% R@3=100% MRR=0.894
# hashing+sqlite  : R@1=73% R@3=100% MRR=0.864
```

> 运行前置：本地 PostgreSQL 已建库 `agent_kb` 并 `CREATE EXTENSION vector`；模型位于 `D:\models\onnx-int8\`。

## 上下文工程 / MCP / 框架对照（阶段 5）

- **上下文压缩** `app/agent/context.py`：超预算时保留 system + 最近若干轮，较早的轮次用**抽取式摘要**（保留 order_id/status/amount 等关键字段），保证关键信息不丢；可注入 LLM 摘要器。runner 传 `context_manager=` 即启用。
- **MCP server** `app/mcp/`：把同一套工具目录暴露为 MCP server（`tools/list`、`tools/call` 适配 ToolRegistry）。
  ```bash
  python -m app.mcp      # stdio
  ```
- **自研 harness vs LangGraph**：`app/harness_langgraph/` 用 StateGraph 复刻同一 loop，跑同一批 eval 对照。
  ```bash
  python -m app.harness_langgraph --limit 20
  ```

## 测试

```bash
python -m pytest
```

覆盖：重试/降级、结构化修复、mock 规则与故障、SSE、tracer、工具校验、agent loop、会话隔离、eval、写工具幂等、HITL、权限、注入防护、脱敏、状态机、checkpoint 持久化、重启断点续跑、失败重试策略、取消、成本/预算熔断、失败分类、LLM-judge、回归门禁、压测、上下文压缩、MCP、框架对照。

## 目录

```
app/
  api/          chat、agent（run/confirm/cancel/audit）、research（队列）、traces、sse
  domains/      多域：tickets / data_analysis(+sales_data) / research(+research_corpus)
  memory/       记忆四层：bm25/chunking/embedding/fusion/rerank/rewrite/knowledge(RAG)/episodic/manager/seed
  llm/          base / provider / fake / scripted / resilient / structured / runtime
  agent/        loop（自研 harness）、context（上下文压缩）、protocol、session、state（checkpoint）、statemachine、audit
  harness_langgraph/  LangGraph 对照 harness
  mcp/          MCP server（工具暴露）
  tools/        registry、mock_client、readonly、writes
  security/     injection（注入防护）、redact（脱敏）
  persistence/  SQLite（run checkpoint + session）
  eval/         graders、judge、analysis、gate、oracle、runner、CLI
  obs/          tracer、cost、ratelimit
  bench/        并发压测
mocks/          业务 mock（含幂等）+ 故障注入
.github/        CI（pytest + eval 门禁 + 压测冒烟）
eval/           数据集生成 + datasets/
tests/
docs/           选题与开发路线
```
