# Agent 项目备战：方向判断与工程检查清单

> 生成时间：2026-09-26
> 用途：为 AI 应用 / Agent 开发方向的招聘准备项目时，作为**方向校准 + 工程标准基线 + 落地自检表**的备用文档。
> 内容来源：基于 2026-09 阶段多家企业 JD（社招 Agent 开发 / AI 应用工程师、校招 agent 工程师、上海实验室 Agent 开发工程师等）与公开技能图谱（吴恩达 AI Engineering Skills Map 2026-08）的归纳，非泛泛而谈的路线推荐。

## 使用约定（重要）

1. **本文档的 P0/P1/P2 标注的是"缺项风险维度"，不是实施排期。**
   - P0 = 缺了就不配称"生产级 Agent 项目"
   - P1 = 有则显著拉开与候选人大盘的差距
   - P2 = 按选题需要再定，可能完全不做
   - P0 内部各项之间没有默认先后顺序：实施顺序取决于选题。例如任务型多步 Agent 的状态机 / checkpoint 属于流程骨架，需先落地；而 Eval 体系需要"有稳定可测的流程"之后才有意义。**具体排期由选题讨论后单独确定，不得把本文档当排期表使用。**
2. 本文档只记录技术判断与工程标准，不包含任何"面试话术 / 叙事包装"类内容。
3. 文末自检表在讨论与编码过程中逐项填写，"验证证据"一列必须填入可复查的客观产物（测试报告、trace 截图、CI 记录、压测数据等），不接受主观描述。

---

## 一、市场在筛什么人

当前 JD 里 Agent 方向的要求已明显收敛成四类信号：

### 1. 框架层（资格赛，区分度已很低）

- LangGraph / LangChain / AutoGen / CrewAI 至少一个用熟。
- ReAct / Plan-and-Execute / 多 Agent 协作范式讲得清楚。
- **警示**：各类培训营在批量生产"会搭 LangGraph demo"的候选人，这部分只做是够不上差异化的。

### 2. 机制层（把 Agent 当"系统"而非"模型应用"）

JD 中已出现明确表述（权限控制、人工确认、异常兜底、状态流转）：

- Tool Calling / Function Calling 的真实机制。
- MCP 协议（2026 年已成事实标准）。
- 记忆管理、上下文工程、多步规划与状态流转。
- 人工确认（HITL）节点、异常兜底、执行控制。

### 3. 工程层（当前最稀缺的分水岭）

- **评测体系（evals）**：离线 + 在线，任务级指标、回归门禁。
- **可观测性（tracing）**：逐步 trace 推理链路与工具调用，而非只有应用日志。
- **成本治理**：token 统计、预算上限、超限熔断（多 Agent 场景单次成本可放大数量级）。
- **安全防御**：prompt injection（工具返回值 / 文档中嵌入的恶意指令）、越权调用、数据隔离。
- **交付基础**：Docker / CI、API 服务化、流式接口。

### 4. 算法 / 训练层（另一条赛道，不建议主攻）

SFT / DPO / RLHF、vLLM / Ray / 推理优化、Transformer 原理与论文要求——这类岗位（如"算法工程师-AI Agent"）与后端转 AI 的工程型候选人错位。

**定位结论**：锚定 **"Agent 应用 / 研发工程师 + AI 应用后端"** 这条线，不与算法岗拼训练与推理基础设施。

### 核心判断

> **2026 年 Agent 项目的分水岭不在"会不会搭"，而在"能不能证明自己搭的东西可靠"。**

会跑通 demo 的人已经过剩；能回答下面三个问题的人很少：

1. 你怎么知道这个 Agent 变好了还是变差了？（→ Eval）
2. 任务跑到一半失败了、进程挂了，怎么恢复？（→ 状态机 / checkpoint / 幂等）
3. 一把并发打进来，队列、限流、成本、长任务占用怎么办？（→ 工程治理）

项目的所有设计都应围绕这三问展开。

---

## 二、必须避开的三个同质化陷阱

### 陷阱一：三件套形态（聊天机器人 + RAG + 几个工具）

市面上教程批量产出的形态，区分度接近零。

**要求**：选题必须是**多步任务型**——

- 有状态（中间结果需要跨步骤保存与流转）；
- 有真实"执行"（会写、会调用外部系统产生副作用，而不是只"回答"）；
- 有失败可能（存在中途出错、部分完成、需要回滚的路径）；
- 越接近真实业务流程，越能承载工程化设计。

### 陷阱二：只有 happy path

绝大多数 Agent 项目从未处理过，而这些正是 JD 里"异常兜底 / 状态管理"所指的内容，也是部署到真实服务器后**必然**遇到的：

- 工具调用失败后，Planner 如何反应（重试？换工具？上报？）；
- 模型陷入循环（反复调用同一工具 / 反复自我修正）如何检测与熔断；
- 任务执行中途进程崩溃 / 部署重启，如何断点续跑；
- 部分步骤成功、部分失败时，副作用如何补偿或回滚；
- 超长任务下上下文膨胀如何控制而不丢关键信息。

### 陷阱三：框架 API 熟练 ≠ 理解 Agent

用了 LangGraph 但说不出：

- agent loop 的本质（LLM 输出 → 解析工具调用 → 执行 → 结果回填 context → 循环直到终止条件）；
- checkpointer 的存储模型与序列化方式；
- interrupt / 人工介入是如何在图执行中实现的；
- 状态合并（reducer）为什么存在、缺失会出什么问题。

被追问即卡壳。**对策见第五节主张一。**

---

## 三、按优先级排列的工程检查清单

## P0 —— 缺项即不成立

### 3.1 Eval 体系（最想强调的一条）

当前自学 Agent 的人几乎人人都跳过的一步，也是"初级"与"能交付"的分界线。

- 任务级数据集：起始规模 50–100 条即可，但必须是**端到端任务**（输入→期望的最终状态/产物），不是单轮 QA 对。
- 评分器混合设计：
  - 规则校验（结构校验、必填字段、工具调用轨迹断言、最终状态断言）——优先，零成本且稳定；
  - LLM-as-judge（质量类主观维度，需固定评审 prompt + 温度 0 + 双评/仲裁，并**定期用人工抽检校准**）。
- 回归门禁：改动 prompt / 换模型 / 改图结构后，指标必须过阈值才可合并进主干（接入 CI）。
- 错误分析闭环：失败样例要能按原因分类（检索失败 / 工具选择错误 / 规划错误 / 输出格式 / 模型能力），否则 eval 只是数字。
- 工具选型：离线 promptfoo / Ragas / DeepEval 类；线上 Langfuse / LangSmith。

### 3.2 可观测性

- 每次 run 的**完整 trace**：每一步的输入 prompt、模型输出、工具调用与返回、token 消耗、各步延迟、总成本。
- trace 必须能与业务任务 ID 关联（否则无法定位到具体用户任务）。
- 成本治理：per-run token/成本统计 + 预算上限 + 超限熔断（不是只记录不拦截）。
- 与传统监控的分工：CPU/内存/QPS 不能反映 Agent 行为，推理链路质量是独立监控维度。

### 3.3 状态持久化与失败恢复

- checkpoint 机制（每步可恢复，含中间状态序列化）。
- 任务状态机：显式定义状态与迁移（待执行 / 执行中 / 待人工 / 失败 / 完成 / 已取消），禁止隐式状态。
- 断点续跑：进程崩溃或部署重启后能从最近 checkpoint 继续，而非从头再来。
- 工具调用幂等：写操作工具需有幂等键或"已执行"判定，避免恢复时重复产生副作用（这是恢复设计里最难也最有含金量的部分）。
- 失败路径策略：明确每一类失败的处理方式（重试上限 / 退避 / 降级 / 挂起转人工 / 失败终止），并写进文档。

### 3.4 安全

- prompt injection 防护：把工具返回值、检索文档、外部数据视为**不可信输入**；最小权限 + 危险动作强制人工确认。
- 工具白名单与参数校验（schema 级校验，不接受模型直接拼装的裸参数）。
- 危险操作 HITL：写/删/支付/对外发送类操作必须有确认环节，且确认记录需持久化可审计。
- 密钥管理：不入代码、不入日志、不入 trace（trace 里的 prompt 可能夹带敏感数据，需脱敏）。
- 会话 / 租户级数据隔离：状态存储必须按归属隔离，防越权读取他人任务状态。
- 审计：谁在何时让 Agent 调了什么工具、参数与结果，需可回溯。

### 3.5 LLM 集成基本功

- 流式输出（SSE）及中断处理（客户端断开、生成中途失败）。
- 结构化输出（JSON schema 约束 + 解析失败的修复链路）。
- 超时、重试、退避、多模型降级（主模型不可用时的备份路径）。
- 限流与并发控制（模型侧 RPM/TPM 配额与本地并发闸门的配合）。

## P1 —— 有则显著拉开差距

### 3.6 MCP

- 把自己的工具暴露为 MCP server，或作为 MCP client 接入外部工具链（2026 年事实标准，JD 出现频率高）。

### 3.7 上下文工程

- 长任务下的 history 压缩 / 摘要策略（并验证压缩后关键信息不丢）。
- memory 分层：工作记忆（当前任务状态）/ 情景记忆（跨任务经验）/ 知识存储，各自存储介质与淘汰策略。
- 上下文窗口预算管理：显式的 token 预算分配与裁剪优先级。

### 3.8 HITL 审批闭环

- 人工介入的完整链路：触发条件 → 挂起 → 通知 → 人工决策 → 恢复到原状态继续执行。

### 3.9 Agent 压测（方法论与普通接口压测不同）

- 真正要测的是：并发 run 下的队列调度与公平性、模型侧限流触发点、长任务对资源的占用时间、成本失控风险、单任务延迟分布（P50/P95/P99）。
- 关键指标需同时覆盖吞吐与成本两维（Agent 场景吞吐上不去往往是模型限流而非算力）。

## P2 —— 按选题决定，可能不做

### 3.10 多 Agent 编排

**不是必备项。** 单 Agent + 工具 loop 能讲透的场景，硬上 supervisor/worker 是负分——"这里为什么要拆成两个 Agent"答不上来会直接暴露项目是搭出来的而非想出来的。仅在任务存在明确的**权限边界隔离**或**上下文规模隔离**需求时才拆分。

### 3.11 微调 / 私有化部署

除非目标公司 JD 普遍要求，否则投入产出比低，且与"Agent 工程能力"的证明目标不在同一维度。

---

## 四、执行策略（由上文推导的两条主张）

### 主张一：框架照用，但必须加一个"自研最小 harness"

用主流框架（LangGraph 等）做项目主体以满足市场对框架经验的要求，**另起独立模块**，不依赖任何框架，自行实现：

- 最小 agent loop（LLM 调用 → 工具解析 → 执行 → 结果回填 → 终止判定）；
- 工具调用解析与校验；
- checkpoint 存取与中断恢复；
- 失败重试 / 熔断。

然后用**同一批 eval 任务**在两套实现上跑对比。收益：

1. 框架原理能讲出实现依据，而不是背 API；
2. "自行实现过 Agent Loop / Function Calling"本身是多家 JD 的原文加分项；
3. 成本极低（数百行量级），却是性价比最高的"理解证明"。

### 主张二：差异化不在 Python，在"Agent × 后端工程"的交叉带

纯提示词调优、纯 RAG 参数调优的竞争者是培训班量产的候选人；而**会做权限模型、会设计状态机、懂异步队列与幂等、能把 Agent 嵌进现有业务系统**的复合能力，市场供给很少。

因此项目中"Agent 治理层"应当做**重**而不是做**轻**：

- 任务调度与队列；
- 状态机与恢复；
- 租户 / 会话隔离；
- 审计与操作回溯；
- 熔断、限流、成本管控；
- 人工审批流。

这一层既是生产必需，也是工程背景者学得最快、讲得最真实的部分。

---

## 五、自检表（讨论与编码过程中逐项填写）

填写规则：

- **机制说明**：这项技术到底在解决什么问题（自己组织语言，不接受复制本文档）。
- **本项目落地方案**：具体到模块 / 表 / 接口 / 流程的设计。
- **验证证据**：客观产物（测试报告、CI 记录、trace 截图、压测数据、故障演练记录），禁止填"已实现"。
- **掌握程度**：能讲清原理与取舍 / 能实现但细节靠查 / 仅知道概念——用于暴露薄弱项。

> 自检表填写时间：2026-09-29。证据均为可复查产物（测试名 / 命令 / 输出文件）。实现代码位于仓库根 `app/`、`mocks/`、`eval/`、`scripts/`、`.github/`。

### P0

| 项 | 机制说明 | 本项目落地方案 | 验证证据 | 掌握程度 |
|---|---|---|---|---|
| Eval 数据集与评分器 | 用固定任务集把"改好改坏"变成可测量 | v1 `eval/datasets/readonly_v1.jsonl`（52 条只读）；v2 `eval/datasets/hard_v2.jsonl`（20 条：多工具链/故障注入/HITL/注入对抗/权限）；`app/eval/graders.py` 规则评分（工具轨迹子序列 + 结果 path 断言 + 完成性 + 状态/注入/权限/重试断言）；`app/eval/judge.py` LLM-judge（固定 prompt、温度 0、双评、保守仲裁） | `tests/test_eval.py`、`tests/test_eval_extras.py`、`tests/test_eval_hard.py`；oracle 自检 v1 52/52、v2 20/20；真实模型 `mimo-v2.6-flash`：v1 51–52/52、v2 改进提示后 20/20；`baseline_judge.json` 含双评 verdicts | 能讲清原理与取舍 |
| 评测驱动的一次真实改进 | 用评测定位问题并验证改进 | v2 首跑 16/20，失败分类揭示"模型只在文字里追问确认、不调用写工具"→ 修改系统提示（直接调用工具、权限/确认交系统处理）→ 复跑 20/20 | `baseline_hard.json`（改进后）；`app/agent/loop.py` SYSTEM_PROMPT_TEMPLATE | 能讲清原理与取舍 |
| 回归门禁（CI） | 指标不过阈值不允许合并 | `app/eval/gate.py`（min-pass-rate + baseline + max-regression）；CLI `--min-pass-rate/--baseline-pass-rate/--max-regression` 未过退出码 1；`.github/workflows/ci.yml` 接入 | `tests/test_eval_extras.py::test_gate_fails_below_threshold`、`::test_gate_detects_regression`；CI 步骤 "Eval harness self-check + regression gate" | 能讲清原理与取舍 |
| 错误分析与失败分类 | 失败必须能按原因归因，否则 eval 只是数字 | `app/eval/analysis.py`：budget/loop/max_steps/permission/user_rejected/tool_missing/tool_selection_error/tool_error/observation_mismatch/no_answer；报告输出 `failure_summary` | `tests/test_eval_extras.py::test_classify_tool_missing`、`::test_classify_observation_mismatch`；`scripts/run_baseline.py` 输出 failure_summary | 能讲清原理与取舍 |
| 全链路 trace | 一次 run 失败要能定位到具体步骤 | `app/obs/tracer.py`（自建 Trace/Span，可换 Langfuse）；每次 run 记 metadata + 每步 span，`GET /traces/{id}` 回放 | `tests/test_api.py::test_chat_records_trace`；`tests/test_agent_api.py::test_agent_run_endpoint`（断言 llm.step span）；`GET /traces/{trace_id}` | 能讲清原理与取舍 |
| 成本统计与超限熔断 | 记录 token/成本并在超预算时中断 | `app/obs/cost.py`；runner 每次 LLM 调用累计 token/cost，超 `AGENT_MAX_COST_USD` → `budget_exceeded` + 状态 failed；run 事件返回 cost_usd | `tests/test_cost.py::test_run_cost_is_accumulated`、`::test_budget_breaker_trips`；run 事件字段 `cost_usd` | 能讲清原理与取舍 |
| checkpoint 与断点续跑 | 进程崩溃/重启后从最近状态继续 | `app/persistence/db.py`（SQLite）；`app/agent/state.py` serialize/deserialize + RunStore；每步 put 落库 | `tests/test_checkpoint.py::test_checkpoint_persists_state`、`::test_resume_after_restart_executes_write_exactly_once` | 能讲清原理与取舍 |
| 任务状态机 | 显式状态与迁移，禁止隐式状态 | `app/agent/statemachine.py`：pending/running/awaiting_confirmation/completed/failed/cancelled + 迁移表，非法迁移抛错 | `tests/test_statemachine.py` | 能讲清原理与取舍 |
| 写操作幂等 | 重试/恢复不产生重复副作用 | 写调用带 `Idempotency-Key`（run_id+tool+args 派生）；mock 维护幂等表，重复 key 返回 `idempotent_replay` | `tests/test_writes.py::test_refund_is_idempotent`；`tests/test_checkpoint.py` 重启演练断言 `idempotent_replay=True` 且只退一次 | 能讲清原理与取舍 |
| 失败路径策略（重试/降级/转人工） | 每类失败有明确处理方式 | 幂等/只读工具对超时与 5xx 重试（`AGENT_TOOL_ATTEMPTS`+退避），4xx 不重试；耗尽回填模型决策；max_steps/死循环→failed；危险写挂起待人工 | `tests/test_checkpoint.py::test_read_timeout_is_retried_then_reported`；`tests/test_agent_loop.py::test_loop_detected`/`::test_max_steps` | 能讲清原理与取舍 |
| prompt injection 防护 | 工具返回/外部数据视为不可信 | `app/security/injection.py`：注入特征扫描 + 包裹为 DATA；**授权/确认在代码层强制**，注入无法越权 | `tests/test_injection.py::test_scan_untrusted_detects_patterns`、`::test_injection_flagged_and_cannot_escalate` | 能讲清原理与取舍 |
| 工具白名单与参数校验 | 不接受模型裸拼参数 | `app/tools/registry.py`：白名单 + pydantic `model_validate`（schema 级） | `tests/test_tools.py::test_validate_args_rejects_bad_params`；`tests/test_agent_loop.py::test_invalid_args_recovers` | 能讲清原理与取舍 |
| 危险操作人工确认 + 审计 | 写/删类操作需人工确认且可审计 | 危险工具挂起 run（awaiting_confirmation）→ `/agent/confirm` approve/reject → 恢复；`app/agent/audit.py` + `/audit/confirmations` | `tests/test_hitl.py`；`tests/test_agent_api.py::test_hitl_confirm_endpoint` | 能讲清原理与取舍 |
| 密钥管理与 trace 脱敏 | 密钥不入代码/日志/trace | `.env`（gitignore）+ 环境变量；`app/security/redact.py` 集成进 tracer（metadata/attributes 自动脱敏） | `tests/test_redact.py` | 能讲清原理与取舍 |
| 租户 / 会话数据隔离 | 会话状态按归属隔离防越权 | `app/agent/session.py` SessionStore 按 session_id 隔离并持久化 | `tests/test_session.py::test_session_isolation` | 能讲清原理与取舍 |
| SSE 流式与中断处理 | 流式输出与客户端中断 | `app/api/sse.py`；`/chat/stream`、`/agent/run/stream`（step/final/done），generator finally 记录 trace | `tests/test_api.py::test_chat_stream_emits_sse_events`；`tests/test_agent_api.py::test_agent_stream_endpoint` | 能讲清原理与取舍 |
| 结构化输出与解析修复 | 解析失败可修复 | `app/llm/structured.py`：去代码块/抽平衡括号/去尾逗号 + pydantic 校验 | `tests/test_structured.py` | 能讲清原理与取舍 |
| 超时/重试/模型降级 | 主模型不可用走备份 | `app/llm/resilient.py`：per-attempt 超时 + 指数退避重试 + 多模型 fallback | `tests/test_llm_retry.py` | 能讲清原理与取舍 |
| 限流与并发闸门 | 控制并发与模型侧限流 | `app/obs/ratelimit.py`：ConcurrencyGate 接入 `/agent/run`；TokenBucketLimiter + ThrottledLLM | `tests/test_loadtest.py::test_loadtest_bounds_concurrency`；压测输出 `max_concurrent` | 能讲清原理与取舍 |
| 容器化部署与文档 | 可交付 | `Dockerfile`、`docker-compose.yml`（app+mocks）、`README.md` | 文件存在；**本机未安装 docker，未实际构建**（待在有 docker 环境验证） | 能实现但细节靠查 |

### P1

| 项 | 机制说明 | 本项目落地方案 | 验证证据 | 掌握程度 |
|---|---|---|---|---|
| MCP server / client | 工具以 MCP 协议暴露/接入 | `app/mcp/`：同一套 ToolRegistry 适配 MCP server（`tools/list`、`tools/call`），`python -m app.mcp` stdio | `tests/test_mcp.py`（list/call/error/build_server） | 能讲清原理与取舍 |
| history 压缩与效果验证 | 长任务上下文不爆但不丢关键信息 | `app/agent/context.py`：超预算保留 system+最近轮，较早轮抽取式摘要（保留 order_id/status/amount 等）；runner 可注入 | `tests/test_context.py::test_compression_preserves_key_facts`、`::test_runner_completes_with_tiny_budget` | 能讲清原理与取舍 |
| memory 分层设计 | 工作/情景/知识记忆分层与淘汰 | 四层：① 工作记忆=RunContext（含上下文压缩）；② 会话记忆=SessionStore（SQLite）；③ 情景记忆=EpisodicMemory（SQLite，按 domain+role 隔离、BM25 召回、上限淘汰，run 后自动记入、run 开始注入"历史经验"块）；④ 语义记忆=KnowledgeBase（RAG：入库/分块/BM25 检索，暴露为 `search_knowledge` 工具）。检索为自研 BM25（中英分词），留稠密向量/pgvector 替换接口 | `tests/test_memory.py`；`app/memory/*` | 能讲清原理与取舍 |
| RAG 检索质量（完善版） | 分块 / 混合检索 / 重排 / 引用 / 评估 | `app/memory/`：句级分块；**本地 ONNX bge-base-zh 嵌入（768维）** + **PostgreSQL/pgvector 0.8.2 向量库（HNSW）** + **ONNX bge-reranker-base 精排**；混合检索 BM25+向量 RRF 融合；可选 LLM 重排/多查询改写；scope 过滤；结果带 title/source 供引用；`search_knowledge` + `/knowledge/*`。可插拔：`hashing/sqlite` 兜底 | `tests/test_rag.py`、`tests/test_memory.py`；`python -m app.eval.retrieval`：**onnx+pgvector R@1=82% MRR=0.894** vs hashing R@1=73% MRR=0.864；`scripts/ask.py` 真实模型检索+引用；`agent_kb` 库 `kb_chunks`(768维) | 能讲清原理与取舍 |
| 上下文窗口预算分配 | 显式 token 预算与裁剪优先级 | `app/agent/context.py`：`max_tokens` + `keep_recent` + 保留 system 优先 | `tests/test_context.py` | 能讲清原理与取舍 |
| HITL 完整审批闭环 | 触发→挂起→通知→决策→恢复 | 危险工具挂起→`/agent/confirm`→恢复执行/驳回；确认记录落审计并持久化 | `tests/test_hitl.py`；`tests/test_agent_api.py::test_hitl_confirm_endpoint` | 能讲清原理与取舍 |
| Agent 压测（吞吐/延迟/成本） | 并发下吞吐/延迟分位/成本 | `app/bench/loadtest.py`：ConcurrencyGate + 延迟 P50/P95/P99 + 吞吐 + 成本；可选 `--rpm` 叠加模型限流 | `tests/test_loadtest.py`；`python -m app.bench --runs 40 --concurrency 8` | 能讲清原理与取舍 |

### P2（按选题定，未做则在"本项目落地方案"列写"不做 + 技术原因"）

| 项 | 机制说明 | 本项目落地方案 | 验证证据 | 掌握程度 |
|---|---|---|---|---|
| 多 Agent 编排（拆分的真实理由） | 仅在权限/上下文隔离有真实需求时拆分 | **不做 + 技术原因**：单 Agent + 工具 loop 已覆盖工单场景，无明确权限或上下文规模隔离需求，硬拆 supervisor/worker 是负分 | — | 能讲清原理与取舍 |
| 微调 / 私有化部署 | 训练/部署另一条赛道 | **不做 + 技术原因**：与"Agent 工程能力"证明不在同一维度，ROI 低 | — | 仅知道概念 |

### 自研 harness 对照

| 项 | 机制说明 | 本项目落地方案 | 验证证据 | 掌握程度 |
|---|---|---|---|---|
| 最小 agent loop 实现 | 不依赖框架的 LLM→解析→执行→回填→终止 | `app/agent/loop.py` AgentRunner（含确认挂起/恢复、成本熔断、重试） | `tests/test_agent_loop.py` | 能讲清原理与取舍 |
| 工具解析与校验 | 解析动作 JSON 并做 schema 校验 | `app/agent/protocol.py` + `app/tools/registry.py` | `tests/test_tools.py`、`tests/test_agent_loop.py` | 能讲清原理与取舍 |
| checkpoint / 恢复实现 | 每步可序列化并可恢复 | `app/agent/state.py` + `app/persistence/db.py` | `tests/test_checkpoint.py` | 能讲清原理与取舍 |
| 与框架实现的 eval 对比结论 | 同一批 eval 对比自研与框架 | `app/harness_langgraph/`（StateGraph 复刻同一 loop，同一工具/协议/数据集） | `python -m app.harness_langgraph --limit 20` → self 20/20 (24ms) vs langgraph 20/20 (59ms)；`tests/test_harness_compare.py` | 能讲清原理与取舍 |
