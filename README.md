# 电脑外设智能售后 Agent

基于 **LangGraph ReAct Agent + FAISS 向量检索 + FastAPI SSE 流式**的售后智能助手：
登录/注册、知识库问答（故障排除 · 保养 · 选购）、保修与使用记录查询、售后上报、使用报告生成。

## 架构

```
浏览器（内嵌 HTML/CSS/JS，SSE 打字机流式）
   │  Authorization: Bearer <JWT>
   ▼
FastAPI（app.py）
   ├── /api/login     登录：校验密码 → 签发 JWT；含防爆破锁定
   ├── /api/register  注册：分配4位ID（密码≥6位）
   ├── /api/report    售后上报（身份取自 token）
   └── /api/chat      SSE 流式聊天（身份取自 token）
          │
          ▼
LangGraph ReAct Agent（agent/：6 个工具 + 3 个中间件）
   ├── rag_summarize        → rag/：FAISS 检索 + 提示词总结
   ├── fetch_external_data / query_warranty / get_user_id
   ├── get_current_month / fill_context_for_report
   └── 身份经运行时上下文 current_user 注入（取自 JWT，防越权）
          │
          ▼
SQLite（service/database_service.py：users / purchases / repairs / reports）

安全与配置：
  service/auth_service.py   JWT 签发与校验（FastAPI 依赖 get_current_user）
  service/login_guard.py    登录防爆破（滑动窗口 + 锁定）
  model/fectory.py          API Key 缺失即拒绝启动（fail-fast）
  utils/config_handler.py   .env 加载 + require_env
```

## 快速开始

```bash
# 1. 安装依赖（本机 pip 默认源不可用，需指定官方源）
.venv/Scripts/python.exe -m pip install -i https://pypi.org/simple/ -r requirements.txt

# 2. 配置环境变量：复制 .env.example 为 .env 并填好 JWT_SECRET
#    （DeepSeek / DashScope 的 Key 若已在系统环境变量中配置，可不在 .env 重复）

# 3. 启动（默认 http://127.0.0.1:8618）
.venv/Scripts/python.exe app.py
# 或双击 启动网页.bat；端口可用环境变量 APP_PORT 覆盖（默认 8618）
```

首次运行且数据库为空时，会自动从 `data/external/records.csv` 导入演示数据。
注意：导入时逐个用户做 bcrypt 哈希（约 170 个），耗时数十秒，属预期的一次性成本。

## 配置说明

| 配置项 | 位置 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | .env / 系统环境变量 | 对话模型（deepseek-v4-pro），缺失拒绝启动 |
| `DASHSCOPE_API_KEY` | .env / 系统环境变量 | 向量模型（text-embedding-v4），缺失拒绝启动 |
| `JWT_SECRET` | .env | 登录凭证签名密钥；`python -c "import secrets; print(secrets.token_urlsafe(48))"` 生成 |
| 聊天/嵌入模型名 | `config/rag.yml` | |
| 向量库参数（chunk、top-k 等） | `config/faiss.yml` | |
| 数据库路径 / 保修期 | `config/agent.yml` | |
| 提示词文件路径 | `config/prompts.yml` | |
| 检索旋钮（召回条数/融合权重/重排） | `config/retrieval.yml` | 第 1 步新增，含调参实测记录 |
| 编排旋钮（路由词表/checkpoint/报告预算/记账） | `config/orchestration.yml` | 第 2 步新增 |

## 本次改造内容（安全基线）

- **修复越权（IDOR）**：业务接口不再信任请求体 `user_id`——原实现任何人伪造该字段即可读他人数据；
  现在身份一律从 `Authorization: Bearer <JWT>` 解析，登录成功签发 2 小时有效凭证。
- **密码哈希升级**：`sha256(固定盐)` → **bcrypt**（随机盐、可调工作因子）；历史哈希登录时
  **惰性迁移**，存量账号无需重置密码、无需停机。
- **登录防爆破**：同一（用户ID, IP）5 分钟内失败满 5 次锁定 15 分钟。
- **会话隔离**：会话记忆按 `用户ID:会话ID` 存取，拿到他人会话 ID 也无法续读对话。
- **配置 fail-fast**：API Key / JWT_SECRET 缺失时启动即报错（原来静默返回 None，请求时才失败）。
- 注册密码最小长度 6 位；移除页面上的演示默认口令提示。

## 测试

```bash
.venv/Scripts/python.exe -m pytest
```

59 个用例，覆盖：登录签发 token、无凭证/伪造/过期 token 一律 401、上报身份取自 token、
登录失败锁定、注册密码长度、新旧密码哈希兼容与惰性迁移（第 0 步）；
条目切分、融合权重与来源字段（第 1 步）；会话持久化与 10 轮裁剪、意图路由、报告结构化与
截断降级、并行派发、运行历史与越权防护、token 记账（第 2 步）。
测试使用临时数据库（不触碰项目数据），**不调用外部大模型 API**——
`tests/conftest.py` 里有联网哨兵：任何漏改的替身都会立刻报错而不是打真实接口。

## 第 1 步：RAG 检索纵深（已完成）

四个子项，每个都有评测数字背书（评测集见 `eval/golden.yaml`，52 题标注到条目级）：

1. **条目级切分 + chunk 账本**（`rag/chunker.py`、`rag/chunk_store.py`）：
   按问答对/编号条目结构化切分（替代 200 字机械切分），内容级去重（去页眉噪声后判重）——
   重建实测：PDF 151 条中 150 条与 TXT 判重，6 个文件产出 781 条干净 chunk；
2. **混合检索 + 精排**（`rag/hybrid_retriever.py`，参数在 `config/retrieval.yml`）：
   向量 + BM25 两路召回各 40 条 → 0.4/0.6 加权融合 → gte-rerank-v2 精排取 5 条；
3. **引用溯源**（`rag/rag_service.py` + `agent/tools/agent_tools.py` + `app.py`）：
   回答携带结构化来源（文件/条目/标题/片段），以 SSE `sources` 事件下发，
   前端在气泡下方渲染"参考来源"折叠列表；
4. **评测体系**（`eval/`）：检索侧 recall@5 / MRR@10，答案侧 LLM-as-judge 三维打分。

### 检索质量对比（52 题实测）

| 检索模式 | recall@5 | MRR@10 | 说明 |
|---|---|---|---|
| vector_k3（改造前） | 0.962 | 0.865 | 纯向量 3 条 |
| vector_k40 | 0.981 | 0.872 | 仅扩大召回池 |
| hybrid | **1.000** | 0.892 | 向量+BM25 融合（0.4/0.6，经评测调参） |
| hybrid_rerank | **1.000** | **0.950** | 融合 + gte-rerank-v2 精排（生产默认） |

> 说明：本知识库小且干净，纯向量基线本身较高；收益主要体现在 MRR（正确的排得更靠前）
> 与边缘题。调参实测还发现：RAGFlow 的默认权重 0.7/0.3 在本语料反而降召回
> （0.904），据此调整为 0.4/0.6——记录在 `config/retrieval.yml`。

### 答案质量（12 题抽样，LLM-as-judge）

relevance 5.00 · groundedness 4.67 · completeness 5.00（满分 5；明细见 `eval/results/answer_quality.md`）

### 如何复跑评测

```bash
.venv/Scripts/python.exe -m eval.run_retrieval                 # 四种模式对比（免费，秒级~分钟级）
.venv/Scripts/python.exe -m eval.run_answer --limit 12         # 答案质量抽样（每题 2 次 LLM 调用）
.venv/Scripts/python.exe -m scripts.reindex                    # 知识库更新后增量重建索引
.venv/Scripts/python.exe -m scripts.reindex --rebuild          # 换 embedding 模型/切分策略后全量重建
```

## 目录结构

```
app.py                      FastAPI 单文件应用（路由 + 内嵌前端 + SSE 事件；执行时间线与报告卡片）
agent/                      LangGraph ReAct Agent（工具 + 中间件；工具发布检索来源）
agent/orchestration/        第2步编排层：graph（图装配）/ router（意图路由）/ researchers（三路并行）/
                            report_schema（结构化报告）/ events（节点事件）/ checkpoint（会话落库）
rag/                        检索链路：chunker（条目切分）/ chunk_store（chunk 账本）/
                            vector_store（FAISS + 索引版本检测）/ hybrid_retriever（召回+融合+重排）/
                            rag_service（检索 + 总结 + 溯源）
scripts/                    运维脚本：reindex（索引重建）/ usage_report（用量报表）/
                            prune_checkpoints（会话库维护）
eval/                       评测：golden.yaml（52 题）/ run_retrieval.py / run_answer.py / results/
service/                    数据库（含运行历史三表）、鉴权、登录护栏（会话记忆已弃用）
model/                      模型工厂（DeepSeek / DashScope，含向量化用量记账）
utils/                      配置、日志、路径、usage_ledger（token 记账）
config/                     6 个 yaml 配置（retrieval.yml 检索旋钮 / orchestration.yml 编排旋钮）
prompts/                    提示词文本（含第2步的 3 个子研究者与报告合成提示词）
data/                       演示业务数据（CSV + SQLite + checkpoints.db）与知识库文件
faiss.db/                   FAISS 索引与 chunk 账本产物（不入 git，reindex 可重建）
tests/                      pytest 测试（59 用例：鉴权/密码/切分/融合/持久化/路由/报告/记账）
requirements.txt            依赖清单
.env.example                环境变量模板（.env 不入 git）
```

## 第 2 步：编排架构（已完成）

四个子项（对标 open_deep_research 的 supervisor / RAGFlow 的 checkpoint / Dify 的运行记录）：

1. **supervisor 路由 + 复用既有 Agent**（`agent/orchestration/router.py`、`graph.py`）：
   纯规则判定"报告场景 vs 日常问答"（生成类动词 ∧ 报告类名词同时命中，词表在
   `config/orchestration.yml`）；日常问答仍走改造前的单 ReAct 快路径，**零额外 LLM 调用**，
   规则漏判由原有的 `fill_context_for_report` 工具兜底（行为与改造前一致）；
2. **并行子研究者 + 结构化报告**（`agent/orchestration/researchers.py`、`report_schema.py`）：
   报告场景用 LangGraph `Send` 扇出三路**真并行**调研（故障诊断 / 保修政策 / 历史工单），
   合成器产出 Pydantic 结构化报告并同时下发"分节纯文本 + 结构化卡片"；
   合成输入超预算时按档位截断重试（先截结论长度，再按信息密度丢路：历史→保修）；
3. **checkpoint 落库**（`agent/orchestration/checkpoint.py`）：
   LangGraph 官方 SqliteSaver → `data/database/checkpoints.db`，
   `thread_id = "用户ID:会话ID"`（与原内存记忆同一个隔离键），保留最近 10 轮
   （`trim_history` 只在用户消息边界裁剪，避免产生孤儿 ToolMessage）；
   进程重启后同一会话可续聊；原 `service/session_memory_service.py` 已弃用（保留文件 + 测试防回退）；
4. **节点事件 + 运行历史 + token 记账**（`agent/orchestration/events.py`、`utils/usage_ledger.py`）：
   节点级 SSE 事件（`node_start` / `node_end` / `report` / `run`）驱动前端"执行时间线"与报告卡片；
   新增 `runs` / `run_nodes` / `usage_events` 三张表，记录每轮对话的路由、节点耗时、token 成本
   （回调入账覆盖工具内部 LLM 调用与并行子研究者；嵌入/重排按 DashScope 返回值单独记账）；
   `GET /api/runs`、`GET /api/usage` 只返回登录用户本人的数据。

### 第 2 步实测（真实调用）

| 场景 | 结果 |
|---|---|
| 日常问答 | 单 ReAct 路径不变：一轮实测 6639 tokens / 3 次模型调用 / 35.6s，检索命中 5 条来源 |
| 报告生成 | 三路并行（15s / 28s / 41s 同时收尾）→ 合成，端到端 73s、10 条来源、结构化报告成功 |
| 重启续聊 | 换 saver 实例读同一库，历史完整（`tests/test_checkpoint_memory.py`）|
| 越权防护 | 运行历史/用量接口按登录身份过滤，用他人 run_id 查不到（`tests/test_run_history.py`）|

> 结构化输出的实测结论（已落地，记录在 `agent/orchestration/researchers.py` 说明里）：
> DeepSeek 思考模式**拒绝强制 tool_choice**（`Thinking mode does not support this tool_choice`），
> 也**不支持 json_schema 形式的 response_format**；因此改为"提示词给出 JSON 契约（由 Pydantic
> 模型自动生成）+ 容错解析 + 校验"，失败自动降级为纯文本合成（报告仍可读，卡片标注降级）。

### 运维命令

```bash
.venv/Scripts/python.exe -m scripts.usage_report --days 7         # 用量与运行报表（花了多少）
.venv/Scripts/python.exe -m scripts.prune_checkpoints --dry-run   # 会话库维护（先演练再删）
.venv/Scripts/python.exe -m agent.orchestration.graph             # 编排图自检（打印节点与边）
.venv/Scripts/python.exe -m agent.orchestration.router            # 意图路由自检（典型问法判定）
```

## 后续计划

1. **让 RAG 少一次模型调用**：当前 `rag_summarize` 工具内部会调用一次模型做总结，外层 Agent 再生成最终回答。
   计划先迁移提示词，再用 `eval/run_answer.py` 做 A/B（按 relevance / groundedness 判断质量是否下降），
   有数据支撑后再合并成一次调用；
2. **量化重排的延迟代价**：检索已切到 `hybrid_rerank`（MRR@10 0.950 vs 融合档 0.892），
   下一步用 `eval/run_retrieval.py --modes hybrid,hybrid_rerank` 测两者的端到端响应时间差，
   再决定是否长期保留重排；
3. **生产化改造**：容器化部署；把 SQLite 换成 PostgreSQL 以支持多实例；
   把向量检索独立成服务——当前 BM25 索引建在进程内存里，且数据库 / RAG / checkpointer
   都是进程内单例，多副本部署时每个实例各持一份。
