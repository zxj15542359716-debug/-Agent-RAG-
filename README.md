# 电脑外设智能售后 Agent

基于 **LangGraph ReAct Agent + FAISS 向量检索 + FastAPI SSE 流式**的售后智能助手：
登录/注册、知识库问答（故障排除 · 保养 · 选购）、保修与使用记录查询、售后上报、使用报告生成。

> 本目录是 `E:\C++\.创新_Agent` 的**工作副本**（2026-09-16 全量备份），
> 原项目保持冻结，全部改造在本目录进行。

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
# 或双击 启动网页.bat；端口可用环境变量 APP_PORT 覆盖，便于与原目录服务同时运行
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

覆盖：登录签发 token、无凭证/伪造/过期 token 一律 401、上报身份取自 token、
登录失败锁定、注册密码长度、新旧密码哈希兼容与惰性迁移。
测试使用临时数据库（不触碰项目数据），不调用外部大模型 API。

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
app.py                      FastAPI 单文件应用（路由 + 内嵌前端 + SSE sources 事件）
agent/                      LangGraph ReAct Agent（工具 + 中间件；工具发布检索来源）
rag/                        检索链路：chunker（条目切分）/ chunk_store（chunk 账本）/
                            vector_store（FAISS + 索引版本检测）/ hybrid_retriever（召回+融合+重排）/
                            rag_service（检索 + 总结 + 溯源）
scripts/reindex.py          索引重建入口（增量 / --rebuild 全量）
eval/                       评测：golden.yaml（52 题）/ run_retrieval.py / run_answer.py / results/
service/                    数据库、会话记忆、鉴权、登录护栏
model/                      模型工厂（DeepSeek / DashScope）
utils/                      配置、日志、路径、文件处理
config/                     5 个 yaml 配置（含 retrieval.yml：召回/融合/重排旋钮）
prompts/                    提示词文本
data/                       演示业务数据（CSV + SQLite）与知识库文件
faiss.db/                   FAISS 索引与 chunk 账本产物（不入 git，reindex 可重建）
tests/                      pytest 测试（18 用例：鉴权/密码/切分/融合）
requirements.txt            依赖清单
.env.example                环境变量模板（.env 不入 git）
```

## 后续计划（第 2 步：编排架构）

按"改造路线图"推进，下一步（对齐 open_deep_research / Dify 的经验）：

1. 单 ReAct → supervisor + 并行子研究者（报告生成场景拆"故障诊断/保修政策/历史工单"并行调研）；
2. 报告结构化输出（Pydantic：故障类型/根因/步骤/条款/来源），token 超限截断重试；
3. LangGraph checkpoint 落库（替代进程内会话记忆：断点续跑 + 会话持久化）；
4. 节点级事件协议 + 运行历史表 + token 成本记账；
5. 进行中已识别的优化：RAG 双次 LLM 调用（工具内总结 + 外层生成）先用 `run_answer.py`
   做 A/B 再决定是否合并。
