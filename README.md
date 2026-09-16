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

## 目录结构

```
app.py                      FastAPI 单文件应用（路由 + 内嵌前端）
agent/                      LangGraph ReAct Agent（工具 + 中间件）
rag/                        向量库（FAISS）与检索总结
service/                    数据库、会话记忆、鉴权、登录护栏
model/                      模型工厂（DeepSeek / DashScope）
utils/                      配置、日志、路径、文件处理
config/                     4 个 yaml 配置
prompts/                    提示词文本
data/                       演示业务数据（CSV + SQLite）与知识库文件
faiss.db/                   FAISS 索引产物（不入 git，可重建）
tests/                      pytest 测试
requirements.txt            依赖清单
.env.example                环境变量模板（.env 不入 git）
```

## 后续计划（第 1 步：检索纵深）

1. 混合召回（向量 + BM25）→ 加权融合 → cross-encoder 精排；
2. 结构化切分（按条目）+ 内容级去重 + 索引重建入口 + embedding 版本检测；
3. SSE `sources` 事件：回答附可追溯的来源；
4. 检索评测集（golden set）与 recall@k / MRR 对比表，量化每次调参收益。
