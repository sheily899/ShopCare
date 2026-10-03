# ShopCare 项目开发规则

## 核心执行铁律

1. 写任何代码前，必须完整阅读 `memory-bank/` 下的架构与设计文档。
2. 严格遵循 `memory-bank/tech-stack.md` 中的锁定依赖，严禁私自引入未声明的第三方库。
3. 保持内部数据流为 dataclass，仅在 API 边界（`ChatRequest` / `ChatResponse`）使用 Pydantic。
4. 每完成一个实施计划步骤，必须先通过本地运行验证，再更新 `memory-bank/progress.md` 与 `memory-bank/architecture.md`。
5. 永远不要伪造未实现的外部业务系统：无订单写操作、无真实扣款，转人工仅设建议标志。
6. 每完成一个独立功能或 implementation-plan task，必须在本地验证通过后检查 `git status`、`git diff` 与影响范围，并创建一次清晰的 Git commit；不得把未验证的半成品标记为完成。

## 开始工作前

- 使用 Codex 时，先阅读 `memory-bank/system-spec.md`、`memory-bank/PRD.md`、`memory-bank/design-document.md` 和 `memory-bank/tech-stack.md`。
- `memory-bank/progress.md` 记录施工进度；`memory-bank/architecture.md` 记录新建代码文件的职责。
- 先确认当前任务属于已批准的 P0/P1/P2 范围；未确认的扩展想法记录为待决策项，不直接实现。

## 事实与边界

- 以 `memory-bank/system-spec.md` 为代码事实基线，以 `memory-bank/PRD.md` 为产品范围基线。
- 不得把 Mock、fallback、建议、交接摘要写成真实订单查询、退款、支付、改地址、工单创建或人工已接手。
- 不得擅自引入 Java 后端、LangGraph、交叉编码器 Rerank、分布式 APM 或未批准的外部业务写接口。
- 修改稳定 API、数据结构、阈值、collection、Redis key 或依赖版本前，先更新相关规格并记录理由。

## 编码与验证

- 保持原生 Python asyncio、FastAPI、Pydantic API 模型和内部 dataclass 的架构边界。
- 小步修改，每次只处理一个明确关注点；优先补测试，再实现逻辑，再运行回归验证。
- 任何 Agent、工具、RAG、记忆或路由异常都必须保留既定的局部降级边界。
- 不提交密钥、`.env`、运行期数据、缓存或生成结果。
- 完成代码变更后同步更新 `memory-bank/progress.md` 与 `memory-bank/architecture.md`，并检查规格文档是否过期。
- 提交前检查 `git status`、`git diff`、测试结果和实际影响范围。
- GitHub 目标仓库为 `https://github.com/sheily899/ShopCare.git`。当前目录尚未初始化 Git 仓库；首次施工前先确认本地仓库与该 remote 的连接方式。默认先完成本地 commit，只有在明确具备推送授权且需要远端同步时才执行 push。

## 代码注释规范（铁律）

1. 每个新建的 Python 文件顶部，必须包含 Module Docstring，说明该文件的架构职责与边界。
2. 所有公共类与异步函数，必须使用标准 Google Style Docstring，详细注明 Args、Returns 与业务逻辑概述。
3. 凡涉及业务阈值（如 0.5、0.45、0.8、15、86400）、状态机跳转分支、异常吞噬（try...except pass）的代码行，必须在上方添加 Inline 注释，解释“为什么这么定（Why）”。
4. 严禁出现无意义的废话注释（如 `# 定义变量`）。

## 文档与交流

- 默认使用中文；技术名词保留英文原文。
- 解释时区分已验证事实、设计决策、推断和未实现能力。
- 重要阶段完成后同步更新已授权的飞书项目日志，并在写入后重新读取验证。
