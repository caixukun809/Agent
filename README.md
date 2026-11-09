🤖 Agent 探索与实践仓库
本项目是一个围绕大语言模型（LLM）应用的综合性工程实践仓库，涵盖了从单 Agent 链路搭建、RAG（检索增强生成）系统到多 Agent 协作的完整技术栈。项目注重工程化落地，内置了工具调用、上下文与记忆生命周期管理以及多种主流框架的集成方案。

📂 项目结构总览
text
Agent/
├── Langchain/          # Agent 链路（单 Agent 核心逻辑与交互）
├── RAG/                # RAG 检索增强生成（Pipeline、服务与 UI）
├── Multi-Agent/        # 多 Agent 协作（AutoGen / CrewAI / LangGraph）
├── tools/              # 统一的工具注册与运行时（Registry / Runtime）
├── langgraph/          # LangGraph 专项实现（单 Agent / 基础链路）
├── skill/              # 技能模块（可复用的 Agent 能力）
├── workspace/          # 工作空间（记忆存储、日志等）
├── MEMORY_DESIGN.md    # 上下文与记忆生命周期的详细设计文档
├── llm.py              # 大模型调用统一入口
└── .idea/              # 工程配置
🚀 核心模块
1. Agent 链路 (Langchain/)
本模块基于 LangChain 生态，实现了从基础对话到复杂任务处理的完整 Agent 链路。

多种交互形态：提供 cli.py（命令行交互）、app.py（Web 应用）以及 sdk_agent.py（SDK 调用）等多端入口。

任务型 Agent：内置 research_agent.py，展示了如何让 Agent 使用搜索、计算等工具进行深度研究并返回结构化结果。

工具与记忆集成：Agent 默认接入统一的 tools 工具库以及上下文记忆机制，确保多轮对话的连贯性与工具调用的准确性。

2. RAG 检索增强生成 (RAG/)
提供了一套轻量且可独立部署的 RAG 解决方案，旨在解决大模型幻觉与私域知识问答问题。

核心管线：rag_pipeline.py 封装了文档加载、文本分割、向量化、存储与检索的完整流程。

服务化与可视化：rag_server.py 将 RAG 能力暴露为 HTTP 接口，rag_ui.html 提供了一个轻量级的可视化界面，方便快速测试检索与生成效果。

向量检索：结合 MEMORY_DESIGN.md 中的设计，底层可支持 FAISS 等本地向量索引，实现高效的语义召回。

3. 多 Agent 协作 (Multi-Agent/)
本模块对比并实现了当前主流的三种多 Agent 协作框架，每种框架下均包含独立的场景示例：

AutoGen (AutoGen/multi_agent.py)：基于 Microsoft AutoGen，展示了多角色（如 Coder、Reviewer）对话与协作完成任务的能力。

CrewAI (CrewAI/)：基于 CrewAI 框架，实现了基于 Role-Goal-Task 结构的流水线式协作。

LangGraph (Langgraph/)：利用图（Graph）结构编排多 Agent 的状态流转与条件分支，适合构建复杂、可循环的协作拓扑。

🛠️ 公共基础设施
工具系统 (tools/)
实现了标准化的工具注册与调度中心：

registry.py：全局工具注册表，统一管理各类工具的元数据与调用入口。

runtime.py：工具运行时环境，负责处理工具调用的权限、沙箱与生命周期。

catalog.json：工具目录的静态描述，便于 Agent 动态发现与加载工具。

上下文与记忆机制 (MEMORY_DESIGN.md)
项目在 MEMORY_DESIGN.md 中详细设计并实现了 Agent 的记忆生命周期，核心包括：

分层存储：使用 SQLite 管理 sessions（会话）、messages（原始消息）、memory_candidates（记忆候选）与 memories（长期记忆）；使用 FAISS 作为本地向量索引，实现 memory_id -> vector 的语义检索。

短期与长期记忆：短期上下文默认只读取最近 8 轮对话；长期记忆在每轮结束时自动抽取，并经过事实级去重与冲突处理后才写入。

Token 预算控制：内置 tiktoken 对上下文、工具输出和摘要进行 Token 预算管理，防止超长溢出。

⚙️ 快速开始
环境准备：建议使用 Python 3.10+，并创建虚拟环境。

安装依赖：进入对应的子模块（如 Langchain、RAG 或 Multi-Agent/AutoGen）执行 pip install -r requirements.txt。

配置模型：在 llm.py 或环境变量中配置你的大模型 API Key（如 OpenAI、DeepSeek 等）。

运行示例：

Agent 链路：python Langchain/cli.py 或 python Langchain/app.py

RAG：python RAG/rag_server.py 后访问 UI 页面

多 Agent：python Multi-Agent/AutoGen/multi_agent.py

📝 设计文档
MEMORY_DESIGN.md：上下文与记忆生命周期详细设计（含参数说明与生命周期流程）。
