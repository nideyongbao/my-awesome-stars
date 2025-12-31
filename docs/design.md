# Auto-Smart-Stars 架构设计文档

## 1. 核心需求
结合 GitHub Actions 和 LLM API，实现类似 Starred 的逻辑，自动对 GitHub Star 的仓库进行智能分类。

## 2. 核心架构设计
整个系统由以下几个模块组成：

1.  **触发器 (Trigger)**
    - **方式**: 每天凌晨 (Cron Job) 或 手动触发 (Workflow Dispatch)。
    - **目的**: 确保持续更新，同时允许用户随时干预。

2.  **增量逻辑 (Delta Check)**
    - **机制**: 脚本比对本地缓存 (`stars_cache.json`) 和 GitHub 最新获取的 Star 列表。
    - **优势**: 只对 **新增** 的仓库调用 LLM 进行分类，大幅节省 Token 费用，实现几乎零成本运行。

3.  **LLM 判官 (The Classifier)**
    - **模型**: 调用 DeepSeek-V3 / GPT-4o-mini / Gemini Flash 等高性价比模型。
    - **输入**: 仓库的 `Readme` 或 `Description`。
    - **输出**: 预定义的分类标签（如 `AI-Sys-Train`, `AI-App-Agent` 等）。

4.  **渲染 (Render)**
    - **产物**: 根据分类结果生成结构化的 `README.md`。
    - **动作**: 将更新后的 `stars_cache.json` 和 `README.md` Commit 并 Push 回仓库。

## 3. 为什么这个方案“优雅”？
1.  **成本可控**: 引入缓存机制 (`stars_cache.json`)，避免了每次全量跑 LLM。假设你明天新 Star 了 3 个项目，Action 只会调用 3 次 API。
2.  **分类精准**: 相比于 GitHub 自带的 Topics (很多仓库乱填或不填)，LLM 能深度理解语义。
    - *例子*: 它能理解 "A fast implementation of FlashAttention" 属于 `AI-Sys-Perf` (系统性能优化) 而不是简单的 `AI`。
3.  **完全自动化**: 用户唯一需要做的动作就是点击 GitHub 的 **Star** 按钮。剩下的元数据抓取、分类、整理全由 Agent 自动完成。

## 4. 技术栈
- **Python**: 核心逻辑实现。
  - `PyGithub`: 操作 GitHub API。
  - `openai`: 调用兼容 OpenAI 接口的 LLM。
  - `tqdm`: 显示进度条。
- **GitHub Actions**: 自动化 CI/CD 平台。
- **GitHub Secrets**: 安全存储 Token 和 Key。
