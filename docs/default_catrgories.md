# 默认分类说明

> 文件名保留历史拼写 `default_catrgories.md`，避免旧链接失效。

分类定义不再复制到文档或 Python 常量中。唯一权威来源是仓库根目录的
`categories.json`。

当前设计：

- 10 个 Vault 主分类：6 Domains、3 Pillars、`其他`。
- 21 个主题子分类（含 `Other` 和 `Personal-Repositories`）：只保留当前主要研究方向。
- 旧 39 类只作为 `legacy_names` 迁移别名。
- LLM 只能返回已有研究主题的 `category_id`，不能选择个人仓库标记或自动追加分类。
- `nideyongbao/*` 由 owner 规则稳定归入 `Personal-Repositories`，不调用 LLM。
- 已人工确认的单仓库修订写入 `repository_overrides`，按 GitHub ID 覆盖自动分类。

设计依据和旧分类逐项结论见 [分类体系审计](category_audit.md)。
