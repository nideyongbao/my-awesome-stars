# GitHub Actions 运行指南

工作流支持每日定时运行和 `workflow_dispatch` 手动运行。

## 首次运行

1. 在仓库的 Actions 页面选择 `Update Awesome Stars`。
2. 点击 `Run workflow`，使用默认分支运行。
3. 等待离线测试、分类器和提交步骤全部成功。

## 运行顺序

1. 安装 `requirements.txt`。
2. 执行 `python -m unittest -v test_main.py`。
3. 执行 `python main.py`。
4. 仅在生成文件变化时提交并推送。

## 检查结果

- `README.md` 应按 Vault 6+3 主分类分组。
- `stars_cache.json` 中每项应包含 `category_id` 和 `vault_category`。
- `nideyongbao/*` 应全部位于 `Personal-Repositories`，并保留原技术主题的
  `subject_*` 字段。
- `repository_overrides` 中的仓库应标记为 `manual_override`，且不调用 LLM。
- 新仓库应具有 `clone_url`、`default_branch` 等最新元数据。
- 取消 Star 或不可见的仓库应从活动快照消失，并出现在
  `removed_stars.json`。

## 定时

工作流当前配置为每天 UTC 00:00 触发。GitHub 的定时任务可能排队，实际开始和
提交时间不保证精确等于计划时间。
