# My Awesome Stars 架构设计

## 目标

本仓库维护 GitHub Stars 的可审计快照，并使用与本地 Obsidian Vault 一致的
6+3 主分类。GitHub 侧保留更细的主题分类，但不允许 LLM 动态创建目录。

## 分类模型

分类分为两层：

1. `vault_category`：固定为 Vault 的 6 Domains、3 Pillars 和 `其他`。
2. `category_id`：`categories.json` 中定义的细粒度主题 ID。

`categories.json` 是唯一 taxonomy。每个主题包含：

```json
{
  "id": "inference-scheduling-kvcache",
  "name": "Inference-Scheduling-KVCache",
  "description": "请求调度、Continuous Batching 与 KV Cache 管理",
  "vault_category": "Domain-推理框架",
  "legacy_names": []
}
```

LLM 只从既有研究主题返回 `category_id`，`Personal-Repositories` 不进入 Prompt。
程序根据 taxonomy 补出 `vault_category`，因此 Vault 消费快照时不需要维护第二份
映射。

当前只保留 21 个主题子类（含 `Other` 和个人仓库标记）。旧 39 类的逐项处理见
[分类体系审计](category_audit.md)。

旧缓存以 `classification_source: legacy_migration` 标记，原始分类理由保存在
`legacy_reasoning`；新仓库使用 `classification_source: llm`。这样旧分类名不会
被误解为当前 taxonomy。

`nideyongbao/*` 是个人仓库，使用确定性的 owner 规则归入
`Personal-Repositories`，并标记 `classification_source: owner_rule`。规则覆盖前
的技术主题保存在 `subject_category_id`、`subject_category` 和
`subject_vault_category`，所以专属入口不会抹掉原主题信息。以后新增的个人仓库
也会直接进入该类，不消耗 LLM 调用。

人工审查确认的单仓库修订记录在 `categories.json.repository_overrides`，以稳定的
GitHub 仓库 ID 为键。覆盖规则优先于历史分类和 LLM，并在快照中标记
`classification_source: manual_override`；因此缓存重建、仓库改名或取消后重新
Star 都不会丢失人工结论。

## 执行链路

1. 读取并校验 taxonomy。
2. 读取旧快照，将旧分类字符串迁移为稳定 `category_id`。
3. 从 GitHub 获取当前可见的全部 Star。
4. 对已有仓库刷新元数据并复用分类；个人仓库应用 owner 规则，其余新增仓库调用 LLM。
5. 验证实际遍历数量等于 GitHub `totalCount`。
6. 比较新旧 ID 集合，识别从当前可见 Star 集合消失的仓库。
7. 通过安全阈值后，原子写入活动快照、移除记录和 README。

## 元数据刷新

已有仓库每天刷新以下字段，而不是只更新 Star 数：

- `name`、`url`、`clone_url`
- `description`、`language`、`topics`
- `default_branch`
- `archived`、`fork`、`size_kb`
- `pushed_at`、`updated_at`

仓库改名、转移、默认分支变化或归档后，下一次同步会反映到快照。

## 取消 Star 与安全边界

活动快照始终只包含当前 API 可见的 Star。消失的仓库会：

- 从 `stars_cache.json` 和 README 移除；
- 写入 `removed_stars.json`，保留最后状态与时间；
- 重新 Star 后从移除记录中清除。

因为仓库删除、转私有或 Token 权限变化也会造成“不可见”，单次移除超过旧快照
20% 时默认停止覆盖。确认外部状态正常后才能使用 `--allow-large-removal`。

## 幂等与失败处理

- `--render-only` 只迁移本地缓存和重建 README，不访问网络。
- GitHub 实际数量与 `totalCount` 不一致时不覆盖旧快照。
- taxonomy 无效、Token 缺失或新仓库无法分类时不提交半成品。
- JSON 和 README 使用同目录临时文件加 `os.replace` 原子替换。
- 同一仓库使用 GitHub repo ID 作为稳定主键。
