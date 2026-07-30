# 手动部署指南

## 文件结构

```text
/
├── .github/workflows/update.yml
├── categories.json
├── main.py
├── removed_stars.json
├── requirements.txt
├── stars_cache.json
├── test_main.py
└── README.md
```

不要在文档中复制 `main.py` 或 taxonomy；实际文件是唯一维护源。

## 安装

```powershell
python -m pip install -r requirements.txt
```

## 离线审查

离线测试：

```powershell
python -m unittest -v test_main.py
```

只迁移现有缓存并重建 README：

```powershell
python main.py --render-only
```

该命令不读取 `GH_TOKEN`，也不会访问 GitHub 或 LLM。

## 完整同步

配置 `GH_TOKEN` 和 `LLM_API_KEY` 后：

```powershell
python main.py
```

已有仓库复用分类并刷新元数据；`nideyongbao/*` 自动归入
`Personal-Repositories` 且不调用 LLM，其他新增仓库才会调用 LLM。

重新尝试分类 `other` 条目：

```powershell
python main.py --reclassify-other
```

一次有超过 20% 仓库不可见时会触发保护。只有确认 Token 权限和 GitHub Star
列表正常后才使用：

```powershell
python main.py --allow-large-removal
```

## 成功产物

- `stars_cache.json`：当前可见的活动 Star 快照。
- `removed_stars.json`：从活动集合消失的仓库审计记录。
- `README.md`：按 Vault 主分类和主题子分类生成的目录。

个人仓库条目使用 `classification_source: owner_rule`，原技术主题保存在
`subject_category_id`、`subject_category` 和 `subject_vault_category`。

## 人工分类修订

需要固定单个仓库的分类时，在 `categories.json.repository_overrides` 中按 GitHub
仓库数字 ID 添加 `name`、`category_id`、`reasoning` 和 `classified_at`。同步时
该规则优先于缓存和 LLM，生成条目标记为
`classification_source: manual_override`。
