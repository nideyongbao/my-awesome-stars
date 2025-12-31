# 首次运行指南 (Run Guide)

由于 GitHub Actions 的安全机制，新的 Workflow 文件上传后**不会自动运行**（除非配置了 push 触发器，但本仓库配置的是 schedule 和 workflow_dispatch）。您必须手动触发第一次运行。

## 步骤

1.  **打开 Actions 页面**
    进入您的仓库页面，点击顶部的 **Actions** 标签。
    URL: `https://github.com/nideyongbao/my-awesome-stars/actions`

2.  **选择 Workflow**
    在左侧列表中，点击 **Update Awesome Stars**。
    > 注意：不要停留在 "All workflows" 页面，一定要点击具体的 workflow 名称。

3.  **触发运行**
    - 在右侧页面，您会看到一个蓝色的横幅或按钮：**Run workflow**。
    - 点击它，会弹出一个配置框，保持默认分支 (`Branch: main`) 不变。
    - 再次点击绿色的 **Run workflow** 按钮。

4.  **等待结果**
    - 页面会自动刷新，您会看到一个新的运行记录（黄色圆点 🟡 表示正在运行）。
    - 运行完成后，它会变成绿色对勾 ✅。

## 运行后检查
运行成功后，回到 **Code** 标签页：
- `README.md`: 应该已经被更新，展示了分类后的 Star 列表。
- `stars_cache.json`: 应该已经被创建/更新，包含了已抓取的仓库数据。

## 下次运行
- **自动**: 系统会在每天 **UTC 0:00** (北京时间 8:00) 自动运行。
- **手动**: 随时可以按照上述步骤手动触发更新。
