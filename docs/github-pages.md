# GitHub Pages 网站

网站地址：<https://nideyongbao.github.io/my-awesome-stars/>

网页使用现有 `stars_cache.json` 和 `categories.json`，提供主分类、细分类、语言筛选、全文关键词搜索，以及 Stars、最近收录、最近推送和名称排序。筛选条件保存在网址参数中，可以复制分享。最近收录按本仓库首次记录时间排序；Stars 和推送时间来自最近同步快照。

## 首次部署

1. 在仓库 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。Custom domain 留空，不需要点击 Jekyll 或 Static HTML 模板的 Configure。
2. 将网站代码和 `.github/workflows/pages.yml` 提交并推送到远端 `main`。如果使用分支或 PR，先合并到 `main`。
3. 打开仓库 **Actions → Deploy GitHub Pages**，查看自动触发的任务。`build` 负责测试与生成，`deploy` 负责发布。
4. 两个任务成功后，打开上方网站地址。Settings → Pages 也会显示访问链接，首次生效可能需要等待几分钟。

首次部署只读取已经提交的数据，不访问 GitHub Stars API 或分类 API，因此不需要新增 Secret。

需要手动发布时，在 **Actions → Deploy GitHub Pages → Run workflow → Branch: main → Run workflow** 运行即可。此按钮要求工作流文件已经存在于默认分支。

## 后续如何自动更新

```text
Update Awesome Stars（现有定时或手动更新）
  → 更新数据与 README，并推送到 main

Deploy GitHub Pages（每天北京时间 12:00，UTC 04:00）
  → 读取最新 main 数据、生成 _site/、发布网站
```

网页源文件、构建脚本、测试或数据手动推送到 `main` 时，也会触发发布。PR 只运行测试与构建，不发布。

Pages 的定时配置为 `0 4 * * *`（UTC），即北京时间每天中午 12:00，给此前的数据更新留出时间。GitHub 的定时调度可能延迟，不能保证精确在 12:00 开始或完成。

数据更新与网站发布是两个独立的定时任务，不再监听更新任务结束后立即发布。构建时检出最新 `main`。如果当天数据更新失败或尚未完成，12:00 的发布使用最近一次已提交的快照；网站构建或部署失败时保留线上旧版本。`GITHUB_TOKEN` 自动推送的提交不会触发另一个普通 `push` 工作流，Pages 的每日发布由自己的 `schedule` 触发。

## 本地预览

需要 Python 3.10+，网站构建不需要第三方 Python 包、Node 构建工具或 API 密钥。

```sh
python -m unittest -v test_main.py test_site.py
python build_site.py
python -m http.server 8000 --directory _site
```

访问 <http://localhost:8000/>。使用 Ctrl+C 停止服务器。也可以直接打开 `_site/index.html`；数据已经嵌入页面，无需联网获取。

## 文件分工

| 文件 | 用途 |
| --- | --- |
| `site/index.html` | 页面模板与站点名称 |
| `site/style.css` | 页面样式与移动端布局 |
| `site/app.js` | 搜索、筛选、排序和网址参数 |
| `build_site.py` | 从快照生成网站，转义外部文本，仅输出展示字段 |
| `.github/workflows/pages.yml` | 自动构建和部署 |
| `_site/` | 生成结果，已加入 `.gitignore`，无需提交 |

样式和脚本使用 `./` 相对路径，可在 `/my-awesome-stars/` 下正常加载。只有 `_site/` 中的网页文件会作为 Pages 产物上传，不会发布整个仓库。

README 中的网页入口由 `main.py` 生成，因此自动更新后也会保留。

## 排查部署问题

- **Actions 中没有 Deploy GitHub Pages**：确认 `pages.yml` 已推送到远端默认分支 `main`。
- **部署提示 Pages 未开启**：回到 Settings → Pages 确认 Source 为 GitHub Actions。
- **部署等待审批或被分支规则拦截**：查看 Settings → Environments → github-pages 的审批和部署分支限制，确保允许 `main`。
- **网页显示旧内容**：检查最新数据更新和 Deploy GitHub Pages 是否成功，并刷新浏览器；数据更新失败或尚未完成时，网站使用最近一次已提交的快照。
- **出现 404**：确认首次部署成功，网址包含 `/my-awesome-stars/`；若刚部署完成，稍后重试。

参考：[GitHub Pages 自定义工作流](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)。
