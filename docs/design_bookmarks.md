# Auto-Smart-Bookmarks 系统设计方案

## 1. 核心痛点
与 GitHub Stars 不同，网页书签面临更复杂的问题：
1.  **信息密度低**: 只有一个 URL 和标题，缺乏描述（GitHub Repo 好歹有 Description）。
2.  **分类维度混杂**: 有的是工具，有的是文章，有的是视频。
3.  **层级僵化**: 浏览器文件夹很难表达“既是 AI 又是前端”的多维属性。

## 2. 优雅的架构设计: "The Librarian" (图书管理员模式)

我们不只要“分类”，而是要“重组”。

### 架构图
```mermaid
graph TD
    A[浏览器导出 bookmarks.html] --> B(Python 解析器)
    B --> C{URL 预处理}
    C -->|获取 Meta Info| D[抓取 Title/Desc/Keywords]
    D --> E[LLM 智能归档]
    E -- 动态决策体系 --> F[生成知识图谱]
    F --> G1[输出: 整理后的 bookmarks_clean.html (供浏览器导入)]
    F --> G2[输出: Obsidian/Notion 知识库 (Markdown)]
```

### 3. 核心创新点

#### 3.1 上下文增强 (Context Enrichment)
由于 URL 本身信息太少，Python 脚本必须对每个 URL 发起一个轻量级请求 (HEAD 或 GET)，利用 `BeautifulSoup` 提取：
- `<title>`
- `<meta name="description">`
- `<meta name="keywords">`
这相当于给 LLM “喂料”，让它知道链接里到底是什么。

#### 3.2 动态多维分类 (Dynamic Multi-dimensional Taxonomy)
为了解决“扩展性”问题，我们放弃单一的文件夹结构，采用 **"Path + Tags"** 策略。

prompt 设计思路：
> "这个链接是关于 Rust 语言编写的高性能 Web 服务器教程。
> 请给出：
> 1.  **物理路径**: `Development/Backend/Rust` (用于浏览器文件夹，唯一)
> 2.  **逻辑标签**: `#Tutorial`, `#HighPerformance`, `#Web` (用于知识库，无限扩展)
> 3.  **一句话推荐**: '通过构建 Web Server 学习 Rust 的实战教程' (改写原来的标题)"

#### 3.3 "Inbox Zero" 理念
系统将书签分为两个区域：
- **Archive (已归档)**: `stars_cache.json` 类似逻辑，已经分好类的。
- **Inbox (新收草稿)**: 每次脚本运行，只处理新加入的书签，处理完自动移入 Archive。

## 4. 输出形态方案

### 方案 A: 回归浏览器 (Browser Loop)
生成一个新的 `cleaned_bookmarks.html`。
- 用户清空浏览器书签。
- 导入这个新文件。
- **效果**: 也就是你的书签栏会自动变整齐，所有乱七八糟的链接都被移动到了正确的文件夹里。

### 方案 B: 知识库化 (Knowledge Base)
生成 Markdown 文件树：
- `Knowledge/AI/LLM/DeepSeek_Guide.md`
- 文件内容包含 URL、LLM 生成的摘要、以及 Tags。
- 适合配合 Obsidian 使用。

## 5. 可行性验证 (MVP 代码片段)

我们需要引入 `requests` 和 `beautifulsoup4`。

```python
import requests
from bs4 import BeautifulSoup

def enrich_url_info(url):
    try:
        resp = requests.get(url, timeout=5)
        soup = BeautifulSoup(resp.text, 'html.parser')
        title = soup.title.string if soup.title else ""
        desc = ""
        meta = soup.find('meta', attrs={'name': 'description'})
        if meta:
            desc = meta.get('content')
        return f"Title: {title}\nDesc: {desc}"
    except:
        return "Info fetch failed"

# LLM Prompt 示例
prompt = """
分析这个网页信息：
{enriched_info}

请输出 JSON 格式：
{
    "folder_path": "Tech/AI/LLM",  // 动态生成，如果没有合适的就新建
    "tags": ["Tutorial", "Python"],
    "new_title": "DeepSeek 接入实战指南" // 优化后的标题
}
"""
```

## 6. 总结
这个方案的优雅之处在于：**它把“死”的书签变成了“活”的知识**。
如果不只是想整理收藏夹，而是想构建个人知识库，方案 B 是最佳选择；如果只是想让浏览器清爽，方案 A 足够完美。
