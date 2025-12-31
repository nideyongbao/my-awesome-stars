import os
import json
import time
import github
from github import Github
from openai import OpenAI
from tqdm import tqdm

# --- 配置部分 ---
GITHUB_TOKEN = os.getenv("GH_TOKEN")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# --- 默认分类体系 (根据 docs/default_catrgories.md 设计) ---
DEFAULT_CATEGORIES = [
    # AI System (核心关注区)
    "AI-Sys-Train (分布式训练框架, DeepSpeed, Megatron-LM)",
    "AI-Sys-Inference (推理引擎与后端, vLLM, TGI, TensorRT-LLM)",
    "AI-Sys-Compiler (编译器与图优化, TVM, MLIR, Triton)",
    "AI-Sys-Device (异构计算与硬件接口, CUDA, ROCm)",
    "AI-Sys-Ops (MLOps, 实验管理, 模型监控)",
    # AI Algorithm & Models
    "AI-Algo-LLM (语言模型架构与微调, Llama, Qwen, LoRA)",
    "AI-Algo-Vision (计算机视觉与生成, Stable Diffusion, YOLO)",
    "AI-Algo-Audio (语音识别与合成, Whisper, TTS)",
    "AI-Algo-Multi (多模态与新架构, CLIP, Mamba, MoE)",
    "AI-Algo-Theory (纯理论代码, 论文复现, 数学库)",
    # AI Engineering & Application
    "AI-App-Agent (智能体, 规划与记忆, AutoGPT, MetaGPT)",
    "AI-App-RAG (检索增强生成与向量库, LangChain, LlamaIndex)",
    "AI-App-Framework (应用开发框架, Dify, Flowise)",
    "AI-Data-Eng (数据处理, ETL, 标注工具)",
    # General Development
    "Dev-Web-FullStack (现代Web开发, Next.js, React, FastAPI)",
    "Dev-Infra-Cloud (云原生, 容器, K8s)",
    "Dev-DB-Storage (数据库与存储, PostgreSQL, Redis)",
    "Dev-Lang-Core (编程语言核心资源, Rust, Python, C++)",
    "Dev-Sec (安全工具与逆向工程)",
    # Tools & Misc
    "Tools-Efficiency (生产力与终端工具, Oh-My-Zsh, Raycast)",
    "Tools-Media (图像视频处理工具, FFmpeg)",
    "Proj-RedLoop (RedLoop相关项目)",
    "CS-Education (教程, 面试, 路线图)",
    "Uncategorized (无法分类)"
]

CACHE_FILE = "stars_cache.json"
CATEGORY_FILE = "categories.json"

# --- Prompt 模板 (JSON 输出 + 思维链) ---
PROMPT_TEMPLATE = """
你是一个资深的 GitHub 仓库分类专家。你的任务是将给定的仓库归类到最合适的类别中。

### 输入信息
- 仓库名: {repo_name}
- 描述: {description}
- 主要语言/Topics: {topics}

### 预设分类体系 (Name: Description)
{categories_json}

### 决策逻辑
1. **优先匹配**：首先尝试从[预设分类体系]中寻找最匹配的类别。
   - 如果是底层算子、CUDA优化，必须选 `AI-Sys-` 开头的类别。
   - 如果是 Agent 或 RAG 相关，优先选 `AI-App-` 开头的类别。
2. **新建分类**：只有当[预设分类体系]中**完全没有**合适的类别时（例如遇到了区块链、量子计算等新领域），才允许新建分类。
   - 新分类格式必须为：`New-Category-Name (简短中文描述)`。
   - 例如：`Tech-Blockchain (区块链与Web3)`。
   - 严禁创建与现有体系重叠的分类（例如不要创建 `Web-Frontend`，因为已有 `Dev-Web-FullStack`）。

### 输出格式 (必须是纯 JSON)
请仅输出一个 JSON 对象，不要包含 Markdown 标记或其他文本：
{{
    "category": "分类名称 (描述)",
    "confidence": "high/medium/low",
    "reasoning": "简短的分类理由（10个字以内）"
}}
"""


def load_categories():
    """加载分类体系，如果有动态扩展的分类则合并"""
    if os.path.exists(CATEGORY_FILE):
        with open(CATEGORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return DEFAULT_CATEGORIES.copy()


def save_categories(categories):
    """保存分类体系（包含动态扩展的新分类）"""
    with open(CATEGORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)


def get_llm_classification(repo_name, description, topics, current_categories):
    """调用 LLM 进行分类，返回 JSON 结果"""
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    prompt = PROMPT_TEMPLATE.format(
        repo_name=repo_name,
        description=description,
        topics=", ".join(topics) if topics else "N/A",
        categories_json=json.dumps(current_categories, ensure_ascii=False, indent=2)
    )

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},  # 强制 JSON 模式
            temperature=0.1
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        if "429" in str(e):
            print(f"⚠️ LLM Rate Limit hit for {repo_name}. Sleeping for 60s...")
            time.sleep(60)
        else:
            print(f"LLM Error: {e}")
        return {"category": "Uncategorized", "confidence": "low", "reasoning": "API Error"}


def update_readme(data, categories):
    """生成 README.md"""
    # 动态收集所有分类
    all_categories = set()
    for repo in data.values():
        all_categories.add(repo['category'])

    # 排序：优先 categories 列表顺序，新分类按字母序，Uncategorized 最后
    sorted_cats = []
    seen = set()

    for cat in categories:
        if cat in all_categories:
            sorted_cats.append(cat)
            seen.add(cat)

    remaining = [c for c in all_categories if c not in seen and c != "Uncategorized"]
    remaining.sort()
    sorted_cats.extend(remaining)

    if "Uncategorized" in all_categories:
        sorted_cats.append("Uncategorized")

    # 分组
    grouped = {cat: [] for cat in sorted_cats}
    for repo in data.values():
        cat = repo['category']
        if cat in grouped:
            grouped[cat].append(repo)
        else:
            if "Uncategorized" not in grouped:
                grouped["Uncategorized"] = []
            grouped["Uncategorized"].append(repo)

    # 生成内容
    md = "# 🌟 My Awesome AI Stars\n\n> 🤖 自动生成于 GitHub Actions, Powered by LLM.\n\n"
    md += "## 目录\n"
    for cat in sorted_cats:
        cat_key = cat.split(" ")[0]
        if " " not in cat:
            cat_key = cat
        count = len(grouped[cat])
        md += f"- [{cat} ({count})](#{cat_key.lower()})\n"

    md += "\n---\n"

    for cat in sorted_cats:
        repos = grouped[cat]
        if not repos:
            continue

        cat_key = cat.split(" ")[0]
        if " " not in cat:
            cat_key = cat

        md += f"## <span id='{cat_key.lower()}'>{cat}</span>\n\n"
        md += "| Project | Description | Stars | Language |\n"
        md += "|---|---|---|---|\n"
        repos.sort(key=lambda x: x['stars'], reverse=True)
        for r in repos:
            desc = (r.get('description') or "").replace("|", r"\|").replace("\n", " ")
            lang = r.get('language') or "N/A"
            md += f"| [{r['name']}]({r['url']}) | {desc[:100]} | {r['stars']} | {lang} |\n"
        md += "\n"

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(md)


def main():
    # 1. 加载分类体系
    categories = load_categories()

    # 2. 读取缓存
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    else:
        cache = {}

    # 3. 获取 GitHub Stars (使用新版 Auth)
    auth = github.Auth.Token(GITHUB_TOKEN)
    g = Github(auth=auth)
    user = g.get_user()
    print(f"Fetching stars for user: {user.login}...")

    starred_repos = user.get_starred()

    # 4. 增量更新逻辑
    new_cache = {}

    for repo in tqdm(starred_repos, total=starred_repos.totalCount):
        repo_id = str(repo.id)

        # 如果缓存里有，且不是 Uncategorized，直接复用
        if repo_id in cache and cache[repo_id].get('category') != 'Uncategorized':
            cache[repo_id]['stars'] = repo.stargazers_count
            cache[repo_id]['language'] = repo.language
            new_cache[repo_id] = cache[repo_id]
        else:
            # 新仓库或需要重新分类
            print(f"🤖 Classifying: {repo.full_name}")

            # 获取 Topics
            try:
                topics = repo.get_topics()
            except Exception:
                topics = []

            result = get_llm_classification(
                repo.name,
                repo.description or "",
                topics,
                categories
            )

            category_name = result.get("category", "Uncategorized")

            # --- 动态扩展逻辑 ---
            if category_name not in categories and "(" in category_name:
                print(f"✨ 发现新领域，自动扩展分类体系: {category_name}")
                categories.append(category_name)
                save_categories(categories)

            entry = {
                "name": repo.full_name,
                "url": repo.html_url,
                "description": repo.description,
                "stars": repo.stargazers_count,
                "category": category_name,
                "language": repo.language,
                "topics": topics,
                "confidence": result.get("confidence", "unknown"),
                "reasoning": result.get("reasoning", ""),
                "crawled_at": time.time()
            }
            new_cache[repo_id] = entry
            time.sleep(1)  # 避免 LLM Rate Limit

    # 5. 保存缓存
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(new_cache, f, ensure_ascii=False, indent=2)

    # 6. 保存分类体系
    save_categories(categories)

    # 7. 生成 README
    update_readme(new_cache, categories)
    print("Done!")


if __name__ == "__main__":
    main()
